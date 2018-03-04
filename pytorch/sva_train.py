"""
Using GRU on MLB features + attention to encode the video sequences
"""
import argparse
import os, time, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import shutil
import time, collections
from PIL import Image
import lmdb, io
import pdb
from sva_loader import *
from sva_sym import *
import pickle

parser = argparse.ArgumentParser(description='PyTorch version of SVA')
parser.add_argument('--qa-path', type=str, 
                    default='/rundata/vqa2.0/train+val+vg_train_sampling.pkl',
                    help='path to the questions ans answers, comma without space for separation')
parser.add_argument('--val-qapath', type=str,
                    default='/rundata/vqa2.0/train+val_val_ansall.pkl')
parser.add_argument('--lmdb-path', type=str, default='/rundata/coco_res152_lmdb')
parser.add_argument('--skip-thought-dict', type=str, default='/rundata/vqa1.0/skip-argdict.pkl',
                    help='initialize the GRU for skip-thought vector')
parser.add_argument('--chk-path', type=str, default='/hdd1/chk/vqa-sva',
                    help='path to store the checkpoints')
parser.add_argument('--gpus', type=str, default='1',
                    help='which gpu(s) for training, e.g., 0,1,2,3')
parser.add_argument('--test-gpu', type=int, default=0,
                    help='use which GPU for testing')

parser.add_argument('--qdp', type=float, default=0.25,
                    help='dropout for Bayesian GRU')
parser.add_argument('--gdp', type=float, default=0.25,
                    help='general dropout ratio (for other layers)')
parser.add_argument('--cdm', type=int, default=1200,
                    help='common embedding dimension for MLB')
parser.add_argument('--crf-iter', type=int, default=3,
                    help='number of MF/LBP iterations')
parser.add_argument('--uni-mag', type=float, default=0.04,
                    help='magnitude of the random uniform initialization')
parser.add_argument('--clip', type=float, default=-1,
                    help='clip gradient')

parser.add_argument('--batch-size', type=int, default=300,
                    help='batch size')
parser.add_argument('--test-batch-size', type=int, default=256,
                    help='batch size for validation')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=0,
                    help='weight decay')
parser.add_argument('--a1', type=float, default=0.9,
                    help='parameter for Adam')
parser.add_argument('--a2', type=float, default=0.999,
                    help='parameter for Adam')
parser.add_argument('--lr-factor-epoch', type=int, default=13,
                    help='time the lr with a factor every x epoches')
parser.add_argument('--lr-factor', type=str, default='0.25',
                    help='lr decay factor')

parser.add_argument('--begin-epoch', type=int, default=0,
                    help='which epoch to begin. if >0, load from checkpoints')
parser.add_argument('--begin-save-epoch', type=int, default=0,
                    help='which epoch to save model.')
parser.add_argument('--begin-epoch-val', type=int, default=0,
                    help='which epoch to begin evaluation. if >0, evaluate begin from the epoch')
parser.add_argument('--num-epoch', type=int, default=40,
                    help='total epoches')
parser.add_argument('--print-every', type=int, default=25,
                    help='print training stats every # updates')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', default=25, type=int,
                    help='printing frequence')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='start epoch')

parser.add_argument('--val', default=False, action='store_true')
parser.add_argument('--val-batch-size', default=4, type=int,
                    help='batch size for validation')
parser.add_argument('--batch-multip', default=2, type=int,
                    help='multiply the batch size by this factor')

def lmdb_loader(env):
    txn = env.begin(write=False)

    def _loader(key):
        return txn.get(str(key))

    return _loader

def main():

    global args
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    args.lrf = eval(args.lr_factor)

    model = SVA(crf_iter=args.crf_iter, common_embed_size=args.cdm, epot_common_dim=args.cdm,
                 general_dp=args.gdp, qemb_dp=args.qdp, n_ans=2000, uni_mag=args.uni_mag)

    model = torch.nn.DataParallel(model, device_ids=[i for i in range(len(args.gpus.split(',')))]).cuda()

    lr_list = [{'params':model.module.parameters(), 'lr':args.lr}]

    criterion = nn.CrossEntropyLoss().cuda() #sm_loss(0, n_cls=2000)

    optimizer = torch.optim.Adam(lr_list, args.lr, weight_decay=args.wd)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    #cudnn.benchmark = True
    env = lmdb.open(args.lmdb_path, max_dbs=3)
    train_loader = torch.utils.data.DataLoader(
        SVALoader(
            qa_path=args.qa_path, 
            loader=lmdb_loader(env), 
            training=True, max_seq_len=25, seed=1234,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        SVALoader(
            qa_path=args.val_qapath, 
            loader=lmdb_loader(env), 
            training=False, max_seq_len=25, seed=1234,
        ),
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, 
        pin_memory=True)#, collate_fn=custom_collate)


    name2label = pickle.load(open(os.path.join('/'.join(args.qa_path.split('/')[:-1]), 'a2ix_top2000.pkl')))
    label2name = {int(val):key for key, val in name2label.items()}
    out_path = args.chk_path
    if args.val:
        ans_list, top3_score_list= validate(val_loader, model, criterion, label2name)
        json.dump(ans_list, open(os.path.join(out_path, 'ans', 'val_ans.json'), 'w'))
        json.dump(top3_score_list, open(os.path.join(out_path, 'scores', 'val_top3_score.json'), 'w'))
        return
    

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        os.makedirs(os.path.join(out_path, 'ans'))
        os.makedirs(os.path.join(out_path, 'scores'))
        os.makedirs(os.path.join(out_path, 'models'))

    lr_factor_iterations = args.lr_factor_epoch * len(train_loader) * args.batch_multip
    global_iters = args.start_epoch * len(train_loader)
    optimizer.zero_grad()
    for epoch in range(args.start_epoch, args.num_epoch):

        # train for one epoch
        global_iters=train(train_loader, model, criterion, optimizer,
              epoch, lr_factor_iterations, global_iters)

        # evaluate on validation set
        if epoch >= args.begin_epoch_val:
            ans_list, top3_score_list= validate(val_loader, model, criterion, label2name)
            json.dump(ans_list, open(os.path.join(out_path, 'ans', 'epoch%d_ans.json'%(epoch+1)), 'w'))
            json.dump(top3_score_list, open(os.path.join(out_path, 'scores', 'epoch%d_top3_score.json'%(epoch+1)), 'w'))
        
        # remember best prec@1 and save checkpoint
        if epoch > args.begin_save_epoch:
            out_fname = os.path.join(out_path, 'models', 'epoch-%04d.pth'%(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'best_prec1': best_prec1,
                # 'prec1': prec1,
                'optimizer' : optimizer.state_dict(),
            }, filename=out_fname)

def train(train_loader, model, criterion, optimizer, epoch,
          lr_factor_iterations, global_iters):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (qid, ifeat, qtoken, qmask, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end, ifeat.size(0))

        label = label.cuda(async=True)
        ifeat_var = Variable(ifeat)
        qtoken_var = Variable(qtoken)
        qmask_var = Variable(qmask)
        target_var = torch.autograd.Variable(label)

        # forward
        scores = model(ifeat_var, qtoken_var, qmask_var)

        # compute the loss graph
        # supervise on each glimpse
        loss_var = criterion(scores, target_var) / args.batch_multip
        loss_total = loss_var / args.batch_multip
        
        loss.update(loss_var.cpu().data[0], ifeat_var.size(0))
    
        prec1 = accuracy(scores, target_var)[0]
        top1.update(prec1.data.cpu()[0], ifeat_var.size(0))

        # compute gradient and do SGD step
        loss_total.backward()
        global_iters += 1
        if (global_iters)%args.batch_multip==0:
            if args.clip>0:
                nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        formatter = 'Date-time: {date}, {date_time}\t' + \
                    'Epoch: [{0}][{1}/{2}]\t' + \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' + \
                    'Loss {loss.val:.3f} ({loss.val:.3f})\t' + \
                    'prec1 {top1.val:.3f} ({top1.avg:.3f})\t'
        if i % args.print_freq == 0:
            print(formatter.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   date=time.strftime("%d:%m:%Y"),date_time=time.strftime("%H:%M:%S"),
                   data_time=data_time, loss=loss, top1=top1))

        if global_iters % lr_factor_iterations == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * args.lrf
                print('adjusted lr to {}'.format(param_group['lr']))

    return global_iters

def validate(val_loader, model, criterion, label2name):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    end_time = time.time()
    # {'question_id':int(qid), 'answer':ix2a[ans]} 
    ans_list = []
    top3_ans_list = [] # store the top scores for ensembling
    for i, (qid, ifeat, qtoken, qmask, _) in enumerate(val_loader):
        data_time.update(time.time()-end_time)
        #pdb.set_trace()
        ifeat_var = Variable(ifeat)
        qtoken_var = Variable(qtoken)
        qmask_var = Variable(qmask)
        #target_var = torch.autograd.Variable(label)

        scores = model(ifeat_var, qtoken_var, qmask_var)
        
        sm_score = F.softmax(scores)
        top3_ans_batch = topk_ans(sm_score, qid, label2name, topk=3, convert_sm=False)
        top3_ans_list += top3_ans_batch
        ans_list += [{'question_id': top3dict['question_id'], 'answer': top3dict['answer'][0]} 
                        for top3dict in top3_ans_batch]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        formatter = 'Date time: {date}, {date_time}\t' + \
                    'Test: [{0}/{1}]\t' + \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        if i % (args.print_freq*2) == 0:
            print(formatter.format(
                   i, len(val_loader), batch_time=batch_time,
                   date=time.strftime("%d/%m/%Y"),date_time=time.strftime("%H:%M:%S"),
                   data_time=data_time))
        end_time = time.time()

    print(formatter.format(
                i, len(val_loader), batch_time=batch_time,
                date=time.strftime("%d/%m/%Y"),date_time=time.strftime("%H:%M:%S"),
                data_time=data_time))
    sys.stdout.flush()

    return ans_list, top3_ans_list


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print("pred:{}, target:{}, correct:{}".format(pred, target, correct))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)

        if False:# and correct_k[0] == 0:
            print("k={}".format(k))
            for i in range(pred.size(1)):
                print("pred/target: {}/{}".format(pred[0, i], target[i]))
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('saved checkpoint to {}'.format(filename))
    if is_best:
        shutil.copyfile(filename, os.path.join(args.chk_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def topk_ans(output, qid, label2name, topk=3, convert_sm=True):
    """
    get the topk answers
    output: torch tensor
    """
    output_sm = torch.nn.functional.softmax(output) if convert_sm else output
    scores, pred = output_sm.data.topk(topk, 1, True, True)
    ans_scores = []
    for n in range(output.size(0)):
        answers = [label2name[l] for l in pred[n,:]]
        ps = [x for x in scores[n,:]]
        ans_scores.append({'question_id':qid[n], 'answer':answers, 'score':ps})
    return ans_scores

if __name__ == '__main__':
    main()
