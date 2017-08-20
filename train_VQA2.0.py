import mxnet as mx
import numpy as np
import os
import argparse
import pickle
from symbols import *
from loaders import *

parser = argparse.ArgumentParser(description='Structured Visual Attention on the VQA dataset')
parser.add_argument('--qa-path', type=str, 
                    default='/rundata/vqa2.0/train+val+vg_train_sampling.pkl',
                    help='path to the questions ans answers, comma without space for separation')
parser.add_argument('--lmdb-path', type=str, default='/rundata/coco_res152_lmdb')
parser.add_argument('--skip-thought-dict', type=str, default='/rundata/vqa1.0/skip-argdict.pkl',
                    help='initialize the GRU for skip-thought vector')
parser.add_argument('--chk-path', type=str, default='/hdd1/chk/vqa-sva',
                    help='path to store the checkpoints')
parser.add_argument('--gpus', type=str, default='0',
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
parser.add_argument('--lr-factor', type=float, default=0.25,
                    help='lr decay factor')

parser.add_argument('--begin-epoch', type=int, default=0,
                    help='which epoch to begin. if >0, then load from checkpoints')
parser.add_argument('--num-epoch', type=int, default=40,
                    help='total epoches')
parser.add_argument('--print-every', type=int, default=25,
                    help='print training stats every # updates')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')

args = parser.parse_args()
chk_path = args.chk_path

import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logging.info('starting training with arguments %s', args)

mx.random.seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.chk_path):
    os.makedirs(chk_path)
if chk_path[-1] != '/':
    chk_path += '/model'
else:
    chk_path += 'model'

ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

net = MF_accelerate(args.batch_size, is_train=True, seq_len=25, general_dp=args.gdp, qemb_dp=args.qdp,
                    crf_iter=args.crf_iter, common_embed_size=args.cdm, epot_common_dim=args.cdm,
                    n_gpus=len(ctx), w=14, h=14, idim=2048, n_ans=2000)

train_iter = VQAIter(qa_path=args.qa_path,
                     lmdb_path=args.lmdb_path,
                     batch_size=args.batch_size, 
                     is_train=True, max_seq_len=25,
                     net=net, seed=args.seed)

if args.lr_factor_epoch>0:
    step = args.lr_factor_epoch*(train_iter.n_total // args.batch_size)
else:
    step=1
opt_args = {}
opt_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(step=step, factor=args.lr_factor)
optimizer = mx.optimizer.Adam(learning_rate=args.lr, beta1=args.a1, beta2=args.a2, wd=args.wd, **opt_args)

model = mx.mod.Module(context=ctx, symbol=net, data_names=train_iter.data_names,
                      label_names=train_iter.label_names)

if args.begin_epoch>0:
    _, arg_params, __ = mx.model.load_checkpoint(chk_path, args.begin_epoch)
else:
    # containing only the skip thought weights
    arg_params = pickle.load(open(args.skip_thought_dict))

initializer = mx.initializer.Load(arg_params, 
                                  default_init=mx.initializer.Uniform(args.uni_mag), 
                                  verbose=True)

def top1_accuracy(labels, preds):
    pred_labels = np.argmax(preds, axis=1)
    n_correct = np.where(labels==pred_labels)[0].size
    return n_correct/np.float32(labels.size)

metrics = [mx.metric.CrossEntropy(),mx.metric.CustomMetric(top1_accuracy, allow_extra_outputs=True)]
epoch_end_callback = [mx.callback.do_checkpoint(chk_path, 1)]#, test_callback]
batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.print_every)]

model.fit(train_data=train_iter,
          eval_metric=mx.metric.CompositeEvalMetric(metrics=metrics),
          epoch_end_callback=epoch_end_callback,
          batch_end_callback=batch_end_callback,
          optimizer=optimizer,
          initializer=initializer,
          begin_epoch=args.begin_epoch,
          num_epoch=args.num_epoch)
