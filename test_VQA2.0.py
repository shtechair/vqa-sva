import mxnet as mx
import numpy as np
import pickle, os
from symbols import *
from loaders import *
import json
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data-root', type=str, default='/rundata/vqa2.0')
parser.add_argument('--lmdb-path', type=str, default='/rundata/coco_res152_lmdb')
parser.add_argument('--test-fname', type=str, default='test2017.pkl')
parser.add_argument('--model-root', '-mr', type=str, default='/hdd1/chk/vqa2.0',
                   help='model root dir')
parser.add_argument('--model-setting', '-ms', type=str, default='vqa-sva',
                   help='model setting, I use them to discriminate the models')
parser.add_argument('--gpu', type=int, default=0,
                   help='epoch')
parser.add_argument('--crf-iter', type=int, default=3,
                   help='crf iterations')
parser.add_argument('--cdm', type=int, default=1200,
                   help='commen embedding dimensions')
parser.add_argument('--be', type=int, default=39,
                   help='beginning epoch')
parser.add_argument('--ee', type=int, default=40,
                   help='beginning epoch')
parser.add_argument('--se', type=int, default=1,
                   help='step of epochs')
parser.add_argument('--bsize', type=int, default=256,
                   help='batch size')
args=parser.parse_args()
model_setting = args.model_setting
begin_epoch, end_epoch, step_epoch = args.be, args.ee, args.se

ctx=mx.gpu(args.gpu)

chk_path = os.path.join(args.model_root, model_setting, 'model')

answer_to_index = pickle.load(open(os.path.join(args.data_root, 'a2ix_top2000.pkl')))
ix2a = {val:key for key, val in answer_to_index.items()}

test_net = MF_accelerate(args.bsize, is_train=False, seq_len=25, general_dp=0, qemb_dp=0,
                         crf_iter=args.crf_iter, common_embed_size=args.cdm, epot_common_dim=args.cdm,
                         n_gpus=1, w=14, h=14, idim=2048, n_ans=2000)

# the parsed test data
test_iter = VQAIter(qa_path=os.path.join(args.data_root, args.test_fname),
                    lmdb_path=args.lmdb_path,
                    batch_size=args.bsize,
                    max_seq_len=25,
                    is_train=False)
n_total = test_iter.n_total
out_path = 'test_results'
if not os.path.exists(out_path):
    os.makedirs(out_path)

import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logging.info('Start evaluating models at {}'.format(chk_path))
for epoch in range(begin_epoch, end_epoch, step_epoch):
    _,arg_params, __ = mx.model.load_checkpoint(chk_path, epoch)

    outfile = os.path.join(out_path, '%s-epoch%d.json'%(model_setting, epoch))
    
    initialized=False
    ans_list=[]
    counter=0
    valid_args = test_net.list_arguments()
    for batch in test_iter:
        if not initialized:
          input_shape = {name:shape for name, shape in batch.provide_data+batch.provide_label if name in valid_args}
          exe=test_net.simple_bind(ctx, grad_req='null', **input_shape)
          # copy the weights
          for key in exe.arg_dict.keys():
              if key in arg_params:
                  arg_params[key].copyto(exe.arg_dict[key])

          initialized=True

        # testing the last batch which may not have the same batch_size, so need to reshape
        if test_iter.last_batch_size is not None:
            input_shape = {name:shape for name, shape in batch.provide_data+batch.provide_label}
            exe=exe.reshape(**input_shape)
            # copy the weights
            for key in exe.arg_dict.keys():
                if key in arg_params:
                    arg_params[key].copyto(exe.arg_dict[key])

        for name, val in zip(batch.data_names+batch.label_names, batch.data+batch.label):
            val.copyto(exe.arg_dict[name])

        exe.forward()
        ans_sm=exe.outputs[0].asnumpy()
        ans_arr = np.argmax(ans_sm, axis=1)
        ans_list += [{'question_id':int(qid), 'answer':ix2a[ans]} 
                      for qid, ans in zip(batch.qid, list(ans_arr))]
        counter+=1
        if counter%10==0:
          logging.info("Evaluated {}/{} questions.".format(counter*args.bsize, n_total))

    logging.info('Number of evaluated questions: {}'.format(len(ans_list)))
    json.dump(ans_list, open(outfile, 'w'))

