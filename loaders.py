import mxnet as mx
import numpy as np
import numpy.matlib
import numpy.random
import os
import json, pickle
import random
import logging
import lmdb
import time

class SimpleBatch(object):
  def __init__(self, data_names, data, label_names, label, 
               bucket_key=None, qid=None, ans_all=None, splits=None):
    self.data = data
    self.label = label
    self.data_names = data_names
    self.label_names = label_names
    self.bucket_key = bucket_key

    self.pad = 0
    self.splits=splits # a list of splits
    self.index = None
    self.qid = qid # should be a list of qid's
    self.ans_all = ans_all

    self.provide_data = [(n, x.shape) for n,x in zip(self.data_names, self.data)]
    self.provide_label = [(n, x.shape) for n,x in zip(self.label_names, self.label)]

class VQAIter(mx.io.DataIter):
    def __init__(self, qa_path, lmdb_path, batch_size, 
                 max_seq_len=26, sent_gru_hsize=2400,  
                 is_train=True, net=None, w=14, h=14):
        """
        Data loader for the VQA dataset.abs

        qa_path: path to the question-answer file
        lmdb_path: the LMDB storing the extracted features
        net: symbol of the network, to print its size
        is_train: use answer sampling if set to True
        """
        super(VQAIter, self).__init__()
        random.seed(1234)
        qa_paths = qa_path.split(',')
        logging.info("QA data paths:{}".format(qa_paths))
        env = lmdb.open(lmdb_path, readonly=True)
        self.txn = env.begin()
        self.batch_size = batch_size
        self.is_train=is_train

        # whether to use snake-shaped image data
        self.provide_data = [('img_feature', (batch_size, w*h, 2048)),
                            ('sent_seq', (batch_size, max_seq_len)),
                            ('mask', (batch_size, max_seq_len)),
                            ('sent_l0_init_h', (batch_size, sent_gru_hsize)),
                            ('horizontal_zeros', (batch_size, 1,1,w)),
                            ('vertical_zeros', (batch_size, 1,h,1))]
        
        self.provide_label = [('ans_label', (batch_size,))]

        self.data_names = [t[0] for t in self.provide_data]
        self.label_names = [t[0] for t in self.provide_label]
        self.data_buffer = [np.zeros(t[1], dtype=np.float32) for t in self.provide_data]
        self.label_buffer = [np.zeros(t[1], dtype=np.float32) for t in self.provide_label]
        
        self.qa_list = []
        for path in qa_paths:
            self.qa_list += pickle.load(open(path))

        # print self.provide_data
        if net is not None:
            shape_list = net.infer_shape(**dict(self.provide_data+self.provide_label))
            arg_names = net.list_arguments()
            n_params = 0
            logging.info("Number of parameters:")
            for n, shape in enumerate(shape_list[0]):
                if arg_names[n] not in self.data_names and arg_names[n] not in self.label_names:
                    logging.info("%s: %d, i.e., %.2f M params", arg_names[n], np.prod(shape), np.prod(shape)/1e6)
                    n_params += np.prod(shape)
            logging.info("Total number of parameters:%d, i.e., %.2f M params", n_params, n_params/1e6)

        self.n_total = len(self.qa_list)
        self.reset()

    def reset(self):
        if self.is_train:
            logging.info("Shuffling data...")
            random.shuffle(self.qa_list)

    def __iter__(self):
        candidate_ans = np.zeros((self.batch_size, 10), dtype=np.int32)

        for curr_idx in range(0, self.n_total-self.batch_size+1, self.batch_size):
            qid_list=[]
            for bidx in range(self.batch_size):
                bdata = self.qa_list[bidx+curr_idx]

                self.data_buffer[0][bidx,:,:] = pickle.loads(self.txn.get(bdata['img_path'])).toarray()
                self.data_buffer[1][bidx, :] = bdata['ques']
                self.data_buffer[2][bidx, :] = bdata['qmask']
                
                qid_list.append(bdata['qid'])
                if self.is_train:
                    self.label_buffer[0][bidx] = np.random.choice(bdata['ans_cans'], p=bdata['ans_p'])
                if not self.is_train and len(bdata['ans_all'])>0:
                    # for VQA scoring
                    candidate_ans[bidx] = bdata['ans_all']

            yield SimpleBatch(self.data_names, [mx.nd.array(arr) for arr in self.data_buffer],
                              self.label_names, [mx.nd.array(arr) for arr in self.label_buffer], 
                              qid=qid_list, ans_all=candidate_ans)

        # check if need to add an incomplete batch at validation
        if not self.is_train and curr_idx < self.n_total-self.batch_size:
            curr_idx += self.batch_size
            last_batch_size = n_total - curr_idx
            print("last_batch_size {}".format(last_batch_size))
            candidate_ans = np.zeros((last_batch_size, 10), dtype=np.int32)
            # change the shape of buffer files
            data_buffer=[np.zeros([last_batch_size]+list(shape[1:])) for name, shape in self.provide_data]
            label_buffer=[np.zeros([last_batch_size]+list(shape[1:])) for name, shape in self.provide_label]
            qid_list=[]
            for bidx in range(last_batch_size):
                bdata = self.qa_list[bidx+curr_idx]
                data_buffer[0][bidx,:,:] = pickle.loads(self.txn.get(bdata['img_path'])).toarray()
                data_buffer[1][bidx, :] = bdata['ques']
                data_buffer[2][bidx, :] = bdata['qmask']
                
                qid_list.append(bdata['qid'])
                if len(bdata['ans_all'])>0:
                    label_buffer[0][bidx] = np.random.choice(bdata['ans_cans'], p=bdata['ans_p'])
                    candidate_ans[bidx] = bdata['ans_all']

            yield SimpleBatch(self.data_names, [mx.nd.array(arr) for arr in data_buffer],
                              self.label_names, [mx.nd.array(arr) for arr in label_buffer], 
                              qid=qid_list, ans_all=candidate_ans)
