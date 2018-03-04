import os, json, torch, lmdb, pickle
import torch.utils.data as data
import numpy as np 
import pdb

class SVALoader(data.Dataset):
    def __init__(self, qa_path, loader, training, max_seq_len=25, seed=1234,
                transform=None, target_transform=None):
        super(SVALoader, self).__init__()
        self.loader = loader
        self.training = training

        # concat two datasets
        qa_paths = qa_path.split(',')
        print("QA data paths:{}".format(qa_paths))
        self.qa_list = []
        for path in qa_paths:
            self.qa_list += pickle.load(open(path))

    def __getitem__(self, index):

        bdata = self.qa_list[index]
        ifeature = pickle.loads(self.loader(bdata['img_path'])).toarray()
        qtoken = torch.LongTensor(bdata['ques'])
        qmask = torch.Tensor(bdata['qmask'])
        qid = bdata['qid']

        label = np.random.choice(bdata['ans_cans'], p=bdata['ans_p']) if self.training else 0 
        label = int(label)
        
        return qid, ifeature, qtoken, qmask, label

    def __len__(self):
        return len(self.qa_list)
