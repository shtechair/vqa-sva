import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rnn_dropout import SequentialDropout
from initializer import xavier_initialier
import pdb

class AbstractGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False):
        super(AbstractGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

        # Modules
        self.weight_ir = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_ii = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_in = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_hr = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hi = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hn = nn.Linear(hidden_size, hidden_size, bias=bias_hh)

    def forward(self, x, hx=None):
        raise NotImplementedError

class BayesianGRUCell(AbstractGRUCell):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False,
                       dropout=0.25):
        super(BayesianGRUCell, self).__init__(input_size, hidden_size,
                                          bias_ih, bias_hh)
        self.set_dropout(dropout)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.drop_ir = SequentialDropout(p=dropout)
        self.drop_ii = SequentialDropout(p=dropout)
        self.drop_in = SequentialDropout(p=dropout)
        self.drop_hr = SequentialDropout(p=dropout)
        self.drop_hi = SequentialDropout(p=dropout)
        self.drop_hn = SequentialDropout(p=dropout)

    def end_of_sequence(self):
        self.drop_ir.end_of_sequence()
        self.drop_ii.end_of_sequence()
        self.drop_in.end_of_sequence()
        self.drop_hr.end_of_sequence()
        self.drop_hi.end_of_sequence()
        self.drop_hn.end_of_sequence()

    def forward(self, x, mask, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        x_ir = self.drop_ir(x)
        x_ii = self.drop_ii(x)
        x_in = self.drop_in(x)
        x_hr = self.drop_hr(hx)
        x_hi = self.drop_hi(hx)
        x_hn = self.drop_hn(hx)
        r = F.sigmoid(self.weight_ir(x_ir) + self.weight_hr(x_hr))
        i = F.sigmoid(self.weight_ii(x_ii) + self.weight_hi(x_hi))
        n = F.tanh(self.weight_in(x_in) + r * self.weight_hn(x_hn))

        hx_new = (1 - i) * n + i * hx

        hx = hx_new * mask + hx * (1-mask)

        return hx

class AbstractGRU(nn.Module):



    def __init__(self, input_size, hidden_size,

                       bias_ih=True, bias_hh=False):

        super(AbstractGRU, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.bias_ih = bias_ih

        self.bias_hh = bias_hh

        self._load_gru_cell()



    def _load_gru_cell(self):

        raise NotImplementedError



    def forward(self, x, hx=None, max_length=None):

        batch_size = x.size(0)

        seq_length = x.size(1)

        if max_length is None:

            max_length = seq_length

        output = []

        for i in range(max_length):

            hx = self.gru_cell(x[:,i,:], hx=hx)

            output.append(hx.view(batch_size, 1, self.hidden_size))

        output = torch.cat(output, 1)

        return output, hx


class BayesianGRU(AbstractGRU):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False,
                       dropout=0.25):
        self.dropout = dropout
        super(BayesianGRU, self).__init__(input_size, hidden_size,
                                          bias_ih, bias_hh)

    def _load_gru_cell(self):
        self.gru_cell = BayesianGRUCell(self.input_size, self.hidden_size,
                                        self.bias_ih, self.bias_hh,
                                        dropout=self.dropout)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.gru_cell.set_dropout(dropout)

    def forward(self, x, mask, hx=None, max_length=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        #output = []
        # for efficiency concerns?
        mask = mask.t().contiguous()
        for i in range(max_length):
            hx = self.gru_cell(x[:,i,:], mask[i].view(-1, 1), hx=hx)
            #output.append(hx.view(batch_size, 1, self.hidden_size))
        self.gru_cell.end_of_sequence()
        #output = torch.cat(output, 1)
        #return output, hx
        # only interested in the final vector
        return hx

class SVA(nn.Module):
    def __init__(self, crf_iter=3, common_embed_size=1200, epot_common_dim=1200,
                 general_dp=0.5, qemb_dp=0.25, n_ans=2000, idim=2048, qdim=2400, 
                 uni_mag=0.04):
        
        super(SVA, self).__init__()
        self.crf_iter = crf_iter
        self.general_dp = general_dp
        self.qemb_dp = qemb_dp
        self.cdm = common_embed_size
        self.edm = epot_common_dim
        self.idim = idim

        # question embedding
        self.wemb = nn.Embedding(15031, 620)
        self.ques_emb = BayesianGRU(620, qdim, dropout=qemb_dp)

        self.ifeature_dp = nn.Dropout(self.general_dp)
        self.vpot_visual_emb = nn.Sequential(
                                    nn.Conv2d(idim, common_embed_size, kernel_size=1),
                                    nn.Tanh())
        xavier_initialier(self.vpot_visual_emb[0], magnitude=uni_mag)
        # the vertical conv will share weights with it
        self.epot_visual_emb = nn.Sequential(
                                    nn.Conv2d(idim, epot_common_dim, kernel_size=(1,2)), 
                                    nn.Tanh())
        xavier_initialier(self.epot_visual_emb[0], magnitude=uni_mag)

        self.vpot_verbal_emb = nn.Sequential(
                                    nn.Dropout(general_dp),
                                    nn.Linear(qdim, common_embed_size),
                                    nn.Tanh())
        self.epot_verbal_emb = nn.Sequential(
                                    nn.Dropout(general_dp),
                                    nn.Linear(qdim, epot_common_dim),
                                    nn.Tanh())
        
        self.vpot_score = nn.Conv2d(common_embed_size, 1, 1, bias=False)
        self.epot_score = nn.Sequential(
                                    nn.Conv2d(epot_common_dim, 4, 1),
                                    nn.Tanh())

        self.unstratt_emb = nn.Sequential(
                                    nn.Dropout(general_dp),
                                    nn.Linear(idim, common_embed_size),
                                    nn.Tanh())
        self.stratt_emb = nn.Sequential(
                                    nn.Dropout(general_dp),
                                    nn.Linear(idim, common_embed_size),
                                    nn.Tanh())
        self.verbal_emb = nn.Sequential(
                                    nn.Dropout(general_dp),
                                    nn.Linear(qdim, common_embed_size*2),
                                    nn.Tanh())
                      
        self.out_score = nn.Sequential(
                                    nn.Dropout(general_dp),
                                    nn.Linear(2*common_embed_size, n_ans))

        # TODO: initialization
        

    def forward(self, ifeat, qtoken, mask):
        # ifeat: batch_size x c x h x w
        # qtoken: batch_size x qlen
        # maxlen: runtime parameter, sometimes the questions do not reach the maximum length
        q_wemb = self.wemb(qtoken)
        qemb = self.ques_emb(q_wemb, mask)
        qemb_dim = qemb.size(1)
        bsize = ifeat.size(0)
        
        if self.training and self.general_dp>0:
            ifeat = self.ifeature_dp(ifeat)
        if ifeat.dim()==3:
            # batch_size * (w*h) * dim
            h,w = 14,14
            ifeat = ifeat.view(-1,14, 14, self.idim).permute(0, 3, 1, 2).contiguous()
        else:
            # BCHW
            h, w = ifeat.size(2), ifeat.size(3)

        vpot_vis_emb = self.vpot_visual_emb(ifeat) # batch_size x cdm x h x w
        vpot_ver_emb = self.vpot_verbal_emb(qemb).view(-1, self.cdm, 1, 1)
        vpot_scores = self.vpot_score(vpot_vis_emb*vpot_ver_emb)
        vpot_sig = F.sigmoid(vpot_scores)

        epot_vis_h = self.epot_visual_emb(ifeat)
        epot_vis_v = F.tanh(F.conv2d(ifeat, weight=torch.transpose(self.epot_visual_emb[0].weight, 2,3), 
                                            bias=self.epot_visual_emb[0].bias)) # weight sharing
        epot_ver = self.epot_verbal_emb(qemb).view(-1, self.edm, 1, 1)
        
        # mean field iterations
        vpots_init = [1 - vpot_sig, vpot_sig]
        vpots_update = [1 - vpot_sig, vpot_sig]

        epot_h = self.epot_score(epot_vis_h * epot_ver)
        epot_v = self.epot_score(epot_vis_v * epot_ver)

        # for convenience of summation
        zeros_hor = Variable(torch.zeros(bsize, 1, 1, w)).cuda()
        zeros_ver = Variable(torch.zeros(bsize, 1, h, 1)).cuda()
        for ite in range(self.crf_iter):
            vpots_cat = torch.cat(vpots_update, 1)

            for z in range(2):
                epot_h_z = epot_h[:,z*2:(z+1)*2,:,:]
                s_from_left = torch.sum( epot_h_z * vpots_cat[:,:,:,:-1], 1, keepdim=True)
                s_from_left = torch.cat([zeros_ver, s_from_left], dim=3)
                s_from_right = torch.sum( epot_h_z * vpots_cat[:,:,:,1:], 1, keepdim=True)
                s_from_right = torch.cat([s_from_right, zeros_ver], dim=3)
                
                epot_h_v = epot_v[:,z*2:(z+1)*2,:,:]
                s_from_top = torch.sum( epot_h_v * vpots_cat[:,:,:-1,:], 1, keepdim=True)
                s_from_top = torch.cat([zeros_hor, s_from_top], dim=2)
                s_from_bottom = torch.sum( epot_h_v * vpots_cat[:,:,1:,:], 1, keepdim=True )
                s_from_bottom = torch.cat([s_from_bottom, zeros_hor], dim=2)

                vpots_update[z] = vpots_init[z] * torch.exp(s_from_left + s_from_right 
                                                            + s_from_top + s_from_bottom)

            bsum = vpots_update[0] + vpots_update[1]
            vpots_update[0] = vpots_update[0] / bsum
            vpots_update[1] = vpots_update[1] / bsum
        
        att_str = vpots_update[1] / (torch.sum(torch.sum(vpots_update[1], 2, keepdim=True), 
                                                3, keepdim=True))
        feat_str = torch.sum(torch.sum(ifeat * att_str, 2, keepdim=True), 3, keepdim=True).view(-1, self.idim)
        vemb_str = self.unstratt_emb(feat_str)

        # get the attended vectors
        # unstructured attention and embedding
        att_local = vpot_sig / torch.sum(torch.sum(vpot_sig, 2, keepdim=True), 3, keepdim=True)
        feat_local = torch.sum(torch.sum(ifeat * att_local, 2, keepdim=True), 3, keepdim=True).view(-1, self.idim)
        vemb_local = self.stratt_emb(feat_local)
        
        vis_emb = torch.cat([vemb_local, vemb_str], dim=1)

        # question embedding for answer predicting
        ver_emb = self.verbal_emb(qemb)

        ret = self.out_score(ver_emb*vis_emb)

        return ret

def sm_loss(gamma=0, n_cls=200):
    """
    Actually focal loss + softmax loss
    almost sigmoid when alpha=0.5, gamma=0
    """
    def _loss(scores, labels, need_gt_logit=False):
        eps = 1e-20

        bsize = scores.size(0)
        bix = torch.Tensor([i for i in range(bsize)]).long().cuda()

        pos_scores = F.softmax(scores)[bix, labels.data]
       
        if gamma > 0:
            pos_loss = - torch.pow(1-pos_scores, gamma) * torch.log(pos_scores+eps)
        else:
            pos_loss = - torch.log(pos_scores+eps)

        loss = torch.mean(pos_loss)
        
        if need_gt_logit:
            return loss, -pos_loss
        else:
            return loss
    
    return _loss
