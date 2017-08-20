import mxnet as mx
from collections import namedtuple
# use bayesian GRU
GRUState = namedtuple("GRUState", ["h"])
GRUParam = namedtuple("GRUParam", ["gates_i2h_weight", "gates_i2h_bias",
                                   "gates_h2h_weight", 
                                   "trans_i2h_weight", "trans_i2h_bias",
                                   "trans_h2h_weight"])
GRUDropoutParam = namedtuple("GRUDropoutParam", ["gates_i2h", "gates_h2h", 
                                   "trans_i2h", "trans_h2h"])

def gru(num_hidden, indata, prev_state, param, seqidx, layeridx, 
        prefix, mask=None, dp_param=None):
    # mask=1, update h; otherwise, keep h
    if dp_param is not None:
        indata_gates = mx.sym.broadcast_mul(indata, dp_param.gates_i2h)
        prevh_gates = mx.sym.broadcast_mul(prev_state.h, dp_param.gates_h2h)
        indata_trans = mx.sym.broadcast_mul(indata, dp_param.trans_i2h)
    else:
        indata_gates = indata
        prevh_gates = prev_state.h
        indata_trans = indata


    i2h = mx.sym.FullyConnected(data=indata_gates, 
                                weight=param.gates_i2h_weight,
                                bias=param.gates_i2h_bias,
                                num_hidden=num_hidden*2,
                                name='%s_t%d_l%d_gates_i2h'%(prefix, seqidx, layeridx))
    # use encoder_U
    h2h = mx.sym.FullyConnected(data=prevh_gates,
                                weight=param.gates_h2h_weight,
                                no_bias=True,
                                num_hidden=num_hidden*2,
                                name='%s_t%d_l%d_gates_h2h'%(prefix, seqidx, layeridx))
    gates = i2h+h2h

    gates_act = mx.sym.Activation(gates, act_type='sigmoid')
    slice_gates = mx.sym.SliceChannel(gates_act, num_outputs=2,
                                      name='%s_t%d_l%d_slide'%(prefix, seqidx, layeridx))
    update_gate = slice_gates[0]

    reset_gate = slice_gates[1]

    htrans_i2h = mx.sym.FullyConnected(data=indata_trans, 
                                        weight=param.trans_i2h_weight,
                                        bias=param.trans_i2h_bias,
                                        num_hidden=num_hidden,
                                        name='%s_t%d_l%d_trans_i2h'%(prefix, seqidx, layeridx))

    h_after_reset = prev_state.h * reset_gate
    # use encode_Ux
    if dp_param is not None:
        h_after_reset = mx.sym.broadcast_mul(h_after_reset, dp_param.trans_h2h)

    htrans_h2h = mx.sym.FullyConnected(data=h_after_reset, 
                                        weight=param.trans_h2h_weight,
                                        no_bias=True,
                                        num_hidden=num_hidden,
                                        name='%s_t%d_l%d_trans_h2h'%(prefix, seqidx, layeridx))
    h_trans = htrans_i2h + htrans_h2h
    h_trans_active = mx.sym.Activation(h_trans, act_type='tanh')
    next_h = prev_state.h + update_gate * (h_trans_active - prev_state.h)
    if mask is not None:
        next_h = prev_state.h + mx.sym.broadcast_mul(mask, next_h - prev_state.h)
    return GRUState(h=next_h)

def bayesian_dp_sym(p, shape):
    # make a dropout mask for bayesian dropout
    # calculate the lower bound corresponding to bayesian dp
    assert(p<=0.5)
    uni_min = (0.5 - p)/(1.0 - p)
    rand_num = mx.sym.uniform(low=uni_min, high=1, shape=shape)
    mask = mx.sym.round(rand_num) / (1.0 - p)
    return mask

def GRU_unroll(batch_size, input_seq, in_dim, seq_len, num_hidden, prefix, 
               dropout=0, mask=None, n_gpus=1):
    """
    Data:
    prefix+'l0_init_h': set to all 0
    mask: used for variable length sequences
    need_middle: whether we need the intermediate h. For sentence embedding, set to False.
                    for input module, set to True
    """
    if dropout>0:
        x0 = batch_size if n_gpus==1 else 1 # unsolved problem here...
        dp_param = GRUDropoutParam(gates_i2h=bayesian_dp_sym(dropout, (x0, in_dim)),
                                    gates_h2h=bayesian_dp_sym(dropout, (x0, num_hidden)),
                                    trans_i2h=bayesian_dp_sym(dropout, (x0, in_dim)),
                                    trans_h2h=bayesian_dp_sym(dropout, (x0, num_hidden)))
    else:
        dp_param=None

    layer_num = 0
    gru_param = GRUParam(gates_i2h_weight=mx.sym.Variable('%s_l%d_i2h_gates_weight'%(prefix, layer_num)),
                        gates_i2h_bias=mx.sym.Variable("%s_l%d_i2h_gates_bias" % (prefix, layer_num)),
                        gates_h2h_weight=mx.sym.Variable("%s_l%d_h2h_gates_weight" % (prefix, layer_num)),
                        trans_i2h_weight=mx.sym.Variable("%s_l%d_i2h_trans_weight" % (prefix, layer_num)),
                        trans_i2h_bias=mx.sym.Variable("%s_l%d_i2h_trans_bias" % (prefix, layer_num)),
                        trans_h2h_weight=mx.sym.Variable("%s_l%d_h2h_trans_weight" % (prefix, layer_num)))
    state = GRUState(h=mx.sym.Variable("%s_l%d_init_h" % (prefix, layer_num)))
    wordvec = mx.sym.SliceChannel(data=input_seq, num_outputs=seq_len, 
                                  squeeze_axis=True, name=prefix+'_slice_word')
    masks = mx.sym.SliceChannel(data=mask, num_outputs=seq_len, 
                                squeeze_axis=False, name=prefix+'_slice_mask')
    hiddens = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        mask_t = masks[seqidx]
        
        state = gru(num_hidden, indata=hidden, mask=mask_t,
                        prev_state=state, param=gru_param,
                        seqidx=seqidx, layeridx=layer_num, 
                        dp_param=dp_param, prefix=prefix)
    
        hiddens.append(state.h)

    ret = hiddens[-1]
    return ret

def GRU_sent_encoder(batch_size, max_len, vocab_size, hidden_dim, wordembed_dim, 
                     dropout=0.0, is_train=True, n_gpus=1):
    """
    Implementing the GRU of skip-thought vectors.
    Use masks so that sentences at different lengths can be put into the same batch.

    sent_seq: sequence of tokens consisting a sentence, shape: batch_size x max_len
    mask: 1 indicating valid, 0 invalid, shape: batch_size x max_len
    embed_weight: word embedding, shape: 
    """
    sent_seq = mx.sym.Variable('sent_seq')
    mask = mx.sym.Variable('mask')
    embed_weight = mx.sym.Variable('embed_weight')

    embeded_seq = mx.sym.Embedding(data=sent_seq, input_dim=vocab_size, weight=embed_weight, 
                                   output_dim=wordembed_dim, name='sent_embedding')

    sent_vec = GRU_unroll(batch_size, embeded_seq, mask=mask, 
                          in_dim=wordembed_dim, seq_len=max_len, 
                          num_hidden=hidden_dim, dropout=dropout,
                          prefix='sent', n_gpus=n_gpus)

    return sent_vec

def get_dp_fc_act(sym, num_hidden, name, no_bias=False, act_type=None, dp=0):
    if dp>0:
        sym=mx.sym.Dropout(sym, p=dp)
    sym=mx.sym.FullyConnected(data=sym, num_hidden=num_hidden, no_bias=no_bias, name=name)
    if act_type!=None:
        sym = mx.sym.Activation(sym, act_type=act_type)
    return sym

def meanfield_accelerate(epot_v_h, epot_v_v, epot_s, horizontal_zeros, vertical_zeros, vpot_1, max_iter, 
                         h=14, w=14, epot_common_dim=512, seq_idx=0, epot_weight=None, epot_bias=None):
    """
    Accelerated version of MF.

    vpot_1: batch_size x hw
    ifeature_map: batch_size x c x h x w
    horizontal_zeros: batch_size x 1 x 1 x w
    """
    
    vpot_0 = 1.0 - vpot_1
    vpots_initial = [vpot_0, vpot_1]
    vpots_update = [vpot_0, vpot_1]

    epot_mul_h = mx.sym.broadcast_mul(epot_s, epot_v_h)
    epot_mul_v = mx.sym.broadcast_mul(epot_s, epot_v_v)

    epot_h = mx.sym.Convolution(data=epot_mul_h, kernel=(1,1), num_filter=4, 
                                weight=epot_weight, bias=epot_bias, name='epot_h_t%d'%seq_idx)
    epot_h = mx.sym.Activation(epot_h, act_type='tanh')
    epot_v = mx.sym.Convolution(data=epot_mul_v, kernel=(1,1), num_filter=4, 
                                weight=epot_weight, bias=epot_bias, name='epot_v_t%d'%seq_idx)
    epot_v = mx.sym.Activation(epot_v, act_type='tanh')

    # potential to the left node
    epot_h_i01 = mx.sym.SliceChannel(epot_h, num_outputs=2, axis=1, name='epot_hor_slice_t%d'%seq_idx)
    epot_v_i01 = mx.sym.SliceChannel(epot_v, num_outputs=2, axis=1, name='epot_ver_slice_t%d'%seq_idx)

    for t in range(max_iter):
        b_concat = mx.sym.Concat(vpots_update[0], vpots_update[1], dim=1)
        b_from_left_crop = mx.sym.Crop(b_concat, offset=(0,0), h_w=(h, w-1))
        b_from_right_crop = mx.sym.Crop(b_concat, offset=(0,1), h_w=(h, w-1))
        b_from_top_crop = mx.sym.Crop(b_concat, offset=(0,0), h_w=(h-1, w))
        b_from_bottom_crop = mx.sym.Crop(b_concat, offset=(1,0), h_w=(h-1, w))
        for z_i in range(2):
            s_from_left = mx.sym.sum(epot_h_i01[z_i]*b_from_left_crop, axis=1, keepdims=True)
            s_from_left = mx.sym.Concat(vertical_zeros, s_from_left, dim=3)
            s_from_right = mx.sym.sum(epot_h_i01[z_i]*b_from_right_crop, axis=1, keepdims=True)
            s_from_right = mx.sym.Concat(s_from_right, vertical_zeros, dim=3)

            s_from_top = mx.sym.sum(epot_v_i01[z_i]*b_from_top_crop, axis=1, keepdims=True)
            s_from_top = mx.sym.Concat(horizontal_zeros, s_from_top, dim=2)
            s_from_bottom = mx.sym.sum(epot_v_i01[z_i]*b_from_bottom_crop, axis=1, keepdims=True)
            s_from_bottom = mx.sym.Concat(s_from_bottom, horizontal_zeros, dim=2)

            vpots_update[z_i] = vpots_initial[z_i]* mx.sym.exp( s_from_left+s_from_right+s_from_top+s_from_bottom )
        # normalize
        bsum = vpots_update[0]+vpots_update[1]
        vpots_update[0] = vpots_update[0]/bsum
        vpots_update[1] = vpots_update[1]/bsum
    
    attn_sum = mx.sym.sum(vpots_update[1], axis=(2,3), keepdims=True)
    attn = mx.sym.broadcast_div(vpots_update[1], attn_sum)
    return attn

def MF_accelerate(batch_size, is_train, general_dp=.0, qemb_dp=.0,
                  crf_iter=3, seq_len=26, common_embed_size=1200, epot_common_dim=1200,
                  n_gpus=1, w=14, h=14, idim=2048, n_ans=2000):
    """
    The accelerated version of MF.

    Leaving epot_common_dim for tuning for efficiency considerations.

    img_feature: feature map from CNN, shape: batch_size x idim x h x w
    vertical_zeros: zero padding for potential values, shape: batch_size x 1 x h x 1
    horizontal_zeros: zero padding for potential values, shape: batch_size x 1 x 1 x w
    """
    # these names are used to match the input data names
    ifeature = mx.sym.Variable('img_feature')
    label = mx.sym.Variable('ans_label')
    vertical_zeros = mx.sym.Variable('vertical_zeros')
    horizontal_zeros = mx.sym.Variable('horizontal_zeros')
    
    if not is_train:
        # ignore the dp operators to avoid extra computation
        general_dp, qemb_dp=0.0, 0.0

    ifeature_map = mx.sym.SwapAxis(ifeature, dim1=1, dim2=2)
    ifeature_map = mx.sym.Reshape(ifeature_map, shape=(-1, idim, h, w))
    ifeature_map_vpot = ifeature_map if general_dp==0 else mx.sym.Dropout(ifeature_map, p=general_dp)
    ifeature_map_epot = ifeature_map if general_dp==0 else mx.sym.Dropout(ifeature_map, p=general_dp)

    # some constants are from skip thought vector
    qembed = GRU_sent_encoder(batch_size, seq_len, 15031, 2400, 620, 
                              dropout=qemb_dp, is_train=is_train, n_gpus=n_gpus)

    # get the projected potentials with convolutions for acceleration
    # vpot: vertice potential, epot: edge potential
    epot_conv_weight=mx.sym.Variable('epot_conv_weight')
    epot_conv_weight_t = mx.sym.transpose(epot_conv_weight, axes=(0,1,3,2))

    vpot_v = mx.sym.Convolution(data=ifeature_map_vpot, kernel=(1,1), 
                                num_filter=common_embed_size, name='vpot_v')
    vpot_v = mx.sym.Activation(vpot_v, act_type='tanh')
    epot_v_h = mx.sym.Convolution(data=ifeature_map_epot, kernel=(1,2), weight=epot_conv_weight,
                                  num_filter=epot_common_dim, name='epot_v_h')
    epot_v_h = mx.sym.Activation(epot_v_h, act_type='tanh')
    epot_v_v = mx.sym.Convolution(data=ifeature_map_epot, kernel=(2,1), weight=epot_conv_weight_t,
                                  num_filter=epot_common_dim, name='epot_v_v')
    epot_v_v = mx.sym.Activation(epot_v_v, act_type='tanh')

    vpot_q = get_dp_fc_act(qembed, num_hidden=common_embed_size, name='qproj_v',
                           act_type='tanh', dp=general_dp)
    vpot_q = mx.sym.Reshape(vpot_q, shape=(-1, common_embed_size, 1,1))
    epot_q = get_dp_fc_act(qembed, num_hidden=epot_common_dim, name='qproj_e',
                           act_type='tanh', dp=general_dp)
    epot_q = mx.sym.Reshape(epot_q, shape=(-1, epot_common_dim, 1,1))

    # get the potentials  
    vpot_mul = mx.sym.broadcast_mul(vpot_q, vpot_v)
    attn_score = mx.sym.Convolution(data=vpot_mul, kernel=(1,1), num_filter=1, 
                                    no_bias=True, name='v_attn_score')

    vpot_1 = mx.sym.Activation(data=attn_score, act_type='sigmoid')

    epot_weight = mx.sym.Variable('epot_weight')
    epot_bias = mx.sym.Variable('epot_bias')    
    attn_struct = meanfield_accelerate(epot_v_h, epot_v_v, epot_q, horizontal_zeros, vertical_zeros, 
                                       vpot_1, crf_iter, h=h, w=w, epot_common_dim=epot_common_dim, 
                                       epot_weight=epot_weight, epot_bias=epot_bias)
    # using pooling to prevent cuda error
    ifeat_struct_weigh = mx.sym.broadcast_mul(ifeature_map, attn_struct)
    ifeat_structure = mx.sym.Pooling(data=ifeat_struct_weigh, kernel=(h,w), pool_type='sum')
    ifeat_structure = mx.sym.Reshape(ifeat_structure, shape=(-1, idim))
    ifeat_str_proj = get_dp_fc_act(ifeat_structure, num_hidden=common_embed_size, 
                                   name='struc_proj', act_type='tanh', dp=general_dp)

    # concatenate the unstructured attention
    vpot_1_norm = mx.sym.broadcast_div(vpot_1, mx.sym.sum(vpot_1, axis=(2,3), keepdims=True))
    ifeat_weigh = mx.sym.broadcast_mul(vpot_1_norm, ifeature_map) 
    ifeat_vpot = mx.sym.Pooling(ifeat_weigh, kernel=(h,w), pool_type='sum')
    ifeat_vpot = mx.sym.Reshape(ifeat_vpot, shape=(-1, idim))
    vpot_proj = get_dp_fc_act(ifeat_vpot, num_hidden=common_embed_size, 
                              name='vpot_proj', act_type='tanh', dp=general_dp)

    ifeat_concat = mx.sym.Concat(ifeat_str_proj, vpot_proj, dim=1)

    qproj1 = get_dp_fc_act(qembed, num_hidden=common_embed_size*2, 
                           name='qproj1', act_type='tanh', dp=general_dp) 
    mul1 = ifeat_concat * qproj1

    if general_dp>0:
        mul1 = mx.sym.Dropout(mul1, p=general_dp)
    out_fc = mx.sym.FullyConnected(data=mul1, num_hidden=n_ans, name='out_fc')
    out_score = mx.sym.SoftmaxOutput(data=out_fc, label=label, name='out_sm')

    return out_score
