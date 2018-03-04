import torch
import numpy as np
def xavier_initialier(module, magnitude=3, rnd_type='uniform', factor_type='avg'):
    """
    Xavier initializer,
    Valid factor_type: 'avg', 'in', 'out'
    Valid rnd_type: 'uniform', 'gaussian'
    """

    # initialize the rest with Xavier
    # reference: https://github.com/dmlc/mxnet/blob/master/python/mxnet/initializer.py
    if module.bias is not None:
      module.bias.data.zero_()
    shape = module.weight.size()
    hw_scale = 1.
    if len(shape) > 2:
        hw_scale = np.prod(shape[2:])
    try:
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
    except:
        raise ValueError("Strange shape {} of module {}".format(shape, module))
    factor = 1
    if factor_type == "avg":
        factor = (fan_in + fan_out) / 2.0
    elif factor_type == "in":
        factor = fan_in
    elif factor_type == "out":
        factor = fan_out
    else:
        raise ValueError("Incorrect factor type")
    scale = np.sqrt(magnitude / factor)
    if rnd_type == 'uniform':
        module.weight.data.uniform_(-scale, scale)
    elif rnd_type == 'gaussian':
        module.weight.data.normal_(0, scale)
    else:
        raise ValueError("Incorrect rnd type! Weight not initialized.")

def fixmag_initialier(data, magnitude=3, rnd_type='gaussian'):
    """
    Directly specify the magnitude
    Valid factor_type: 'avg', 'in', 'out'
    Valid rnd_type: 'uniform', 'gaussian'
    """
    if rnd_type == 'uniform':
        data.uniform_(-magnitude, magnitude)
    elif rnd_type == 'gaussian':
        data.normal_(0, magnitude)
    else:
        raise ValueError("Incorrect rnd type! Weight not initialized.")

