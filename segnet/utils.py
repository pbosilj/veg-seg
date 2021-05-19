from typing import Dict
from operator import itemgetter

from .segnet_basic import SegNetBasic
from torch.optim import SGD

def init_SGD_optimizer(net: SegNetBasic,
                    base_lr = 0.01,
                    weight_decay = 0.0005,
                    momentum = 0.9,
                    fine_tune = False):

    net_weights = map(itemgetter(1), filter(lambda x: 'bias' not in x[0], net.named_parameters()))
    net_biases = map(itemgetter(1), filter(lambda x: 'bias' in x[0], net.named_parameters()))

    # need to un-hardcode these
    
    #base_lr = 0.01
    #weight_decay = 0.0005
    #momentum = 0.9
    
    net_weights = list(map(itemgetter(1), filter(lambda x: 'bias' not in x[0] and 'conv_classifier' not in x[0], net.named_parameters())))
    net_biases = list(map(itemgetter(1), filter(lambda x: 'bias' in x[0] and 'conv_classifier' not in x[0], net.named_parameters())))

    classifier_weight = net.conv_classifier.weight
    classifier_bias = net.conv_classifier.bias

    if not fine_tune: # training from scratch
        # Classifier learns as fast as the rest of it
        net_weights.append(classifier_weight)
        net_biases.append(classifier_bias)
        
        optimizer = SGD([
                            {'params': net_weights, 'lr': base_lr, 'weight_decay': weight_decay },
                            {'params': net_biases, 'lr': base_lr*2 }
                        ],
                        momentum = momentum, # but note the docs, might need to change value: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
                        lr = base_lr) # probably not needed
                                    
    else: # fine tuning a pre-trained model
        # Increase classifier lr 10-fold
        optimizer = SGD([
                            {'params': net_weights, 'lr': base_lr, 'weight_decay': weight_decay },
                            {'params': net_biases, 'lr': base_lr*2 },
                            {'params': classifier_weight, 'lr': base_lr*10, 'weight_decay': weight_decay },
                            {'params': classifier_bias, 'lr': base_lr*20 },
                        ],
                        momentum = momentum, # but note the docs, might need to change value: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
                        lr = base_lr) # probably not needed
    
    return optimizer
