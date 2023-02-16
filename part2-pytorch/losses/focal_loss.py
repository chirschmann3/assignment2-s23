"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################

    #############################################################################
    per_cls_weights = [(1 - beta) / (1 - beta**n) for n in cls_num_list]

    # normalize so that sum of "alphas" = # of classes
    per_cls_weights = torch.Tensor([len(per_cls_weights) * a / sum(per_cls_weights) for a in per_cls_weights])

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################

        #############################################################################
        # below adapted from https://gist.github.com/samson-wang/e5cee676f2ae97795356d9c340d1ec7f

        sm = torch.softmax(input, dim=1)
        fl = torch.Tensor(self.weight) * -((1 - sm) ** self.gamma) * torch.log(sm)
        # get the loss for only the correct ground truth label
        range_img = torch.arange(0, input.shape[0])
        loss = sum(fl[range_img, target])

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
