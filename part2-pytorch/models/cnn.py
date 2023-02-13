"""
Vanilla CNN model.  (c) 2021 Georgia Tech

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


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################

        #############################################################################
        infeat = 500

        self.conv = nn.Conv2d(3, 32, 7, stride=1, padding=0)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.LazyLinear(10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################

        #############################################################################
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        outs = self.fc(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
