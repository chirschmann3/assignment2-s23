"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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

import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        H_out = (x.shape[2] - self.kernel_size) // self.stride + 1
        W_out = (x.shape[3] - self.kernel_size) // self.stride + 1
        out = np.zeros([x.shape[0], x.shape[1], H_out, W_out])

        for img in range(x.shape[0]):
            for kernel in range(x.shape[1]):
                for r in range(H_out):
                    for c in range(W_out):
                        # get "snapshot" to apply pooling to
                        receptive_field = x[img, :, r * self.stride:r * self.stride + self.kernel_size,
                                            c * self.stride:c * self.stride + self.kernel_size]
                        # get max of receptive field and put in appropriate place in output
                        out[img, kernel, r, c] = np.max(receptive_field)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        for img in range(x.shape[0]):
            for kernel in range(x.shape[1]):
                for r in range(H_out):
                    for c in range(W_out):
                        receptive_field = x[img, kernel, r * self.stride:r * self.stride + self.kernel_size,
                                            c * self.stride:c * self.stride + self.kernel_size]
                        # get argmax index and location
                        i = receptive_field.argmax()
                        matrix_location = np.unravel_index(i, x.shape)
                        max_num = receptive_field[matrix_location]

                        # use above to update dx with d_out
                        temp_dx = dout[img, kernel, r, c] * max_num
                        self.dx[img, kernel, r * self.stride:r * self.stride + self.kernel_size,
                                             c * self.stride:c * self.stride + self.kernel_size] += temp_dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
