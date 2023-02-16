"""
2d Convolution Module.  (c) 2021 Georgia Tech

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


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        #############################################################################
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          'constant', constant_values=0)

        H_out = (x.shape[2] + (self.padding * 2) - self.kernel_size) // self.stride + 1
        W_out = (x.shape[3] + (self.padding * 2) - self.kernel_size) // self.stride + 1
        out = np.zeros([x.shape[0], self.out_channels, H_out, W_out])

        for img in range(x_padded.shape[0]):
            img_temp = x_padded[img]
            s_c, s_h, s_w = img_temp.strides
            img_vectorized = np.lib.stride_tricks.as_strided(img_temp,
                                                             shape=(H_out, W_out, self.in_channels,
                                                                    self.kernel_size, self.kernel_size),
                                                             strides=(s_h*self.stride, s_w*self.stride, s_c, s_h, s_w),
                                                             writeable=False)
            for r in range(img_vectorized.shape[0]):
                for c in range(img_vectorized.shape[1]):
                    for kernel in range(self.out_channels):
                        receptive_field = np.sum(np.multiply(img_vectorized[r, c], self.weight[kernel])) \
                                          + self.bias[kernel]
                        out[img, kernel, r, c] = receptive_field

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        #############################################################################
        self.dx = np.zeros(x.shape)
        self.dw = np.zeros(self.weight.shape)
        self.db = np.zeros(self.bias.shape)

        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                   'constant', constant_values=0)
        dx_padded = np.zeros(x_padded.shape)

        for img in range(x.shape[0]):
            for r in range(x.shape[2]):
                for c in range(x.shape[3]):
                    for kernel in range(self.out_channels):
                        # get single d_out value & multiply by input snapshot
                        dw_field = dout[img, kernel, r, c] * x_padded[img, :, r:r + self.kernel_size,
                                                                      c:c + self.kernel_size]
                        # all above get added together in the kernel
                        self.dw[kernel, :, :, :] += dw_field

                        # single d_out and multiply by kernel weights
                        dx_field = dout[img, kernel, r, c] * self.weight[kernel]
                        # "stamp" onto appropriate location in dx_padded
                        dx_padded[img, :, r:r + self.kernel_size, c:c + self.kernel_size] += dx_field

                        # Add d_out to appropriate kernel
                        self.db[kernel] += dout[img, kernel, r, c]

        # for img in range(x.shape[0]):
        #     img_temp = x_padded[img]
        #     s_c, s_h, s_w = img_temp.strides
        #     img_vectorized = np.lib.stride_tricks.as_strided(img_temp,
        #                                                      shape=(H_out, W_out, self.in_channels,
        #                                                             self.kernel_size, self.kernel_size),
        #                                                      strides=(s_h*self.stride, s_w*self.stride, s_c, s_h, s_w),
        #                                                      writeable=False)
        #     for r in range(img_vectorized.shape[0]):
        #         for c in range(img_vectorized.shape[1]):
        #             for kern in range(self.out_channels):
        #                 receptive_field = np.sum(np.multiply(img_vectorized[r, c], self.weight[kern])) + self.bias[kern]
        #                 out[img, kern, r, c] = receptive_field


        # must remove padding on dx
        self.dx = dx_padded[:, :, self.padding:(x.shape[2])+self.padding, self.padding:(x.shape[3])+self.padding]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
