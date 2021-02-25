import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
        use_batchnorm=False,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize the weights and biases of a three layer CNN. To initialize:
        #     - the biases should be initialized to zeros.
        #     - the weights should be initialized to a matrix with entries
        #         drawn from a Gaussian distribution with zero mean and
        #         standard deviation given by weight_scale.
        # ================================================================ #
        C, H, W = input_dim
        self.params["W1"] = np.random.normal(
            loc=0, scale=weight_scale, size=(num_filters, C, filter_size, filter_size)
        )
        self.params["b1"] = np.zeros(num_filters)
        if self.use_batchnorm:
            self.params[f"gamma1"] = np.ones(num_filters)
            self.params[f"beta1"] = np.zeros(num_filters)
        stride = 1
        pad = (filter_size - 1) / 2
        conv_out_height = 1 + (H + 2 * pad - filter_size) / stride
        conv_out_width = 1 + (W + 2 * pad - filter_size) / stride
        pool_out_height = 1 + int((conv_out_height - 2) / 2)
        pool_out_width = 1 + int((conv_out_width - 2) / 2)
        self.params["W2"] = np.random.normal(
            loc=0,
            scale=weight_scale,
            size=(num_filters * pool_out_height * pool_out_width, hidden_dim),
        )
        self.params["b2"] = np.zeros(hidden_dim)
        if self.use_batchnorm:
            self.params[f"gamma2"] = np.ones(hidden_dim)
            self.params[f"beta2"] = np.zeros(hidden_dim)
        self.params["W3"] = np.random.normal(
            loc=0, scale=weight_scale, size=(hidden_dim, num_classes)
        )
        self.params["b3"] = np.zeros(num_classes)
        if use_batchnorm:
            self.params[f"gamma3"] = np.ones(num_classes)
            self.params[f"beta3"] = np.zeros(num_classes)

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{"mode": "train"} for i in np.arange(3)]
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = "test" if y is None else "train"
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        W1, b1, gamma1, beta1 = (
            self.params["W1"],
            self.params["b1"],
            self.params["gamma1"] if self.use_batchnorm else None,
            self.params["beta1"] if self.use_batchnorm else None,
        )
        W2, b2, gamma2, beta2 = (
            self.params["W2"],
            self.params["b2"],
            self.params["gamma2"] if self.use_batchnorm else None,
            self.params["beta2"] if self.use_batchnorm else None,
        )
        W3, b3, gamma3, beta3 = (
            self.params["W3"],
            self.params["b3"],
            self.params["gamma3"] if self.use_batchnorm else None,
            self.params["beta3"] if self.use_batchnorm else None,
        )
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the three layer CNN.  Store the output
        #   scores as the variable "scores".
        # ================================================================ #
        out, conv_relu_pool_cache = conv_relu_pool_forward(
            X, W1, b1, conv_param, pool_param
        )
        if self.use_batchnorm:
            out, bn_cache_1 = spatial_batchnorm_forward(
                out, gamma1, beta1, self.bn_params[0]
            )
        out, affine_relu_cache = affine_relu_forward(out, W2, b2)
        if self.use_batchnorm:
            out, bn_cache_2 = batchnorm_forward(out, gamma2, beta2, self.bn_params[1])
        out, affine_cache = affine_forward(out, W3, b3)
        if self.use_batchnorm:
            out, bn_cache_3 = batchnorm_forward(out, gamma3, beta3, self.bn_params[2])
        scores = out
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        if y is None:
            return scores

        loss, grads = 0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backward pass of the three layer CNN.  Store the grads
        #   in the grads dictionary, exactly as before (i.e., the gradient of
        #   self.params[k] will be grads[k]).  Store the loss as "loss", and
        #   don't forget to add regularization on ALL weight matrices.
        # ================================================================ #
        loss, dout = softmax_loss(scores, y)
        loss += (
            self.reg
            * 0.5
            * (np.linalg.norm(W1) + np.linalg.norm(W2) + np.linalg.norm(W3))
        )

        if self.use_batchnorm:
            dout, dgamma3, dbeta3 = batchnorm_backward(dout, bn_cache_3)
        dout, dw3, db3 = affine_backward(dout, affine_cache)
        if self.use_batchnorm:
            dout, dgamma2, dbeta2 = batchnorm_backward(dout, bn_cache_2)
        dout, dw2, db2 = affine_relu_backward(dout, affine_relu_cache)
        if self.use_batchnorm:
            dout, dgamma1, dbeta1 = spatial_batchnorm_backward(dout, bn_cache_1)
        dout, dw1, db1 = conv_relu_pool_backward(dout, conv_relu_pool_cache)

        grads["W3"] = dw3 + self.reg * W3
        grads["b3"] = db3
        if self.use_batchnorm:
            grads["gamma3"] = dgamma3
            grads["beta3"] = dbeta3

        grads["W2"] = dw2 + self.reg * W2
        grads["b2"] = db2
        if self.use_batchnorm:
            grads["gamma2"] = dgamma2
            grads["beta2"] = dbeta2

        grads["W1"] = dw1 + self.reg * W1
        grads["b1"] = db1
        if self.use_batchnorm:
            grads["gamma1"] = dgamma1
            grads["beta1"] = dbeta1
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads
