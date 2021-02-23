import numpy as np
from nndl.layers import *
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


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of a convolutional neural network.
    #   Store the output as 'out'.
    #   Hint: to pad the array, you can use the function np.pad.
    # ================================================================ #
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    out = np.zeros(
        (
            N,
            F,
            int(1 + (H + 2 * pad - HH) / stride),
            int(1 + (W + 2 * pad - WW) / stride),
        )
    )
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = x_pad.shape
    N, F, H_prime, W_prime = out.shape
    for i in range(N):
        for f in range(F):
            curr_y = 0
            for h_prime in range(H_prime):
                curr_x = 0
                for w_prime in range(W_prime):
                    out[i][f][h_prime][w_prime] += (
                        np.sum(
                            np.multiply(
                                x_pad[i, :, curr_y : curr_y + HH, curr_x : curr_x + WW],
                                w[f],
                            )
                        )
                        + b[f]
                    )
                    curr_x += stride
                curr_y += stride

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    N, F, out_height, out_width = dout.shape
    x, w, b, conv_param = cache
    N, C, H, W = x.shape

    stride, pad = [conv_param["stride"], conv_param["pad"]]
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    num_filts, _, f_height, f_width = w.shape

    dw = np.zeros_like(w)
    dx = np.zeros_like(xpad)
    db = np.zeros_like(b)
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of a convolutional neural network.
    #   Calculate the gradients: dx, dw, and db.
    # ================================================================ #
    for i in range(N):
        for f in range(F):
            curr_y = 0
            for h_prime in range(out_height):
                curr_x = 0
                for w_prime in range(out_width):
                    dw[f, :, :, :] += np.multiply(
                        dout[i, f, h_prime, w_prime],
                        xpad[
                            i,
                            :,
                            curr_y : curr_y + f_height,
                            curr_x : curr_x + f_width,
                        ],
                    )

                    curr_x += stride
                curr_y += stride

    for i in range(N):
        for f in range(F):
            curr_y = 0
            for h_prime in range(out_height):
                curr_x = 0
                for w_prime in range(out_width):
                    dx[
                        i, :, curr_y : curr_y + f_height, curr_x : curr_x + f_width
                    ] += np.multiply(
                        dout[i, f, h_prime, w_prime],
                        w[f, :, :, :],
                    )

                    curr_x += stride
                curr_y += stride
    dx = dx[:, :, pad : pad + H, pad : pad + W]
    db = np.sum(dout, axis=(0, 2, 3))
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling forward pass.
    # ================================================================ #
    stride = pool_param["stride"]
    HH = pool_param["pool_height"]
    WW = pool_param["pool_width"]
    N, C, H, W = x.shape
    out = np.zeros((N, C, int((H - HH) / stride + 1), int((W - WW) / stride + 1)))
    N, C, H_prime, W_prime = out.shape

    for i in range(N):  # for each image
        for c in range(C):  # for each channel
            curr_y = 0
            for h in range(H_prime):  # from top to bottom
                curr_x = 0
                for w in range(W_prime):  # from left to right
                    out[i, c, h, w] = np.max(
                        x[
                            i,
                            c,
                            curr_y : curr_y + HH,
                            curr_x : curr_x + WW,
                        ]
                    )
                    curr_x += stride
                curr_y += stride
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = (
        pool_param["pool_height"],
        pool_param["pool_width"],
        pool_param["stride"],
    )

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling backward pass.
    # ================================================================ #
    stride = pool_param["stride"]
    HH = pool_param["pool_height"]
    WW = pool_param["pool_width"]
    N, C, H, W = x.shape
    out = np.zeros((N, C, int((H - HH) / stride + 1), int((W - WW) / stride + 1)))
    N, C, H_prime, W_prime = out.shape
    dx = np.zeros_like(x)
    for i in range(N):  # for each image
        for c in range(C):  # for each channel
            curr_y = 0
            for h in range(H_prime):  # from top to bottom
                curr_x = 0
                for w in range(W_prime):  # from left to right
                    max_indices = np.argmax(
                        x[
                            i,
                            c,
                            curr_y : curr_y + HH,
                            curr_x : curr_x + WW,
                        ]
                    )

                    dx[i, c, curr_y : curr_y + HH, curr_x : curr_x + WW][
                        np.unravel_index(max_indices, [HH, WW])
                    ] = dout[i, c, h, w]
                    curr_x += stride
                curr_y += stride
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm forward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you
    #   implemented in HW #4.
    # ================================================================ #

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm backward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you
    #   implemented in HW #4.
    # ================================================================ #

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dgamma, dbeta
