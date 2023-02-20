import math
import torch as nn
import torch.nn.functional as F


def gelu(x):
    """
    GELU activation function. 

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input tensor.
    """

    # The GELU function is defined as x * 0.5 * (1.0 + erf(x / sqrt(2)))
    # where erf(x) is the error function.
    # The following code implements the GELU function using the tanh function
    # instead of the error function, which is a close approximation of erf(x).

    # Compute the GELU activation function using the tanh approximation of erf(x)
    return (
        x
        * 0.5
        * (1.0 + nn.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * nn.pow(x, 3.0))))
    )

def softmax(x, dim=-1):
    """Compute the softmax of a tensor in a numerically stable way.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input tensor.
    """
    max_x, _ = nn.max(x, dim=dim, keepdim=True)
    exp_x = nn.exp(x - max_x)
    sum_exp_x = nn.sum(exp_x, dim=dim, keepdim=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x
