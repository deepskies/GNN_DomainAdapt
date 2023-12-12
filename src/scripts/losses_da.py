"""
Compute mmd loss for domain adaptation. 

"""

import torch
from functools import partial


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.

    Args:
        x (torch.Tensor): a tensor of shape [num_x_samples, num_features]
        y (torch.Tensor): a tensor of shape [num_y_samples, num_features]

    Returns:
        torch.Tensor: a distance matrix of dimensions [num_x_samples, num_y_samples].
    """
    if not x.dim() == y.dim() == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size(1) != y.size(1):
        raise ValueError('The number of features should be the same.')

    norm = lambda x: torch.sum(torch.pow(x, 2), 1)
    return torch.transpose(norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1)

def gaussian_kernel_matrix(x, y, sigmas):
    """Gaussian RBF kernel to be used in MMD adapted from: https://github.com/HKUST-KnowComp/FisherDA/blob/master/src/loss.py.

    Args:
        x (torch.Tensor): a tensor of shape [num_samples, num_features]
        y (torch.Tensor): a tensor of shape [num_samples, num_features]
        sigmas: free parameter that determins the width of the kernel

    Returns:
        torch.Tensor: a tensor of shape [num_x_samples, num_y_samples] with the kernel values
    """
    beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = torch.matmul(beta, dist.contiguous().view(1, -1))
    return torch.sum(torch.exp(-s), 0).view(*dist.size())

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    
    MMD is a distance-measure between the samples of the distributions of x and y.
    
    Args:
        x (torch.Tensor): a tensor of shape [num_samples, num_features]
        y (torch.Tensor): a tensor of shape [num_samples, num_features]
        kernel (function, optional): kernel function. Defaults to gaussian_kernel_matrix.
        
    Returns:
        torch.Tensor: a scalar denoting the squared maximum mean discrepancy loss.
    """
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    # We do not allow the loss to become negative. 
    cost = torch.clamp(cost, min=0.0)
    return cost

def mmd_distance(hs, ht):
    """Maximum Mean Discrepancy - MMD adapted from: https://github.com/HKUST-KnowComp/FisherDA/blob/master/src/loss.py.

    Args:
        hs (torch.Tensor): source domain encodings
        ht (torch.Tensor): target domain encodings

    Returns:
        torch.Tensor: a scalar denoting the squared maximum mean discrepancy loss.
    """
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
              10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix,
                              sigmas=torch.Tensor(sigmas).float().cuda())
    loss_value = maximum_mean_discrepancy(hs, ht, kernel=gaussian_kernel)
    return torch.clamp(loss_value, min=1e-4)