# utils/losses.py

import torch
import torch.nn as nn
from utils.prob_utils import cluster_pdf
from torch.distributions import MultivariateNormal
from torch.linalg import inv


def drift_loss(force, yf):
    force, _ = force(yf)
    return torch.mean(torch.norm(force, dim=1))

def calculate_combined_weights_per_point(data_points, pis, mus, sigmas):
    pis = torch.tensor(pis,dtype=torch.float32)
    mus = torch.tensor(mus,dtype=torch.float32)
    sigmas = torch.tensor(sigmas,dtype=torch.float32)
    N, d = data_points.shape
    K = mus.shape[0]

    pdf_values = torch.zeros(N, K, dtype=data_points.dtype, device=data_points.device)
    relative_likelihoods = torch.zeros(N, K, dtype=data_points.dtype, device=data_points.device)

    for k in range(K):
        mvn = MultivariateNormal(loc=mus[k], covariance_matrix=sigmas[k])
        pdf_values[:, k] = torch.exp(mvn.log_prob(data_points))

        mu_k = mus[k]
        sigma_k = sigmas[k]
        sigma_k_inv = inv(sigma_k)
        diff = data_points - mu_k
        mahalanobis_sq = torch.sum((diff @ sigma_k_inv) * diff, dim=1)
        relative_likelihoods[:, k] = torch.exp(-0.5 * mahalanobis_sq)

    marginal_likelihood = pdf_values/torch.sum(pdf_values, dim=1, keepdim=True)

    pis_view = pis.view(1, -1)
    weighted_pdfs = pdf_values * pis_view
    marginal_likelihood = torch.sum(weighted_pdfs, dim=1, keepdim=True)
    marginal_likelihood = torch.clamp(marginal_likelihood, min=1e-12)
    posterior_probs = weighted_pdfs / marginal_likelihood

    combined_weights = relative_likelihoods * posterior_probs

    return combined_weights

def aggregate_data_weights(combined_weights_per_point):
    w_t = torch.sum(combined_weights_per_point, dim=0)
    return w_t

def weight_loss(data,pis,mus,sigmas,known_weight):
    known_weight = torch.tensor(known_weight,dtype=torch.float32)
    estimated_possible = calculate_combined_weights_per_point(data, pis, mus, sigmas)
    estimated_weight = aggregate_data_weights(estimated_possible)
    mseloss = nn.MSELoss()
    return mseloss(known_weight,estimated_weight)/sum(known_weight)


def soft_constraint_loss(model, net):
    n_th = len(model.n)
    positive_mask = torch.tensor((net == 1).float().repeat(1, n_th))
    negative_mask = torch.tensor((net == -1).float().repeat(1, n_th))
    mask_value = model.fc_layer_activation.weight * positive_mask + model.fc_layer_repression.weight * negative_mask
    return torch.sum(mask_value ** 2)

def gaussian_kernel(x, mu, sigma):
    return 1 / (sigma * torch.sqrt(2 * torch.tensor(3.14))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def kernel_density_estimate(samples, x_grid, bandwidth):
    pdf = torch.zeros_like(x_grid)
    for sample in samples:
        pdf += gaussian_kernel(x_grid, sample, bandwidth)
    pdf /= len(samples)
    return pdf/pdf.sum()

def gmm_marginal_pdf(x_grid, pis, mus, sigmas, dim):
    pis = torch.tensor(pis,dtype=torch.float32)
    mus = torch.tensor(mus,dtype=torch.float32)
    sigmas = torch.tensor(sigmas,dtype=torch.float32)
    pdf = torch.zeros_like(x_grid)
    for k in range(len(pis)):
        mu = mus[k][dim]
        sigma = torch.sqrt(sigmas[k][dim,dim])
        pdf += pis[k] * gaussian_kernel(x_grid, mu, sigma)
    return pdf/pdf.sum()

def kl_divergence(p_pdf, q_pdf):
    kl = torch.sum(p_pdf * torch.log((p_pdf + 1e-10) / (q_pdf + 1e-10)))
    return kl

def kl_loss(data, pis, mus, sigmas, x=torch.arange(0, 20, 0.1), bw=1):
    n = data.shape[1]
    loss = 0
    for idx in range(n):
        y1 = gmm_marginal_pdf(x, pis, mus, sigmas, idx) 
        y2 = kernel_density_estimate(data[:, idx], x, bw) 
        loss += kl_divergence(y1, y2)
    return loss