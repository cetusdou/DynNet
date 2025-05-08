import torch
import torch.nn as nn
from utils.solver import euler_solver_sde
from train.loss import *

def train_omega(model, y0, yf,ref_grn, num_epochs,mu, lr=1e-4, constraint=True):
    for param in model.variance_net.parameters():
        param.requires_grad = False
    
    # Enable gradient computation for specific layers
    model.fc_layer_activation.weight.requires_grad = True
    model.fc_layer_repression.weight.requires_grad = True
    model.D.requires_grad = False
    model.s.requires_grad = False
    model.D = nn.Parameter(torch.zeros_like(model.D))
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear previous gradients
        force_1, _ = model(yf)
        d_loss = drift_loss(model.velocity_estimator, yf)
        loss = mu*d_loss
        if constraint:
            loss += soft_constraint_loss(model, ref_grn)
        loss.backward()
        optimizer.step()  # Update model parameters
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        losses.append(loss.item())
    
    return losses

def train_s(model, y0, yf, gmm,mu, num_epochs, lr=0.01):
    pis = gmm.weights_
    mus = gmm.means_
    sigmas = gmm.covariances_
    for param in model.variance_net.parameters():
        param.requires_grad = False
    
    model.fc_layer_activation.weight.requires_grad = False
    model.fc_layer_repression.weight.requires_grad = False
    model.D.requires_grad = False
    model.s.requires_grad = True
    model.D = nn.Parameter(torch.zeros_like(model.D))
    num_gene = y0.shape[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        d_loss = drift_loss(model.velocity_estimator, yf)

        
        trajectory = euler_solver_sde(model, y0, num_steps=100, num_genes=num_gene)
        yf_fit = trajectory[-1] 
        w_loss = weight_loss(yf_fit,pis,mus,sigmas,pis*yf.shape[0])
        loss = mu[0]*d_loss + mu[1]*w_loss
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        losses.append(loss.item())
    return losses


def train_variance_net(model, y0, gmm, num_epochs, lr = 0.01):
    pis = gmm.weights_
    mus = gmm.means_
    sigmas = gmm.covariances_
    num_gene = y0.shape[1]
    for param in model.variance_net.parameters():
        param.requires_grad = True
    model.fc_layer_activation.weight.requires_grad = False
    model.fc_layer_repression.weight.requires_grad = False
    model.D = nn.Parameter(torch.ones_like(model.D))
    model.D.requires_grad = True
    model.s.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        trajectory = euler_solver_sde(model,y0,num_steps=100,num_genes=num_gene)
        yf_fit = trajectory[-1]
        k_loss = kl_loss(yf_fit, pis, mus, sigmas)
        loss =  k_loss
        loss.backward()
        optimizer.step()
        
        # 每个epoch打印损失
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        losses.append(loss.item())
    return losses
