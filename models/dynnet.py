import torch
import torch.nn as nn

class DynNet(nn.Module):
    """
    Model for gene regulatory dynamics using Hill functions.
    """
    def __init__(self, h, s_init, D_init, num_gene, n=[4], gene_net=None, use_bias=False):
        super(DynNet, self).__init__()
        self.h = h
        self.num_gene = num_gene
        self.gene_net = gene_net  
        self.sqrth = torch.sqrt(torch.tensor(self.h, dtype=torch.float))
        self.s = nn.Parameter(torch.ones(self.num_gene) * s_init)
        self.D = nn.Parameter(torch.ones(self.num_gene) * D_init)
        self.n = n
        self.k = torch.ones(self.num_gene)

        self.positive_mask = (self.gene_net == 1).float() if gene_net is not None else torch.ones((self.num_gene, self.num_gene))
        self.negative_mask = (self.gene_net == -1).float() if gene_net is not None else torch.ones((self.num_gene, self.num_gene))

        self.n_hills = len(self.n)
        self.fc_layer_activation = nn.Linear(self.num_gene * self.n_hills, self.num_gene, bias=use_bias)
        self.fc_layer_repression = nn.Linear(self.num_gene * self.n_hills, self.num_gene, bias=use_bias)

        self.fc_layer_activation.weight.data = torch.abs(self.fc_layer_activation.weight.data)
        self.fc_layer_repression.weight.data = torch.abs(self.fc_layer_repression.weight.data)

        self.variance_net = nn.Sequential(
            nn.Linear(self.num_gene, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_gene),
            nn.Sigmoid()
        )

    def hill_features(self, x):
        """
        Calculate Hill function features.
        """
        x = x.float()
        s = self.s.float()
        x_s = x / s.squeeze()
        list_interaction = [
            torch.stack([
                torch.pow(x_s, nth) / (1 + torch.pow(x_s, nth)),
                1 / (1 + torch.pow(x_s, nth))
            ], dim=1)
            for nth in self.n
        ]
        interaction_stack = torch.cat(list_interaction, dim=2)
        return interaction_stack

    def velocity_estimator(self, x):
        """
        Estimate velocity.
        """
        hill_forward = self.hill_features(x)
        forward_output_activation = self.fc_layer_activation(hill_forward[:, 0, :])
        forward_output_repression = self.fc_layer_repression(hill_forward[:, 1, :])
        return forward_output_activation + forward_output_repression - self.k * x.squeeze(1)

    def variance_estimator(self, x):
        """
        Estimate variance.
        """
        return self.variance_net(x)

    def forward(self, inputs):
        """
        Forward propagation.
        """
        force = self.velocity_estimator(inputs)
        variance = self.variance_estimator(inputs).float()
        random_numbers = torch.randn_like(inputs) * variance * self.D
        state = inputs + self.h * force + random_numbers * self.sqrth
        return force, state


class DynNet_Hard(nn.Module):
    def __init__(self, h, D_init, gene_net,hidden_dim=64,num_hidden_layers=2):
        super(DynNet_Hard, self).__init__()
        self.h = h
        self.num_gene = gene_net.shape[0]
        self.gene_net = gene_net
        self.sqrth = torch.sqrt(torch.tensor(self.h))
        self.kernel = nn.Parameter(torch.rand((self.num_gene, 3 * self.num_gene))*5)
        self.D = nn.Parameter(torch.ones(self.num_gene)*D_init)
        self.k = torch.ones(self.num_gene)
        # 定义用于估计方差的子网络
        self.variance_estimator = nn.Sequential(
            nn.Linear(self.num_gene, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_hidden_layers)],
            nn.Linear(hidden_dim, self.num_gene),
        )

        self.positive_mask = (self.gene_net == 1).float()
        self.negative_mask = (self.gene_net == -1).float()

    def force(self, x):
        a, b, s = self.kernel[:, :self.num_gene].abs(), self.kernel[:, self.num_gene:2 * self.num_gene].abs(), self.kernel[:, 2 * self.num_gene:].abs()
        k = torch.ones_like(x[0, :])
        n = 4
        x = x.unsqueeze(1)
        active_production = torch.sum(self.positive_mask * (a.unsqueeze(0) * torch.pow(x, n) / (torch.pow(s.unsqueeze(0), n) + torch.pow(x, n))), dim=-1)
        repress_production = torch.sum(self.negative_mask * (b.unsqueeze(0) * torch.pow(s.unsqueeze(0), n) / (torch.pow(x, n) + torch.pow(s.unsqueeze(0), n))), dim=-1)
        production = active_production + repress_production
        degradation = k * x.squeeze(1)

        force = production - degradation

        return force

    def forward(self, x):
        force = self.force(x)
        noiseD = self.variance_estimator(x).abs()
        random_numbers = torch.randn_like(x)*noiseD
        states = x + self.h * force + random_numbers* self.sqrth
        return force, states




class EMT_Sim(nn.Module):
    def __init__(self, h, gene_net,D_init=1):
        super(EMT_Sim, self).__init__()
        self.h = h
        self.num_gene = gene_net.shape[0]
        self.gene_net = gene_net
        self.sqrth = torch.sqrt(torch.tensor(self.h))
        self.kernel = nn.Parameter(torch.rand((self.num_gene, 3 * self.num_gene))*5)
        self.D = nn.Parameter(torch.ones(self.num_gene)*D_init)
        self.variance_estimator = nn.Sequential(
            nn.Linear(self.num_gene, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16,self.num_gene),
        )
        self.k = torch.ones(self.num_gene)
        self.positive_mask = (self.gene_net == 1).float()
        self.negative_mask = (self.gene_net == -1).float()

    def velocity_estimator(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        num_gene = self.num_gene
        a, b, s = self.kernel[:, :num_gene], self.kernel[:, num_gene:2 * num_gene], self.kernel[:, 2 * num_gene:]
        a = torch.abs(a)
        b = torch.abs(b)
        s = torch.abs(s)
        k = self.k
        n = 4
        x = x.unsqueeze(1)
        active_production = torch.sum(self.positive_mask * (a.unsqueeze(0) * torch.pow(x, n) / (torch.pow(s.unsqueeze(0), n) + torch.pow(x, n))), dim=-1)
        repress_production = torch.sum(self.negative_mask * (b.unsqueeze(0) * torch.pow(s.unsqueeze(0), n) / (torch.pow(x, n) + torch.pow(s.unsqueeze(0), n))), dim=-1)
        production = active_production + repress_production
        degradation = k * x.squeeze(1)
        force = production - degradation
        return force


    def forward(self, x):
        force = self.force(x)
        noiseD = self.variance_estimator(x).abs()
        random_numbers = torch.randn_like(x)*noiseD
        odestate = x + self.h * force
        sdestate = x + self.h * force + random_numbers* self.sqrth
        return force, odestate, sdestate


class Bool7D(nn.Module):
    def __init__(self, steps, h, gene_net):
        super(Bool7D, self).__init__()
        self.steps = steps
        self.h = h
        self.num_gene = gene_net.shape[0]
        self.gene_net = gene_net
        self.sqrth = torch.sqrt(torch.tensor(self.h))

        self.kernel = nn.Parameter(torch.rand((self.num_gene, 2 * self.num_gene))*5)
        self.s = nn.Parameter(torch.ones((self.num_gene)))
        self.D = nn.Parameter(torch.ones(self.num_gene))

        # Define variance estimation subnetwork
        self.variance_estimator = nn.Sequential(
            nn.Linear(self.num_gene, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_gene),
            nn.Softmax()
        )

    def force(self, x):
        num_gene = self.num_gene
        if self.gene_net.shape[0] != num_gene:
            raise IndexError('The number of genes is not equal to the shape of network.')
            
        a = torch.abs(self.a)
        b = torch.abs(self.b) 
        s = torch.abs(self.s)
        k = torch.ones_like(x[0, :])
        n = 4
        
        if a.shape[0] != num_gene or b.shape[0] != num_gene or s.shape[0] != num_gene:
            raise IndexError('The number of genes is not equal to the shape of parameters.')
            
        x_s = x / s.squeeze()  # [100, 7]
        x_expanded = x_s.unsqueeze(1)  # [100, 1, 7]
        
        # Calculate production and degradation forces
        x_pow = torch.pow(x_expanded, n)  # [100, 1, 7]
        active_term = x_pow / (1 + x_pow)  # [100, 1, 7]
        active_production = torch.sum(a.unsqueeze(0) * active_term, dim=-1)  # [100, 7]
        
        repress_term = 1 / (1 + x_pow)  # [100, 1, 7] 
        repress_production = torch.sum(b.unsqueeze(0) * repress_term, dim=-1)  # [100, 7]
        production = active_production + repress_production
        degradation = k * x.squeeze(1)

        # Calculate force
        force = production - degradation
        return force

    def forward(self, inputs):
        batch_size = inputs.size(0)
        ode_states = inputs
        sde_states = inputs
        ode_outputs = [inputs]
        sde_outputs = [inputs]
        
        for _ in range(self.steps):
            ode_force = self.force(ode_states)
            sde_force = self.force(sde_states)
            
            # Calculate variance
            variance = torch.exp(self.D)
            random_numbers = torch.randn_like(sde_states) * self.D
            
            sde_state = sde_states + self.h * sde_force + random_numbers * self.sqrth
            ode_state = ode_states + self.h * ode_force
            sde_state = torch.clamp(sde_state, min=0.0)
            
            ode_outputs.append(ode_state)
            ode_states = ode_state
            sde_outputs.append(sde_state)
            sde_states = sde_state
            
        return torch.stack(ode_outputs, dim=1), torch.stack(sde_outputs, dim=1)
