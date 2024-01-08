import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from math import pi as PI
from models.gin import GINEncoder
from models.schnet import SchNetEncoder
from models.egnn import EGNN
from models.utils import *
__all__ = [
    'DualEncoderEpsNetwork',
    'DualDualEncoderEpsNetwork',
]

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super(MultiLayerPerceptron, self).__init__()

        self.layers = [nn.Linear(input_dim, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class MLPEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = nn.Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim])

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length) # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
        return d_emb * edge_attr # (num_edge, hidden)

class DualEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # print(config)
        self.config.cutoff = float(self.config.cutoff)

        try:
            self.SigmaAddWeight = config.SigmaAddWeight
            self.addt = config.addt
        except:
            self.SigmaAddWeight = False
            self.addt = False
        try:
            self.local_net = config.local_net
        except:
            self.local_net = 'gin'
        try:
            self.global_net = config.global_net
        except:
            self.global_net = 'schnet'  
        try:
            use_jump_in_gin = config.use_jump_in_gin
        except:
            use_jump_in_gin = False
         # ============================theta_1====================================================
        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = MLPEdgeEncoder(config.hidden_dim)
        self.edge_encoder_local = MLPEdgeEncoder(config.hidden_dim)

        """
        The graph neural network that extracts node-wise features.
        """
        if self.global_net == 'schnet':
            self.encoder_global = SchNetEncoder(
                addt=self.addt,
                hidden_channels=config.hidden_dim,
                num_filters=config.hidden_dim,
                num_interactions=config.num_convs,
                edge_channels=self.edge_encoder_global.out_channels,
                cutoff=config.cutoff,
                smooth=config.smooth_conv,
            )
        elif self.global_net == 'egnn':
            self.encoder_global = EGNN(
                in_node_nf=1,
                hidden_nf=config.hidden_dim,
                out_node_nf=config.hidden_dim,
                in_edge_nf=1
            )
        else:
            raise NotImplementedError(f'global net')
        if self.local_net == 'gin':
            self.encoder_local = GINEncoder(
                addt=self.addt,
                hidden_dim=config.hidden_dim,
                num_convs=config.num_convs_local,
                use_jump=use_jump_in_gin
            )
        elif self.local_net == 'egnn':
            self.encoder_local = EGNN(
                in_node_nf=1,
                hidden_nf=config.hidden_dim,
                out_node_nf=config.hidden_dim,
                in_edge_nf=1
            )
        else:
            raise NotImplementedError(f'local net')
        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
        )

        # ============================theta_1+theta_2_combine====================================================
        # '''
        # Incorporate parameters together
        # '''
        # self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp,
        #                                    self.edge_encoder_global2, self.encoder_global2, self.grad_global_dist_mlp2])
        # self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp,
        #                                   self.edge_encoder_local2, self.encoder_local2, self.grad_local_dist_mlp2])

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'


        # denoising diffusion
        # betas.shape == (num_diffusion_timesteps,)
        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        ## variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def forward(self, atom_type, pos, bond_index, bond_type, batch, time_step,
                edge_index=None, edge_type=None, edge_length=None, return_edges=False, 
                extend_order=True, extend_radius=True, is_sidechain=None, delta=0.01):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            pos:        Positions (i.e., the conformation), (N, 3)
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """

        '''
        varibles:
            self.betas:       (diff_steps, )
            self.alphas:      (diff_steps, )
            num_timesteps:    diff_steps
            atom_type:        (N, )
            pos:              (N, 3)
            bond_index:       (2, E)
            bond_type:        (E, )
            batch:            (N, )
            edge_index:       (2, e)
            edge_type:        (e, )
            edge_length:      (e, 1)
            local_edge_mask:  (e, )
            sigma_edge:       (e, 1)
            edge_attr_global: (e, hidden_dim)
            node_attr_global: (N, hidden_dim)
            h_pair_global:    (e, 2 * hidden_dim)
            edge_inv_global:  (e, 1)
            edge_attr_local:  (e, hidden_dim)
            node_attr_local:  (N, hidden_dim)
            h_pair_local:     (E_local, 2 * hidden_dim)
            edge_inv_local:   (E_local, 1)
        '''
        


        N = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )

            edge_length = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)

        local_edge_mask = is_local_edge(edge_type)  # (E, )

        edge2graph = batch.index_select(0, edge_index[0])
        a = self.alphas.index_select(0, time_step)  # (num_graphs, )
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)
        sigma_t_edge = (1.0 - a_edge).sqrt() / a_edge.sqrt()

        # =======global========
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )   # Embed edges
        # edge_attr_global.shape == (e, hidden_dim)

        # # Global
        if self.global_net == 'schnet':
            node_attr_global = self.encoder_global(
                z=atom_type,
                t=time_step[batch] / self.num_timesteps,
                edge_index=edge_index,
                edge_length=edge_length,
                edge_attr=edge_attr_global
            )
        # node_attr_global.shape == (N, hidden_dim)
        elif self.global_net == 'egnn':
            node_attr_global = self.encoder_global(
                h=atom_type.unsqueeze(-1).type(torch.float),
                x=pos,
                edges=edge_index,
                edge_attr=edge_type.unsqueeze(-1)
            )[0]
        else:
            raise NotImplementedError(f'global net')


        ## Assemble pairwise features
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )
        # h_pair_global.shape == (e, 2 * hidden_dim)


        ## Invariant features of edges (radius graph, global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global)
        # edge_inv_global.shape == (e, 1)

        # =======local========
        # Encoding local
        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type,

        )   # Embed edges
        # edge_attr_local.shape == (e, hidden_dim)

        # # Local
        if self.local_net == 'gin':
            node_attr_local = self.encoder_local(
                z=atom_type,
                edge_index=edge_index[:, local_edge_mask],
                edge_attr=edge_attr_local[local_edge_mask],
                t=batch
            )
        elif self.local_net == 'egnn':
            node_attr_local = self.encoder_local(
                h=atom_type.unsqueeze(-1).type(torch.float),
                x=pos,
                edges=edge_index,
                edge_attr=edge_type.unsqueeze(-1)
            )[0]
        else:
            raise NotImplementedError(f'local net')

        ## Assemble pairwise features
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )

        edge_inv_local = self.grad_local_dist_mlp(h_pair_local)

        return  edge_inv_global, edge_inv_local,\
                edge_index, edge_type, edge_length, local_edge_mask
    
    def get_loss(self, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                 extend_order=True, extend_radius=True, is_sidechain=None, it=0):

        N = atom_type.size(0)
        node2graph = batch


        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2+1, ), device=pos.device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)  # (G, )
        # Perterb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        # Update invariant edge features, as shown in equation 5-7
        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
            atom_type = atom_type,
            pos = pos_perturbed,
            bond_index = bond_index,
            bond_type = bond_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius,
            is_sidechain = is_sidechain
        )   # (E_global, 1), (E_local, 1)

        edge2graph = node2graph.index_select(0, edge_index[0])
        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        d_perturbed = edge_length
        # Filtering for protein
        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        if self.config.edge_encoder == 'gaussian':
            # Distances must be greater than 0 
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float('inf'))


        # d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()  # (E_global, 1), denoising direction
        a_node=a.index_select(0, node2graph).unsqueeze(-1)  # (E, 1)
        sigma_node = (1.0 - a_node).sqrt() / a_node.sqrt()
        sigma_edge = (1.0 - a_edge).sqrt() / a_edge.sqrt()
        mb_score = (1 - torch.exp(- sigma_edge / d_gt)) * 2 / d_perturbed
        mb_score = torch.where(torch.isnan(mb_score), torch.zeros_like(mb_score), mb_score) * sigma_edge
        
        # torch.max(torch.abs(mb_score)) \approx 10
        # print(torch.abs(mb_score).max())
        mb_score = torch.where(torch.logical_or(mb_score < -100, mb_score > 100), torch.zeros_like(mb_score), mb_score)

        # print(mb_score.max(), (d_perturbed - d_gt + 5).min(), sigma_edge.max(), sigma_edge.min())
        gaussion_score = - (d_perturbed - d_gt) / 2 / sigma_edge
        my_d_score = mb_score + gaussion_score
        d_target = my_d_score

        global_mask = torch.logical_and(
                            torch.logical_or(d_perturbed <= self.config.cutoff, local_edge_mask.unsqueeze(-1)),
                            ~local_edge_mask.unsqueeze(-1)
                        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global)**2
        # loss_global = loss_global * sigma_node
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)
        
        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = (node_eq_local - target_pos_local)**2
        loss_local = 5 * torch.sum(loss_local, dim=-1, keepdim=True)
        if it > 300:
            loss_global = torch.where(loss_global > 1000, torch.zeros_like(loss_global), loss_global)
            loss_local = torch.where(loss_local  > 1000, torch.zeros_like(loss_local), loss_local)
        loss_local = torch.where(loss_local.isinf(), torch.zeros_like(loss_local), loss_local)
        # # loss for atomic eps regression
        # if self.SigmaAddWeight:
        #     loss = loss_global * torch.exp(-sigma_node / 10) + loss_local * torch.exp(-sigma_node / 10)
        # else:
        

        # loss = loss.mean()
        loss_global = loss_global.mean()
        loss_local = loss_local.mean()
        loss = loss_global + loss_local
        # loss_pos = scatter_add(loss_pos.squeeze(), node2graph)  # (G, 1)
        return loss, loss_global, loss_local
    
    def langevin_dynamics_sample(self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius=True, 
                                 n_steps=1, step_lr=0.010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None, global_start_sigma=0.5, w_global=0.2, w_reg=1.0, **kwargs):

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        with torch.no_grad():
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])
            
            pos = pos_init * sigmas[-1]

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                # i: T-1, T-2, T-3, ..., T-n-1, T-n
                # j: T-2, T-3, T-4, ..., T-n,   -1
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)
                # ==================== previous sample ==========================
                edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain
                )
                '''
                pos:              (N, 3)

                edge_inv_global:  (e, 1)
                edge_inv_local:   (E_local, 1)
                edge_index:       (2, e)
                edge_length:      (e, 1)
                local_edge_mask:  (e, )
                '''
                sigma_t = sigmas[i]

                # Local
                node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
                # node_eq_local.shape == (N, 3)

                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                # Global
                # Global
                if sigmas[i] < global_start_sigma:
                    edge_inv_global = edge_inv_global * (1-local_edge_mask.view(-1, 1).float())
                    node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                # Sum
                eps_pos = node_eq_local + node_eq_global * w_global # + eps_pos_reg * w_reg

                # Update
                noise = torch.randn_like(pos)  #  center_pos(torch.randn_like(pos), batch)

                # step_size = step_lr * (sigmas[i] / 0.01) ** 2
                # pos_next = pos + step_size * eps_pos / torch.pow(sigmas[i], 3) + noise * torch.sqrt(step_size*2)
                # elif sampling_type == 'ld':
                step_size = step_lr * (sigmas[i] / 0.01) ** 2
                pos_next = pos + step_size * eps_pos / sigma_t + noise * torch.sqrt(step_size*2)
                

                if torch.isnan(pos_next).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                # pos_next = center_pos(pos_next, batch)
                pos = pos_next

                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                # pos_traj.append(pos.clone().cpu())
        return pos, pos_traj
    