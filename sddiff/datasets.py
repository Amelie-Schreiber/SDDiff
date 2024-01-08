import pickle
import copy
import torch
from collections import defaultdict
from torch_geometric.data import Dataset

class ConformationDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):
        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.data)

        
    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)

class PackedConformationDataset(ConformationDataset):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)
        #k:v = idx: data_obj
        self._pack_data_by_mol()

    def _pack_data_by_mol(self):
        """
        pack confs with same mol into a single data object
        """
        self._packed_data = defaultdict(list)
        if hasattr(self.data, 'idx'):
            for i in range(len(self.data)):
                self._packed_data[self.data[i].idx.item()].append(self.data[i])
        else:
            for i in range(len(self.data)):
                self._packed_data[self.data[i].smiles].append(self.data[i])
        print('[Packed] %d Molecules, %d Conformations.' % (len(self._packed_data), len(self.data)))

        new_data = []
        # logic
        # save graph structure for each mol once, but store all confs 
        cnt = 0
        for k, v in self._packed_data.items():
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                all_pos.append(v[i].pos)
            data.pos_ref = torch.cat(all_pos, 0) # (num_conf*num_node, 3)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            #del data.pos

            if hasattr(data, 'totalenergy'):
                del data.totalenergy
            if hasattr(data, 'boltzmannweight'):
                del data.boltzmannweight
            new_data.append(data)
        self.new_data = new_data

        

    def __getitem__(self, idx):

        data = self.new_data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.new_data)
