import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset import collate_fn


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, mols_dict, graphs_dict, labels_dict):
        super(MoleculeDataset, self).__init__()
        self.smiles = smiles_list
        self.mols = [mols_dict[smile] for smile in smiles_list]
        self.graphs = [graphs_dict[smile] for smile in smiles_list]
        self.labels = [labels_dict[smile] for smile in smiles_list]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.mols[idx], self.graphs[idx], self.labels[idx]

    def get_features_dim(self):
        return max([graph.ndata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs]), \
            max([graph.edata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs])


class CmpnnDataModule(pl.LightningDataModule):

    def __init__(self, train_dataset, train_smiles, valid_smiles, test_smiles, mols, graphs, labels, batch_size, num_workers):
        super().__init__()
        self.train_dataset = train_dataset
        self.train_smiles = train_smiles
        self.valid_smiles = valid_smiles
        self.test_smiles = test_smiles
        self.mols = mols
        self.graphs = graphs
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download/save data, do nothing as data is already provided
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

            valid_dataset = MoleculeDataset(self.valid_smiles, self.mols, self.graphs, self.labels)
            self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

        if stage == 'test' or stage is None:
            test_dataset = MoleculeDataset(self.test_smiles, self.mols, self.graphs, self.labels)
            self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def train_dataloader(self):
        return tqdm(self.train_loader)

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader
