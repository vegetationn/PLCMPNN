import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ogb.graphproppred import collate_dgl
from tqdm import tqdm


class OgbDataModule(pl.LightningDataModule):

    def __init__(self, data_name, result_path, seed, batch_size, dataset, split_idx):
        super().__init__()
        self.data_name = data_name
        self.result_path = result_path
        self.seed = seed
        self.batch_size = batch_size
        self.dataset = dataset
        self.split_idx = split_idx

    def prepare_data(self):
        # download/save data, do nothing as data is already provided
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_loader = DataLoader(self.dataset[self.split_idx["train"]], batch_size=self.batch_size, shuffle=True, collate_fn=collate_dgl)
            self.valid_loader = DataLoader(self.dataset[self.split_idx["valid"]], batch_size=self.batch_size, shuffle=True, collate_fn=collate_dgl)
        if stage == 'test' or stage is None:
            self.test_loader = DataLoader(self.dataset[self.split_idx["test"]], batch_size=self.batch_size, shuffle=True, collate_fn=collate_dgl)

    def train_dataloader(self):
        return tqdm(self.train_loader)

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader
