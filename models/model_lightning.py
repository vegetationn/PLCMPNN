import torch
import pytorch_lightning as pl
import numpy as np
from models.model import CMPNN
from utils import cal_loss, cal_metric, NoamLR


class CMPNN_lightning(pl.LightningModule):

    def __init__(self, node_features, edge_features, hidden_features, output_features, num_step_message_passing, learning_rate, max_epochs, task_loss=None, task_metric=None, label_mean=None, label_std=None, result_path=None, data_name=None, split_type=None, seed=None, batch_size=None, result_dir=None):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.num_step_message_passing = num_step_message_passing
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.task_loss = task_loss
        self.task_metric = task_metric
        self.label_mean = label_mean
        self.label_std = label_std
        self.result_path = result_path
        self.data_name = data_name
        self.split_type = split_type
        self.seed = seed
        self.batch_size = batch_size
        self.result_dir = result_dir

        # model
        self.model = CMPNN(node_features, edge_features, hidden_features, output_features, num_step_message_passing)

    def forward(self, graphs):
        return self.model(graphs)

    def configure_optimizers(self):
        # 初始化优化器和学习率调度器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[2], total_epochs=[self.max_epochs], steps_per_epoch=len(self.trainer.datamodule.train_dataloader()), init_lr=[self.learning_rate], max_lr=[self.learning_rate * 10], final_lr=[self.learning_rate])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        # 训练步骤
        smiles, mols, graphs, labels = batch
        graphs = graphs.to(self.device)
        labels = labels.to(self.device)
        output = self(graphs)
        loss = cal_loss(labels, output, self.task_loss, self.label_mean, self.label_std, self.device)
        return {'loss': loss, 'y_true': labels, 'y_pred': output}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean() / self.batch_size
        self.log('train_avg_loss', avg_loss, prog_bar=True)

        y_true = torch.cat([x['y_true'] for x in outputs]).detach().cpu().numpy().tolist()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).detach().cpu().numpy().tolist()
        metric = cal_metric(np.array(y_true), np.array(y_pred), self.task_metric, self.label_mean.detach().cpu().numpy(), self.label_std.detach().cpu().numpy())
        self.log("train_metric", metric, on_step=False, on_epoch=True, prog_bar=True)

        with open(f'{self.result_dir}/results.txt', 'a') as f:
            f.write(f'Epoch: {self.current_epoch+1}\ntrain: {self.task_metric}: {metric.item():.4f}   loss: {avg_loss.item():.4f}\n')

    def validation_step(self, batch, batch_idx):
        # 验证步骤
        smiles, mols, graphs, labels = batch
        graphs = graphs.to(self.device)
        labels = labels.to(self.device)
        output = self(graphs)
        val_loss = cal_loss(labels, output, self.task_loss, self.label_mean, self.label_std, self.device)
        return {'loss': val_loss, 'y_true': labels, 'y_pred': output}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean() / self.batch_size
        self.log('valid_avg_loss', avg_loss, prog_bar=True)

        y_true = torch.cat([x['y_true'] for x in outputs]).detach().cpu().numpy().tolist()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).detach().cpu().numpy().tolist()
        metric = cal_metric(np.array(y_true), np.array(y_pred), self.task_metric, self.label_mean.detach().cpu().numpy(), self.label_std.detach().cpu().numpy())
        self.log("val_metric", metric, on_step=False, on_epoch=True, prog_bar=True)

        with open(f'{self.result_dir}/results.txt', 'a') as f:
            f.write(f'Epoch: {self.current_epoch+1}\nvalid: {self.task_metric}: {metric.item():.4f}   loss: {avg_loss.item():.4f}\n')

    def test_step(self, batch, batch_idx):
        # 测试步骤
        smiles, mols, graphs, labels = batch
        graphs = graphs.to(self.device)
        labels = labels.to(self.device)
        output = self(graphs)
        test_loss = cal_loss(labels, output, self.task_loss, self.label_mean, self.label_std, self.device)
        return {'loss': test_loss, 'y_true': labels, 'y_pred': output}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean() / self.batch_size
        self.log('test_avg_loss', avg_loss, prog_bar=True)

        y_true = torch.cat([x['y_true'] for x in outputs]).detach().cpu().numpy().tolist()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).detach().cpu().numpy().tolist()
        metric = cal_metric(np.array(y_true), np.array(y_pred), self.task_metric, self.label_mean.detach().cpu().numpy(), self.label_std.detach().cpu().numpy())
        self.log("test_metric", metric, on_step=False, on_epoch=True, prog_bar=True)

        with open(f'{self.result_dir}/results.txt', 'a') as f:
            f.write(f'\ntest: {self.task_metric}: {metric.item():.4f}   test loss: {avg_loss.item():.4f}\n')
