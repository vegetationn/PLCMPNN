import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
import os
import pickle
import numpy as np
import warnings
from utils import seed_everything, initialize_weights, scaffold_split, random_split
from dataloader import CmpnnDataModule
from dataloader.dataset import MoleculeDataset
from models import CMPNN_lightning


def main(args):
    # warning
    warnings.filterwarnings("ignore")
    # seed
    seed_everything(args.seed)
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # result folder
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # load total dataset
    with open(f'{args.data_path}/preprocess/{args.data_name}.pickle', 'rb') as f:
        # names = list(), others = dict() with key in names
        smiles, mols, graphs, labels = pickle.load(f)

    # split dataset(if need!)
    if not os.path.exists(f'{args.result_path}/{args.data_name}_split_{args.split_type}') or args.resplit:

        # build a folder
        if not os.path.exists(f'{args.result_path}/{args.data_name}_split_{args.split_type}'):
            os.makedirs(f'{args.result_path}/{args.data_name}_split_{args.split_type}')

        # scaffold split
        if args.split_type == 'scaffold':
            train_smiles, valid_smiles, test_smiles = scaffold_split(smiles, frac=[0.8, 0.1, 0.1], balanced=True, include_chirality=False, ramdom_state=args.seed)
        # random split
        elif args.split_type == 'random':
            train_smiles, valid_smiles, test_smiles = random_split(smiles, frac=[0.8, 0.1, 0.1], random_state=args.seed)
        else:
            raise "not supported split type, please refer the split type"

        # use pickle instead of txt since there are various smiles that correspond to the same molecule
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}/train.pickle', 'wb') as fw:
            pickle.dump(train_smiles, fw)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}/valid.pickle', 'wb') as fw:
            pickle.dump(valid_smiles, fw)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}/test.pickle', 'wb') as fw:
            pickle.dump(test_smiles, fw)

    # load pickle
    if args.split_type in ['scaffold', 'random']:
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}/train.pickle', 'rb') as f:
            train_smiles = pickle.load(f)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}/valid.pickle', 'rb') as f:
            valid_smiles = pickle.load(f)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}/test.pickle', 'rb') as f:
            test_smiles = pickle.load(f)
    else:
        raise "not supported split type, please refer the split type"

    # task setting
    if args.data_name in ['bace', 'bbbp', 'chembl', 'clintox', 'hiv', 'sider', 'tox21', 'toxcast']:
        args.task_type, args.task_loss, args.task_metric = 'classification', 'bce', 'auc'
    elif args.data_name in ['muv', 'pcba']:
        args.task_type, args.task_loss, args.task_metric = 'classification', 'bce', 'prc-auc'
    elif args.data_name in ['esol', 'freesolv', 'lipophilicity']:
        args.task_type, args.task_loss, args.task_metric = 'regression', 'mse', 'rmse'
    elif args.data_name in ['qm7', 'qm8', 'qm9']:
        args.task_type, args.task_loss, args.task_metric = 'regression', 'mse', 'mae'
    else:
        raise "Not supported task setting, please refer the correct data name!"

    args.task_number = len(labels[train_smiles[0]])

    # normalize label with the shape of (1, task_number)
    if args.task_type == 'regression':
        train_labels = [labels[smile] for smile in train_smiles]
        label_mean = torch.from_numpy(np.nanmean(train_labels, axis=0, keepdims=True)).float().to(device)
        label_std = torch.from_numpy(np.nanstd(train_labels, axis=0, keepdims=True)).float().to(device)
    else:
        label_mean = torch.from_numpy(np.array([[0 for _ in range(args.task_number)]])).long().to(device)
        label_std = torch.from_numpy(np.array([[1 for _ in range(args.task_number)]])).long().to(device)

    # # 创建结果文件夹
    result_dir = f'{args.result_path}/{args.data_name}_split_{args.split_type}/plcmpnn_seed_{args.seed}_batch_{args.batch_size}_lr_{args.learning_rate}'
    version = 0
    while os.path.exists(result_dir):
        version += 1
        result_dir = f'{args.result_path}/{args.data_name}_split_{args.split_type}/plcmpnn_seed_{args.seed}_batch_{args.batch_size}_lr_{args.learning_rate}_v{version}'
    os.makedirs(result_dir, exist_ok=True)

    # 实例化数据模块
    train_dataset = MoleculeDataset(train_smiles, mols, graphs, labels)
    (node_features_dim, edge_features_dim) = train_dataset.get_features_dim()
    datamodule = CmpnnDataModule(train_dataset, train_smiles, valid_smiles, test_smiles, mols, graphs, labels, args.batch_size, args.num_workers)
    # 实例化模型
    model = CMPNN_lightning(node_features_dim, edge_features_dim, args.hidden_features_dim, args.task_number, args.num_step_message_passing, args.learning_rate, args.max_epochs, args.task_loss, args.task_metric, label_mean, label_std, args.result_path, args.data_name, args.split_type, args.seed, args.batch_size, result_dir).to(device)

    # 初始化模型参数并打印
    print(f'\ntrain size: {len(train_smiles):,} | valid size: {len(valid_smiles):,} | test size: {len(test_smiles):,}\n')
    initialize_weights(model)
    print(model)

    # 创建 ModelCheckpoint 对象
    checkpoint_callback = ModelCheckpoint(monitor='valid_avg_loss', dirpath=f'{result_dir}/checkpoints/', filename='best_checkpoint', save_top_k=1, mode='min')

    # 自定义日志记录器
    logger = TensorBoardLogger(save_dir=f'{args.result_path}/{args.data_name}_split_{args.split_type}', name=f'plcmpnn_seed_{args.seed}_batch_{args.batch_size}_lr_{args.learning_rate}_logs')

    # 实例化训练器
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], num_sanity_val_steps=0, logger=logger)

    # 开始训练
    trainer.fit(model, datamodule)

    # 读取检查点参数并写入输出文件
    checkpoint_path = os.path.join(result_dir, 'checkpoints', 'best_checkpoint.ckpt')
    checkpoint = torch.load(checkpoint_path)

    epoch = checkpoint['epoch']
    key = "ModelCheckpoint{'monitor': 'valid_avg_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None, 'save_on_train_epoch_end': True}"
    valid_avg_loss = checkpoint['callbacks'][key]['best_model_score']

    print(f'Best Epoch: {epoch} | Best Loss: {valid_avg_loss:.4f}\n')

    with open(f'{result_dir}/results.txt', 'a') as f:
        f.write(f'\nBest Epoch: {epoch}  Best Loss: {valid_avg_loss:.4f}\n')

    # 加载检查点并创建新的模型对象
    test_model = CMPNN_lightning.load_from_checkpoint(checkpoint_path=f'{result_dir}/checkpoints/best_checkpoint.ckpt', node_features=node_features_dim, edge_features=edge_features_dim, hidden_features=args.hidden_features_dim, output_features=args.task_number, num_step_message_passing=args.num_step_message_passing, learning_rate=args.learning_rate, max_epochs=args.max_epochs, task_loss=args.task_loss, task_metric=args.task_metric, label_mean=label_mean, label_std=label_std, result_path=args.result_path, data_name=args.data_name, split_type=args.split_type, seed=args.seed, batch_size=args.batch_size, result_dir=result_dir).to(device)

    # 使用新的模型对象运行测试
    trainer.test(test_model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    # running setting
    parser.add_argument('--seed', type=int, default=666,
                        help="random seed")
    parser.add_argument('--gpus', type=int, default=1,
                        help="set gpu")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='The learning rate of ADAM optimization.')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='The maximum epoch of training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size')
    parser.add_argument('--result_path', type=str, default='./result',
                        help='The name of result path, for logs, predictions, best models, etc.')
    # dataset setting
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The full path of features of the data.')
    parser.add_argument('--data_name', type=str, default='bace',
                        help='the dataset name')
    parser.add_argument('--split_type', type=str, default='scaffold',
                        help="the dataset split type")
    parser.add_argument('--resplit', action='store_true', default=False,
                        help="resplit the dataset with different comments")
    # model setting
    parser.add_argument('--hidden_features_dim', type=int, default=300,
                        help='the hidden features dimension')
    parser.add_argument('--num_step_message_passing', type=int, default=3,
                        help="the number of CMPNN layers")
    # args executing
    args = parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    # 调用 main 函数，并将解析出来的命令行参数 args 传递给它
    main(args)
