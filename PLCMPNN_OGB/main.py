import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
import os
import numpy as np
import warnings
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from utils import seed_everything, initialize_weights
from dataloader import OgbDataModule
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

    dataset = DglGraphPropPredDataset(name=args.data_name, root=f'{args.result_path}')

    # task setting
    if args.data_name in ['ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molchembl', 'ogbg-molclintox', 'ogbg-molhiv', 'ogbg-molsider', 'ogbg-moltox21', 'ogbg-moltoxcast']:
        args.task_type, args.task_loss, args.task_metric = 'classification', 'bce', 'rocauc'
    elif args.data_name in ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']:
        args.task_type, args.task_loss, args.task_metric = 'regression', 'mse', 'rmse'
    elif args.data_name in ['ogbg-molmuv', 'ogbg-molpcba']:
        args.task_type, args.task_loss, args.task_metric = 'classification', 'bce', 'ap'
    elif args.data_name == 'ogbg-ppa':
        args.task_type, args.task_loss, args.task_metric = 'classification', 'bce', 'acc'
    elif args.data_name == 'ogbg-code2':
        args.task_type, args.task_loss, args.task_metric = 'classification', 'bce', 'F1'
    else:
        raise "Not supported task setting, please refer the correct data name!"

    args.task_number = max([len(label) if len(graph.edata['feat'].shape) > 1 else 0 for graph, label in dataset])

    # normalize label with the shape of (1, task_number)
    split_idx = dataset.get_idx_split()
    if args.task_type == 'regression':
        train_labels = np.concatenate([label.numpy() for _, label in dataset[split_idx["train"]]], axis=0).reshape(-1, 1)
        label_mean = torch.from_numpy(np.nanmean(train_labels, axis=0, keepdims=True)).float().to(device)
        label_std = torch.from_numpy(np.nanstd(train_labels, axis=0, keepdims=True)).float().to(device)
    else:
        label_mean = torch.from_numpy(np.array([[0 for _ in range(args.task_number)]])).long().to(device)
        label_std = torch.from_numpy(np.array([[1 for _ in range(args.task_number)]])).long().to(device)

    print(f'train size: {len(split_idx["train"]):,} | valid size: {len(split_idx["valid"]):,} | test size: {len(split_idx["test"]):,}')

    # evaluator
    evaluator = Evaluator(name=args.data_name)

    # 创建结果文件夹
    result_dir = f'{args.result_path}/{args.data_name.replace("-","_")}/ogbg_seed_{args.seed}_batch_{args.batch_size}_lr_{args.learning_rate}'
    version = 0
    while os.path.exists(result_dir):
        version += 1
        result_dir = f'{args.result_path}/{args.data_name.replace("-","_")}/ogbg_seed_{args.seed}_batch_{args.batch_size}_lr_{args.learning_rate}_v{version}'
    os.makedirs(result_dir, exist_ok=True)

    # 实例化数据模块
    datamodule = OgbDataModule(args.data_name, args.result_path, args.seed, args.batch_size, dataset, split_idx)
    # 实例化模型
    model = CMPNN_lightning(args.hidden_features_dim, args.task_number, args.num_step_message_passing, args.learning_rate, args.max_epochs, args.task_loss, args.task_metric, label_mean, label_std, args.result_path, args.data_name, args.seed, args.batch_size, evaluator, result_dir).to(device)

    # 初始化模型参数并打印
    initialize_weights(model)
    print(model)

    # 创建 ModelCheckpoint 对象
    checkpoint_callback = ModelCheckpoint(monitor='valid_avg_loss', dirpath=f'{result_dir}/checkpoints/', filename='best_checkpoint', save_top_k=1, mode='min')

    # 自定义日志记录器
    logger = TensorBoardLogger(save_dir=f'{args.result_path}/{args.data_name.replace("-","_")}', name=f'ogbg_seed_{args.seed}_batch_{args.batch_size}_lr_{args.learning_rate}_logs')

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
    model = CMPNN_lightning.load_from_checkpoint(checkpoint_path=f'{result_dir}/checkpoints/best_checkpoint.ckpt', hidden_features=args.hidden_features_dim, output_features=args.task_number, num_step_message_passing=args.num_step_message_passing, learning_rate=args.learning_rate, max_epochs=args.max_epochs, task_loss=args.task_loss, task_metric=args.task_metric, label_mean=label_mean, label_std=label_std, result_path=args.result_path, data_name=args.data_name, seed=args.seed, batch_size=args.batch_size, evaluator=evaluator, result_dir=result_dir).to(device)

    # 使用新的模型对象运行测试
    trainer.test(model, datamodule)


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
    parser.add_argument('--data_name', type=str, default='ogbg-moltox21',
                        help='the dataset name')
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
