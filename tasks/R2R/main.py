
import argparse

import sys
import os
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.getcwd())

from tasks.R2R.Agents import get_agent
from tasks.R2R.env import load_features
from tasks.R2R.trainer import Trainer
from tasks.R2R.eval import Judge


parser = argparse.ArgumentParser(description='PyTorch for Matterport3D Agent with Dynamic Convolutional Filters')

# General options
parser.add_argument('--name', type=str, default='custom_experiment',
                    help='name for the experiment')
parser.add_argument('--results_dir', type=str, default='tasks/R2R/results',
                    help='home directory for results')
parser.add_argument('--feature_store', type=str, default='img_features/ResNet-152-imagenet.tsv',
                    help='feature store file')
parser.add_argument('--eval_only', action="store_true",
                    help='if true, does not train the model before evaluating')
parser.add_argument('--seed', type=int, default=42,
                    help='initial random seed')
parser.add_argument('--high_level', action="store_true",
                    help='if set, uses high level configuration for environment and agent')
parser.add_argument('--submission_test', action="store_true",
                    help='if set, creates server submission')

# Training options
parser.add_argument('--num_epoch', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--eval_every', type=int, default=1,
                    help='number of training epochs between evaluations')
parser.add_argument('--patience', type=int, default=30,
                    help='number of epochs to wait before early stopping')
parser.add_argument('--lr', type=float, default=1.0,
                    help='base learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--warmup', type=int, default=4000,
                    help='scheduler warmup steps')
parser.add_argument('--plateau_sched', action="store_true",
                    help='if set, uses reduceOnPlateau lr scheduler')
parser.add_argument('--scheduler_patience', type=int, default=5,
                    help='reduceOnPlateau scheduler patience')
parser.add_argument('--reinforce', action="store_true",
                    help='if set, uses reinforce on ndtw metric')
parser.add_argument('--pretrained', action="store_true",
                    help='if set, loads pretrained weights')
parser.add_argument('--load_from', type=str, default='best',
                    help='name for the experiment to load as pretrained model')
parser.add_argument('--da', action="store_true",
                    help='if set, uses data augmentation with synthetic instructions')
parser.add_argument('--r4r', action="store_true",
                    help='if set, uses r4r dataset')

# Agent options
parser.add_argument('--dynamic', action="store_true",
                    help='if set, uses dynamic agent')
parser.add_argument('--num_heads', type=int, default=1,
                    help='number of heads for multi-headed dynamic convolution')
parser.add_argument('--max_episode_len', type=int, default=20,
                    help='agent max number of steps before stopping')
parser.add_argument('--teacher_forcing', action="store_true",
                    help='if set, uses teacher forcing')
parser.add_argument('--history', action="store_true",
                    help='if set, uses history')
parser.add_argument('--action_dropout', type=float, default=0.2,
                    help='action dropout drop probability')

# Transformer options
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of transformer layers')
parser.add_argument('--d_model', type=int, default=512,
                    help='transformer model dimension')
parser.add_argument('--transformer_heads', type=int, default=8,
                    help='number of attention heads for multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048,
                    help='dimension of the intermediate layer in feed-forward sub-network')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout drop probability')


""" Device info """
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Running on device: {}'.format(device))


def main(opts):

    agent_type = 'Dynamic' if opts.dynamic else 'Transformer_based'

    if opts.r4r:
        splits = 'R4R_train'
    else:
        splits = ['synthetic', 'train'] if opts.da else 'train'

    space = 'high' if opts.high_level else 'low'

    results_path = os.path.join(opts.results_dir, opts.name)

    agent_config = {'n_layers': opts.num_layers,
                    'd_model': opts.d_model,
                    'h': opts.transformer_heads,
                    'teacher_forcing': opts.teacher_forcing,
                    'action_space': space,
                    'max_episode_len': opts.max_episode_len,
                    'num_heads': opts.num_heads,
                    'd_ff': opts.d_ff,
                    'dropout': opts.dropout,
                    'history': opts.history,
                    'device': device,
                    'action_dropout': opts.action_dropout,
                    }

    agent = get_agent(agent_type, agent_config)
    features, img_spec = load_features(opts.feature_store)

    trainer_config = {'action_space': space,
                      'features': features,
                      'img_spec': img_spec,
                      'splits': splits,
                      'batch_size': opts.batch_size,
                      'seed': opts.seed,
                      'results_path': results_path,
                      'exp_name': opts.name,
                      'plateau_sched': opts.plateau_sched,
                      'reinforce': opts.reinforce,
                      }

    if opts.r4r:
        test_split = ['R4R_val_unseen']
    elif opts.submission_test:
        test_split = ['val_seen', 'val_unseen', 'test']
    else:
        test_split = ['val_seen', 'val_unseen']

    judge_config = {'action_space': space,
                    'features': features,
                    'img_spec': img_spec,
                    'splits': test_split,
                    'batch_size': opts.batch_size,
                    'seed': opts.seed,
                    'results_path': results_path,
                    'main_split': 'R4R_val_unseen' if opts.r4r else 'val_unseen',
                    }

    judge = Judge(judge_config)

    if opts.pretrained:
        base_path = os.path.join(args.results_dir, opts.load_from)
        print('Loading agent weights from: {}'.format(base_path))
        agent.load(base_path)

        with torch.no_grad():
            print('Testing pretrained weights...')
            metric, metric_dict = judge.test(agent)
            if metric is not None:
                print('Main metric result for this test: {:.4f}'.format(metric))

        if opts.eval_only:
            return

    elif opts.eval_only:
        with torch.no_grad():
            print('Testing...')
            metric, metric_dict = judge.test(agent)
            print('Main metric result for this test: {:.4f}'.format(metric))
        return

    trainer = Trainer(trainer_config)

    optimizer = optim.Adam(agent.get_trainable_params(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

    if opts.plateau_sched:
        print('Using reduceOnPlateau scheduler, base lr: {}'.format(opts.lr))
        patience = opts.scheduler_patience
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience, verbose=True, min_lr=1e-6)
    else:
        print('Using transformer scheduler, base lr: {} - warmup steps: {}'.format(opts.lr, opts.warmup))

        def lambda_lr(s):
            warm_up = opts.warmup
            s += 1
            return (agent.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    best = trainer.train(agent, optimizer, scheduler, opts.num_epoch, patience=opts.patience, eval_every=opts.eval_every, judge=judge)
    print('Best metric result for this test: {:.4f}'.format(best))

    print('----- End -----')


if __name__ == '__main__':
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.results_dir, args.name)):
        print('WARNING: Experiment with this name already exists! - {}'.format(args.name))
    else:
        os.makedirs(os.path.join(args.results_dir, args.name))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
