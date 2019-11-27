
import os

from tasks.R2R.env import env_list
from tasks.R2R.utils import check_config_trainer, print_progress

import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, config):
        self.results = dict()
        self.config = check_config_trainer(config)
        features = self.config.pop('features')
        self.env = env_list[config['action_space']](features=features,
                                                    img_spec=config['img_spec'],
                                                    batch_size=config['batch_size'],
                                                    seed=config['seed'],
                                                    splits=config['splits']
                                                    )

        self.log_dir = os.path.join(config['results_path'], '..', 'tensorboard_log', config['exp_name'])
        self.summaries = SummaryWriter(self.log_dir)

    def _train_epoch(self, agent, optimizer, scheduler, num_iter):
        self.env.reset_epoch()

        if self.config['reinforce']:  # training with reinforcement learning
            agent.rl_train()
        else:  # training with imitation learning
            agent.train()

        epoch_loss = 0.

        for it in range(num_iter):
            optimizer.zero_grad()
            _, loss = agent.rollout(self.env)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            suffix_msg = 'Running Loss: {:.6f}'.format(epoch_loss / (it+1))
            print_progress(it, num_iter, suffix=suffix_msg)

            if not self.config['plateau_sched']:
                scheduler.step()  # here
        else:
            suffix_msg = 'Running Loss: {:.4f}'.format(epoch_loss / num_iter)
            print_progress(num_iter, num_iter, suffix=suffix_msg)

        return epoch_loss / num_iter

    def train(self, agent, optimizer, scheduler, num_epoch, num_iter_epoch=None, patience=None, eval_every=None, judge=None):
        best_metric = 0.
        mean_loss = 0.
        epoch = 0

        train_sr, val_seen_sr, val_unseen_sr = 0., 0., 0.
        train_spl, val_seen_spl, val_unseen_spl = 0., 0., 0.

        train_sr_best, val_seen_sr_best, val_unseen_sr_best = 0., 0., 0.
        train_spl_best, val_seen_spl_best, val_unseen_spl_best = 0., 0., 0.

        self.summaries.add_text('agent_config', str(agent.config).replace('\n', '\n\n'))
        self.summaries.add_text('trainer_config', str(self.config).replace('\n', '\n\n'))
        if judge is not None:
            self.summaries.add_text('judge_config', str(judge.config).replace('\n', '\n\n'))

        self.summaries.add_scalar('best_metric', best_metric, epoch)
        self.summaries.add_scalar('train_loss', mean_loss, epoch)

        if 'train' in self.config['splits']:
            self.summaries.add_scalar('train_sr', train_sr, epoch)
            self.summaries.add_scalar('train_spl', train_spl, epoch)
            self.summaries.add_scalar('train_sr_best', train_sr_best, epoch)
            self.summaries.add_scalar('train_spl_best', train_spl_best, epoch)
        if 'val_seen' in self.config['splits']:
            self.summaries.add_scalar('val_seen_sr', val_seen_sr, epoch)
            self.summaries.add_scalar('val_seen_spl', val_seen_spl, epoch)
            self.summaries.add_scalar('val_seen_sr_best', val_seen_sr_best, epoch)
            self.summaries.add_scalar('val_seen_spl_best', val_seen_spl_best, epoch)
        if 'val_unseen' in self.config['splits']:
            self.summaries.add_scalar('val_unseen_sr', val_unseen_sr, epoch)
            self.summaries.add_scalar('val_unseen_spl', val_unseen_spl, epoch)
            self.summaries.add_scalar('val_unseen_sr_best', val_unseen_sr_best, epoch)
            self.summaries.add_scalar('val_unseen_spl_best', val_unseen_spl_best, epoch)

        if num_iter_epoch is None:
            num_iter_epoch = len(self.env.data) // self.env.batch_size + 1
        if eval_every is None:
            if judge is None:
                eval_every = num_epoch + 1  # Never tested
            else:
                eval_every = num_epoch  # Test only on the last epoch
        if patience is None:
            patience = num_epoch
        reset_patience = patience

        while epoch <= num_epoch:
            epoch += 1
            mean_loss = self._train_epoch(agent, optimizer, scheduler, num_iter_epoch)
            print("Epoch {}/{} terminated: Epoch Loss = {:.4f}".format(epoch, num_epoch, mean_loss))
            agent.save(os.path.join(self.config['results_path'], 'encoder_weights_last'),
                       os.path.join(self.config['results_path'], 'decoder_weights_last'))

            if epoch % eval_every == 0:
                with torch.no_grad():
                    metric, metric_dict = judge.test(agent)

                if self.config['plateau_sched'] and metric is not None:
                    scheduler.step(metric)  # here

                if metric is not None:
                    print('Main metric results for this test: {:.4f}'.format(metric))
                    if metric > best_metric:
                        best_metric = metric
                        patience = reset_patience
                        print('New best! Saving weights...')
                        agent.save(os.path.join(self.config['results_path'], 'encoder_weights_best'),
                                   os.path.join(self.config['results_path'], 'decoder_weights_best'))
                    else:
                        patience -= eval_every
                        if patience <= 0:
                            print('{} epochs without improvement in main metric ({}) - patience is over!'.format(reset_patience, judge.main_metric))
                            break

                if 'train' in metric_dict:
                    train_sr = metric_dict['train']['success_rate']
                    train_spl = metric_dict['train']['spl']

                    if train_sr > train_sr_best:
                        train_sr_best = train_sr
                        self.summaries.add_scalar('train_sr_best', train_sr_best, epoch)

                    if train_spl > train_spl_best:
                        train_spl_best = train_spl
                        self.summaries.add_scalar('train_spl_best', train_spl_best, epoch)

                    self.summaries.add_scalar('train_sr', train_sr, epoch)
                    self.summaries.add_scalar('train_spl', train_spl, epoch)

                if 'val_seen' in metric_dict:
                    val_seen_sr = metric_dict['val_seen']['success_rate']
                    val_seen_spl = metric_dict['val_seen']['spl']

                    if val_seen_sr > val_seen_sr_best:
                        val_seen_sr_best = val_seen_sr
                        self.summaries.add_scalar('val_seen_sr_best', val_seen_sr_best, epoch)

                    if val_seen_spl > val_seen_spl_best:
                        val_seen_spl_best = val_seen_spl
                        self.summaries.add_scalar('val_seen_spl_best', val_seen_spl_best, epoch)

                    self.summaries.add_scalar('val_seen_sr', val_seen_sr, epoch)
                    self.summaries.add_scalar('val_seen_spl', val_seen_spl, epoch)

                if 'val_unseen' in metric_dict:
                    val_unseen_sr = metric_dict['val_unseen']['success_rate']
                    val_unseen_spl = metric_dict['val_unseen']['spl']

                    if val_unseen_sr > val_unseen_sr_best:
                        val_unseen_sr_best = val_unseen_sr
                        self.summaries.add_scalar('val_unseen_sr_best', val_unseen_sr_best, epoch)

                    if val_unseen_spl > val_unseen_spl_best:
                        val_unseen_spl_best = val_unseen_spl
                        self.summaries.add_scalar('val_unseen_spl_best', val_unseen_spl_best, epoch)

                    self.summaries.add_scalar('val_unseen_sr', val_unseen_sr, epoch)
                    self.summaries.add_scalar('val_unseen_spl', val_unseen_spl, epoch)

                if 'val_loss' in metric_dict:
                    self.summaries.add_scalar('validation_loss', metric_dict['val_loss'], epoch)

                self.summaries.add_scalar('best_metric', best_metric, epoch)
                self.summaries.add_scalar('train_loss', mean_loss, epoch)

        print("Finishing training")
        return best_metric
