
from tasks.R2R.Agents.agent import R2RAgent
from tasks.R2R.eval import DTW
import torch
import numpy as np
import sys
import copy
import os
import PIL

from tasks.R2R.Models import TransformerInstructionEncoder, ImagePlainEncoder, ImageEncoder
from tasks.R2R.Models import ActionDecoder, ActionDecoderHL
from tasks.R2R.utils import append_coordinates, batched_sentence_embedding, to_one_hot

sys.path.append('speaksee')
import speaksee.vocab as ssvoc


class Attentive(R2RAgent):

    env_actions = [
        (0, -1, 0),  # left
        (0, 1, 0),   # right

        (0, 0, 1),   # up
        (0, 0, -1),  # down

        (1, 0, 0),   # forward

        (0, 0, 0),   # <end>
        (0, 0, 0),   # <start>
    ]

    def __init__(self, config):
        super(Attentive, self).__init__(config)
        self.name = 'Attentive'
        self.mode = None

        self.teacher = config['teacher_forcing']
        self.device = config['device']
        self.max_episode_len = config['max_episode_len']

        self.d_model = config['d_model']
        self.h = config['h']
        self.N = config['n_layers']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']

        # low / high level attributes
        self.softmax = torch.nn.Softmax(dim=-1)
        self.hl_stop_index = 36
        self.action_n_output = 6
        self.d_embedding = 512
        self.action_in_features = self.d_embedding if config['action_space'] == 'high' else 8

        # rl attributes
        self.dtw_dict = dict()

        assert self.d_model % self.h == 0, "d_model ({}) is not divisible for number of heads ({})!".format(self.d_model, self.h)
        self.d_att = int(self.d_model / self.h)

        self.glove = ssvoc.GloVe(dim=300)
        self.ignore_index = -17
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.encoder = TransformerInstructionEncoder(N=self.N, d_in=300,
                                                     d_model=self.d_model, d_k=self.d_att, d_v=self.d_att,
                                                     h=self.h, d_ff=self.d_ff, dropout=self.dropout).to(device=self.device)

        self.image_encoder = ImageEncoder(N=self.N, in_features=2051,
                                          d_model=self.d_model, d_k=self.d_att, d_v=self.d_att,
                                          h=self.h, d_ff=self.d_ff, dropout=self.dropout).to(device=self.device)

        # self.image_encoder = ImagePlainEncoder(N=self.N, d_in=2051,
        #                                        d_model=self.d_model, d_k=self.d_att, d_v=self.d_att,
        #                                        h=self.h, d_ff=self.d_ff, dropout=self.dropout).to(device=self.device)  # ablation

        if config['action_space'] == 'low':
            self.action_decoder = ActionDecoder(N=self.N, in_features=self.action_in_features, n_output=self.action_n_output,
                                                d_model=self.d_model, d_k=self.d_att, d_v=self.d_att,
                                                h=self.h, d_ff=self.d_ff, dropout=self.dropout).to(device=self.device)
        else:
            self.action_decoder = ActionDecoderHL(N=self.N, in_features=self.action_in_features, d_embedding=self.d_embedding,
                                                  d_model=self.d_model, d_k=self.d_att, d_v=self.d_att, action_dropout=config['action_dropout'],
                                                  h=self.h, d_ff=self.d_ff, dropout=self.dropout).to(device=self.device)

    @staticmethod
    def n_inputs():
        return len(Attentive.env_actions)

    def train(self):
        self.mode = 'train'
        self.encoder.train()
        self.image_encoder.train()
        self.action_decoder.train()

    def eval(self):
        self.mode = 'eval'
        self.encoder.eval()
        self.image_encoder.eval()
        self.action_decoder.eval()

    def rl_train(self):
        self.mode = 'rl'
        self.encoder.train()
        self.image_encoder.train()
        self.action_decoder.train()

    def get_trainable_params(self):
        param_list = list(self.encoder.parameters()) + list(self.image_encoder.parameters()) + list(self.action_decoder.parameters())
        return param_list

    def get_trainable_parameters_number(self):
        return sum([np.prod(p.size()) for p in self.get_trainable_params()])

    def save(self, encoder_path, policy_path):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.image_encoder.state_dict(), policy_path+'_img')
        torch.save(self.action_decoder.state_dict(), policy_path+'_action')

    def load(self, base_path):
        enc_path = os.path.join(base_path, 'encoder_weights_best')
        hist_path = os.path.join(base_path, 'decoder_weights_best_hist')
        img_path = os.path.join(base_path, 'decoder_weights_best_img')
        action_path = os.path.join(base_path, 'decoder_weights_best_action')

        def load_module(module, path):
            pretrained_dict = torch.load(path)
            dictionary = module.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in dictionary}
            # 2. overwrite entries in the existing state dict
            dictionary.update(pretrained_dict)
            # 3. load the new state dict
            module.load_state_dict(pretrained_dict)

        load_module(self.encoder, enc_path)
        load_module(self.image_encoder, img_path)
        load_module(self.action_decoder, action_path)

    def _get_targets_and_features_low(self, obs):
        target_actions = []
        target_idx = []
        features = []
        local_features = []

        for i, ob in enumerate(obs):
            target_actions.append(
                ob['teacher'] if ob['teacher'] in self.env_actions else (1, 0, 0)
            )
            target_idx.append(self.env_actions.index(
                ob['teacher'] if ob['teacher'] in self.env_actions else (1, 0, 0)
            ))
            feat = (torch.from_numpy(ob['feature']))
            features.append(append_coordinates(feat, ob['heading'], ob['elevation']))

            local_features.append(torch.from_numpy(ob['local_feature']))

        return target_actions, torch.tensor(target_idx), torch.stack(features), torch.stack(local_features)

    def _encode_instruction_transformer(self, instructions):
        instr_embedding, instr_len = batched_sentence_embedding(instructions, self.glove, device=self.device)
        instr_embedding = instr_embedding.transpose(1, 2)

        mask = torch.ones((instr_embedding.shape[0], instr_embedding.shape[1]), dtype=torch.bool).cuda()
        attention_mask = torch.ones((instr_embedding.shape[0], instr_embedding.shape[1], instr_embedding.shape[1]), dtype=torch.bool).cuda()
        for i, _len in enumerate(instr_len):
            mask[i, :_len] = 0
            attention_mask[i, :_len, :_len] = 0

        pe_mask = mask.unsqueeze(dim=-1)

        value = (instr_embedding, pe_mask)
        return value, attention_mask.unsqueeze(1), mask.unsqueeze(dim=1).unsqueeze(dim=1)

    def _get_targets_and_features_high(self, obs):
        batch_size = len(obs)
        features = []
        local_features = []
        navigable_nodes = []  # list of dictionaries {'idx': 'idx.viewpointId', 'idx.heading'}
        target_nodes_list = []
        navigable_mask = torch.ones((batch_size, 37))
        navigable_mask[:, -1] = 0  # action "stop" is always available

        for i, ob in enumerate(obs):
            d = dict()

            feat = (torch.from_numpy(ob['feature']))
            features.append(append_coordinates(feat, ob['heading'], ob['elevation']))
            local_features.append(torch.from_numpy(ob['local_feature']))

            if ob['gt_viewpoint_idx'][0] != ob['viewpoint']:
                target_nodes_list.append(ob['gt_viewpoint_idx'][1])
            else:
                target_nodes_list.append(self.hl_stop_index)

            for navigable_idx in ob['navigableLocations']:

                if navigable_idx == ob['viewpoint']:  # does not add current viewpoint to navigable nodes
                    continue

                ix = int(ob['navigableLocations'][navigable_idx]['index'])
                navigable_mask[i, ix] = 0
                d[str(ix)] = [navigable_idx, ob['navigableLocations'][navigable_idx]['heading']]

                if ix == self.hl_stop_index:
                    print("ERROR: ix equals stop index!!")

            navigable_nodes.append(copy.deepcopy(d))

        target_nodes = torch.tensor(target_nodes_list)
        navigable_mask_logits = navigable_mask.gt(0.5)
        navigable_mask_feature = navigable_mask_logits[:, :36]

        return torch.stack(features), torch.stack(local_features), navigable_nodes, navigable_mask_feature, navigable_mask_logits, target_nodes

    def _next_viewpoint(self, obs, navigable_nodes, action, ended):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            idx = action[i]
            if idx < self.hl_stop_index:
                next_viewpoint_idx.append(idx)
                next_viewpoints.append(navigable_nodes[i][str(idx)][0])
                next_headings.append(navigable_nodes[i][str(idx)][1])
            else:
                next_viewpoint_idx.append('STAY')
                next_viewpoints.append(ob['viewpoint'])
                next_headings.append(ob['heading'])
                ended[i] = True

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def high_level_rollout(self, env):

        assert self.mode is not None, "This agent contains trainable modules! Please call either agent.train() or agent.eval() before rollout"
        assert self.mode in ['train', 'eval', 'rl'], "Agent.mode expected to be in ['train', 'eval', 'rl'], found %s" % self.mode

        obs = env.reset()
        ended = np.array([False] * len(obs))
        losses = []
        action_list = []
        hist_list = []
        losses_rl = []

        gt = [batch['path'][-1] for batch in env.batch]

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        scan_ids = [ob['scan'] for ob in obs]

        if self.mode == 'rl':
            gt_paths = [env.batch[n]['path'] for n in range(len(obs))]
            dtw = []
            for idx in scan_ids:
                if idx not in self.dtw_dict:  # updates internal dictionary with metric objects
                    self.dtw_dict[idx] = DTW(env.graphs[idx])
                dtw.append(self.dtw_dict[idx])
            assert len(dtw) != 0

            predictions = [[it[0] for it in traj[i]['path']] for i in range(len(obs))]
            reward_list = [metric(predictions[i], gt_paths[i], metric='ndtw') for i, metric in enumerate(dtw)]
            reward_past = torch.tensor(reward_list).to(device=self.device)

        instr = [ob['instructions'] for ob in obs]
        value, self_att_mask, enc_mask_w = self._encode_instruction_transformer(instr)

        """ init variables for first step """
        features, local_features, navigable_nodes, feature_mask, logit_mask, target_idx = self._get_targets_and_features_high(obs)
        start_token = torch.zeros((len(obs), self.action_in_features)).to(device=self.device)
        previous_action = start_token

        w_t = self.encoder(value, attention_mask=self_att_mask, attention_weights=None)

        for t in range(self.max_episode_len):

            image_features = features.to(device=self.device)

            v_t = image_features.masked_fill(feature_mask.unsqueeze(dim=-1).to(device=self.device), 0)
            i_t = self.image_encoder(image_features, w_t, None, enc_mask_w)

            action_list.append(previous_action)
            action_seq = torch.stack(action_list).transpose(0, 1).to(device=self.device)

            enc_mask_i = feature_mask.unsqueeze(dim=1).unsqueeze(dim=1).to(device=self.device)
            pred, ctx = self.action_decoder(action_seq, w_t, i_t, v_t, enc_att_mask_w=enc_mask_w, enc_att_mask_i=enc_mask_i)

            masked_pred = pred.masked_fill(logit_mask.to(device=self.device), -np.inf)
            probs = self.softmax(masked_pred)

            log_prob = None
            """ select action based on prediction """
            if self.mode == 'eval':
                p, a_t = probs.max(1)  # argmax
                log_prob = torch.log(p)
            else:
                if self.teacher:
                    a_t = target_idx
                else:
                    m = torch.distributions.Categorical(probs)  # sampling from distribution
                    a_t = m.sample()
                    log_prob = m.log_prob(a_t)

            actions = a_t.squeeze().tolist()

            """ Next step """
            next_viewpoints, next_headings, next_viewpoint_idx, ended = self._next_viewpoint(obs, navigable_nodes, actions, ended)
            hl_actions = [scan_ids, next_viewpoints, next_headings]
            obs = env.step(hl_actions)

            """ update trajectories """
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            """ Compute reward """
            if self.mode == 'rl':
                predictions = [[it[0] for it in traj[i]['path']] for i in range(len(obs))]
                reward_list = [metric(predictions[i], gt_paths[i], metric='ndtw') for i, metric in enumerate(dtw)]
                reward_next = torch.tensor(reward_list).to(device=self.device)
                reward = reward_next - reward_past
                reward_past = reward_next
                # Losses
                """ RL reward """
                assert log_prob is not None
                step_loss_rl = -1. * reward * log_prob
                losses_rl.append(step_loss_rl)
            else:
                """ XE Loss """
                xe_mask = torch.tensor(ended).unsqueeze(dim=-1).to(device=self.device)
                masked_pred = masked_pred.masked_fill(xe_mask, self.ignore_index)
                step_loss = self.criterion(masked_pred, target_idx.to(device=self.device))
                losses.append(step_loss)

            if ended.all():
                break

            """ update variables for next step """
            previous_action = ctx
            features, local_features, navigable_nodes, feature_mask, logit_mask, target_idx = self._get_targets_and_features_high(obs)

        """ Compute the loss for the whole rollout """
        if self.mode == 'rl':
            """ computes mean of rewards """
            episode_loss = torch.mean(torch.stack(losses_rl), dim=0)

            """ gives extra reward depending on success """
            d_th = 3.0
            d_goal = [env.distances[ob['scan']][ob['viewpoint']][_gt] for _gt, ob in zip(gt, obs)]
            d_goal = torch.tensor(d_goal).to(device=self.device)
            success_reward = torch.max(1 - d_goal / d_th, torch.zeros_like(d_goal))
            rollout_loss_rl = torch.mean(episode_loss - success_reward)
        else:
            losses = torch.stack(losses).to(device=self.device)
            rollout_loss_xe = torch.mean(losses)

        rollout_loss = rollout_loss_rl if self.mode == 'rl' else rollout_loss_xe
        return traj, rollout_loss

    def low_level_rollout(self, env):

        assert self.mode is not None, "This agent contains trainable modules! Please call either agent.train() or agent.eval() before rollout"
        assert self.mode in ['train', 'eval', 'rl'], "Agent.mode expected to be in ['train', 'eval', 'rl'], found %s" % self.mode

        obs = env.reset()
        ended = np.array([False] * len(obs))
        losses = []
        action_list = []
        hist_list = []
        losses_rl = []

        gt = [batch['path'][-1] for batch in env.batch]

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        if self.mode == 'rl':
            scan_ids = [ob['scan'] for ob in obs]
            gt_paths = [env.batch[n]['path'] for n in range(len(obs))]
            dtw = []
            for idx in scan_ids:
                if idx not in self.dtw_dict:  # updates internal dictionary with metric objects
                    self.dtw_dict[idx] = DTW(env.graphs[idx])
                dtw.append(self.dtw_dict[idx])
            assert len(dtw) != 0

            predictions = [[it[0] for it in traj[i]['path']] for i in range(len(obs))]
            reward_list = [metric(predictions[i], gt_paths[i], metric='ndtw') for i, metric in enumerate(dtw)]
            reward_past = torch.tensor(reward_list).to(device=self.device)

        instr = [ob['instructions'] for ob in obs]
        value, self_att_mask, enc_mask_w = self._encode_instruction_transformer(instr)

        target_actions, target_idx, features, local_features = self._get_targets_and_features_low(obs)
        previous_action = to_one_hot([Attentive.n_inputs() - 1] * len(obs), Attentive.n_inputs()+1)  # Action at t=0 is <start> for every agent

        w_t = self.encoder(value, attention_mask=self_att_mask, attention_weights=None)

        for t in range(self.max_episode_len):

            image_features = features.to(device=self.device)

            i_t = self.image_encoder(image_features, w_t, None, enc_mask_w)
            # i_t = self.image_encoder(image_features, None, None)  # ablation

            action_list.append(previous_action)
            action_seq = torch.stack(action_list).transpose(0, 1).to(device=self.device)
            pred, probs = self.action_decoder(action_seq, w_t, i_t, enc_att_mask_w=enc_mask_w, enc_att_mask_i=None)

            """ Performs steps """
            log_prob = None

            # Mask outputs where agent can't move forward
            tmp_mask = torch.zeros_like(probs).to(device=self.device)

            for i, ob in enumerate(obs):
                if len(ob['navigableLocations']) <= 1:
                    tmp_mask[i, self.env_actions.index((1, 0, 0))] = 1

            tmp_mask = tmp_mask.gt(0.5)
            probs = probs.masked_fill(tmp_mask, 0.)

            if self.mode == 'eval':
                p, a_t = probs.max(1)  # argmax
                actions = [self.env_actions[idx] for idx in a_t]
                log_prob = torch.log(p)
            else:
                if self.teacher:
                    assert self.mode != 'rl'
                    a_t = target_idx
                    actions = target_actions
                else:
                    m = torch.distributions.Categorical(probs)  # sampling from distribution
                    a_t = m.sample()
                    log_prob = m.log_prob(a_t)
                    actions = [self.env_actions[idx] for idx in a_t]

            """ Next step """
            obs = env.step(actions)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    if actions[i] == (0, 0, 0):
                        ended[i] = True
                    else:
                        traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            """ Compute reward """
            if self.mode == 'rl':
                predictions = [[it[0] for it in traj[i]['path']] for i in range(len(obs))]
                reward_list = [metric(predictions[i], gt_paths[i], metric='ndtw') for i, metric in enumerate(dtw)]
                reward_next = torch.tensor(reward_list).to(device=self.device)

                reward = reward_next - reward_past
                reward_past = reward_next

            # Losses
                """ RL reward """
                assert log_prob is not None
                step_loss_rl = -1. * reward * log_prob
                losses_rl.append(step_loss_rl)
            else:
                """ Xe loss """
                xe_mask = torch.tensor(ended).unsqueeze(dim=-1).to(device=self.device)
                masked_pred = pred.masked_fill(xe_mask, self.ignore_index)
                step_loss = self.criterion(masked_pred, target_idx.to(device=self.device))
                losses.append(step_loss)

            if ended.all():
                break

            target_actions, target_idx, features, local_features = self._get_targets_and_features_low(obs)
            previous_action = to_one_hot(a_t, Attentive.n_inputs()+1)

        """ Compute the loss for the whole rollout """
        if self.mode == 'rl':
            """ computes mean of rewards """
            episode_loss = torch.mean(torch.stack(losses_rl), dim=0)

            """ gives extra reward depending on success """
            d_th = 3.0
            d_goal = [env.distances[ob['scan']][ob['viewpoint']][_gt] for _gt, ob in zip(gt, obs)]
            d_goal = torch.tensor(d_goal).to(device=self.device)
            success_reward = torch.max(1 - d_goal / d_th, torch.zeros_like(d_goal))
            rollout_loss_rl = torch.mean(episode_loss - success_reward)
        else:
            losses = torch.stack(losses).to(device=self.device)
            rollout_loss_xe = torch.mean(losses)

        rollout_loss = rollout_loss_rl if self.mode == 'rl' else rollout_loss_xe
        return traj, rollout_loss
