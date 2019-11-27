
import torch
import numpy as np
import sys
import os

from tasks.R2R.Models import DynamicDecoder, HighLevelDynamicDecoder, InstructionEncoder, TransformerInstructionEncoder
from tasks.R2R.utils import append_coordinates, batched_sentence_embedding, to_one_hot, sinusoid_encoding_table

sys.path.append('speaksee')
import speaksee.vocab as ssvoc


class R2RAgent:

    low_level_actions = [
      (0, -1, 0),  # left
      (0, 1, 0),   # right
      (0, 0, 1),   # up
      (0, 0, -1),  # down
      (1, 0, 0),   # forward
      (0, 0, 0),   # <end>
    ]

    def __init__(self, config):
        self.config = config
        self.name = 'Base'

    def get_name(self):
        return self.name

    def get_config(self):
        return self.config

    def high_level_rollout(self, env):
        raise NotImplementedError

    def low_level_rollout(self, env):
        raise NotImplementedError

    def rollout(self, env):
        if self.config['action_space'] == 'low':
            return self.low_level_rollout(env)
        else:
            return self.high_level_rollout(env)

    def train(self):
        """ Should call Module.train() on each torch.nn.Module, if present """
        pass

    def eval(self):
        """ Should call Module.eval() on each torch.nn.Module, if present """
        pass


class Oracle(R2RAgent):
    def __init__(self, config):
        super(Oracle, self).__init__(config)
        self.name = 'Oracle'

    def high_level_rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))

        while True:
            scan_ids, next_viewpoints, next_headings, actions_idx = [], [], [], []

            for ob in obs:
                gt_viewpoint = ob['gt_viewpoint_idx'][0]
                scan_ids.append(ob['scan'])
                next_viewpoints.append(gt_viewpoint)
                next_headings.append(ob['navigableLocations'][gt_viewpoint]['heading'])
                actions_idx.append(list(ob['navigableLocations'].keys()).index(gt_viewpoint))

            actions = [scan_ids, next_viewpoints, next_headings]
            obs = env.step(actions)
            for i, a in enumerate(actions_idx):
                if a == 0:
                    ended[i] = True
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break

        return traj

    def low_level_rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))

        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = env.step(actions)
            for i, a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break

        return traj


class Stop(R2RAgent):
    def __init__(self, config):
        super(Stop, self).__init__(config)
        self.name = 'Stop'

    def high_level_rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        return traj

    def low_level_rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        return traj


class Random(R2RAgent):
    def __init__(self, config):
        super(Random, self).__init__(config)
        self.name = 'Random'

    def high_level_rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))

        for t in range(20):
            scan_ids, next_viewpoints, next_headings, actions_idx = [], [], [], []

            for ob in obs:
                next_action_idx = np.random.randint(0, len(ob['navigableLocations']))
                next_viewpoint = list(ob['navigableLocations'].keys())[next_action_idx]

                scan_ids.append(ob['scan'])
                next_viewpoints.append(next_viewpoint)
                next_headings.append(ob['navigableLocations'][next_viewpoint]['heading'])
                actions_idx.append(next_action_idx)

            actions = [scan_ids, next_viewpoints, next_headings]
            obs = env.step(actions)

            for i, a in enumerate(actions):
                if a == 0:
                    ended[i] = True
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break

        return traj

    def low_level_rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))

        for t in range(20):
            actions_idx = np.random.randint(0, len(R2RAgent.low_level_actions), len(obs))
            actions = [(0, 1, 0) if len(obs[i]['navigableLocations']) <= 1 and idx == R2RAgent.low_level_actions.index((1, 0, 0))
                       else R2RAgent.low_level_actions[idx] for i, idx in enumerate(actions_idx)]
            obs = env.step(actions)
            for i, a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break

        return traj


class Dynamic(R2RAgent):

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
        super(Dynamic, self).__init__(config)
        self.name = 'Dynamic'
        self.mode = None

        self.device = config['device']
        self.max_episode_len = config['max_episode_len']
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_heads = config['num_heads']
        self.glove = ssvoc.GloVe()
        self.lstm_input_size = 36 * self.num_heads

        self.encoder = InstructionEncoder(input_size=300,
                                          hidden_size=512,
                                          use_bias=True).to(device=self.device)

        if self.config['action_space'] == 'low':
            self.policy = DynamicDecoder(input_size=self.lstm_input_size + Dynamic.n_inputs(),
                                         hidden_size=512, output_size=6,
                                         key_size=128, query_size=128, value_size=512,
                                         image_size=2051, filter_size=512,
                                         num_heads=self.num_heads,
                                         drop_prob=0.2,
                                         use_bias=True,
                                         filter_activation=torch.nn.Tanh(),
                                         policy_activation=torch.nn.Softmax(dim=-1)).to(device=self.device)
        else:
            self.max_navigable = 16
            self.ignore_index = self.max_navigable + 1
            self.teacher = True
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            self.policy = HighLevelDynamicDecoder(input_size=self.lstm_input_size,
                                                  hidden_size=512, output_size=self.num_heads,
                                                  key_size=128, query_size=128, value_size=512,
                                                  image_size=2051, filter_size=512,
                                                  num_heads=self.num_heads,
                                                  drop_prob=0.2,
                                                  use_bias=True,
                                                  filter_activation=torch.nn.Tanh(),
                                                  policy_activation=torch.nn.Softmax(dim=-1)).to(device=self.device)

    @staticmethod
    def n_inputs():
        return len(Dynamic.env_actions)

    def train(self):
        self.mode = 'train'
        self.encoder.train()
        self.policy.train()

    def eval(self):
        self.mode = 'eval'
        self.encoder.eval()
        self.policy.eval()

    def save(self, encoder_path, policy_path):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.policy.state_dict(), policy_path)

    def load(self, base_path):
        enc_path = os.path.join(base_path, 'encoder_weights_best')
        action_path = os.path.join(base_path, 'decoder_weights_best')

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
        load_module(self.policy, action_path)

    def _get_targets_and_features_low(self, obs):
        target_actions = []
        target_idx = []
        features = []

        for i, ob in enumerate(obs):
            target_actions.append(
                ob['teacher'] if ob['teacher'] in self.env_actions else (1, 0, 0)
            )
            target_idx.append(self.env_actions.index(
                ob['teacher'] if ob['teacher'] in self.env_actions else (1, 0, 0)
            ))
            features.append(torch.from_numpy(ob['feature']))

        return target_actions, torch.tensor(target_idx), features

    def pano_navigable_feat(self, obs, ended):

        # Get the 36 image features for the panoramic view (including top, middle, bottom)
        num_feature, feature_size = obs[0]['feature'].shape

        pano_img_feat = torch.zeros(len(obs), num_feature, feature_size + 3)
        navigable_feat = torch.zeros(len(obs), self.max_navigable, feature_size + 3)

        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            features = torch.from_numpy(ob['feature'])  # pano feature: (batchsize, 36 directions, 2048)
            pano_img_feat[i, :] = append_coordinates(features, ob['heading'], ob['elevation'])

            index_list = []
            viewpoints_tmp = []
            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']

            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                index_list.append(int(ob['navigableLocations'][viewpoint_id]['index']))
                viewpoints_tmp.append(viewpoint_id)

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i]:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)

            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay"
            navi_index = index_list[1:]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)

            navigable_feat[i, 1:len(navi_index) + 1] = pano_img_feat[i, navi_index]

        return pano_img_feat.to(self.device), navigable_feat.to(self.device), (viewpoints, navigable_feat_index, target_index)

    @staticmethod
    def _next_viewpoint(obs, viewpoints, navigable_index, action, ended):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            if action[i] >= 1:
                next_viewpoint_idx.append(navigable_index[i][action[i] - 1])  # -1 because the first one in action is 'stop'
            else:
                next_viewpoint_idx.append('STAY')
                ended[i] = True

            # use the available viewpoints and action to select next viewpoint
            next_viewpoints.append(viewpoints[i][action[i]])
            # obtain the heading associated with next viewpoints
            next_headings.append(ob['navigableLocations'][next_viewpoints[i]]['heading'])

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def _encode_instruction_transformer(self, instructions):
        instr_embedding, instr_len = batched_sentence_embedding(instructions, self.glove, device=self.device)
        instr_embedding = instr_embedding.transpose(1, 2)

        positional_encoding = sinusoid_encoding_table(instr_embedding.shape[1], instr_embedding.shape[2])
        pe = positional_encoding.expand(instr_embedding.shape[0], positional_encoding.shape[0], positional_encoding.shape[1]).cuda()

        mask = torch.ones((instr_embedding.shape[0], instr_embedding.shape[1]), dtype=torch.bool).cuda()
        attention_mask = torch.ones((instr_embedding.shape[0], instr_embedding.shape[1], instr_embedding.shape[1]), dtype=torch.bool).cuda()
        for i, _len in enumerate(instr_len):
            mask[i, :_len] = 0
            attention_mask[i, :_len, :_len] = 0

        pe_mask = mask.unsqueeze(dim=-1)
        pe = pe.masked_fill(pe_mask, 0)
        instr_embedding = instr_embedding + pe

        value, _ = self.encoder(instr_embedding, attention_mask=attention_mask.unsqueeze(1), attention_weights=None)
        return value

    def _encode_instruction(self, instructions):
        instr_embedding, instr_len = batched_sentence_embedding(instructions, self.glove, device=self.device)

        value = self.encoder(instr_embedding)
        return value

    def get_trainable_params(self):
        return list(self.encoder.parameters()) + list(self.policy.parameters())

    def high_level_rollout(self, env):
        assert self.mode is not None, "This agent contains trainable modules! Please call either agent.train() or agent.eval() before rollout"
        assert self.mode in ['train', 'eval'], "Agent.mode expected to be in ['train', 'eval'], found %s" % self.mode

        obs = env.reset()
        ended = np.array([False] * len(obs))
        losses = []

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        scan_id = [(ob['scan']) for ob in obs]

        instr = [ob['instructions'] for ob in obs]
        value = self._encode_instruction(instr)

        for t in range(self.max_episode_len):
            """ panoramic features """
            pano_img_feat, navigable_feat, viewpoints_indices = self.pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices

            """ target """
            target = torch.LongTensor(target_index).to(self.device)

            """ forward """
            pred, logits, _ = self.policy(pano_img_feat, value, navigable_feat, navigable_index, init_lstm_state=t == 0)
            losses.append(self.criterion(pred, target))

            """ select action based on prediction """
            probs = logits.clone().detach().to(device=torch.device('cpu'))
            if self.mode == 'eval':
                _, a_t = probs.max(1)  # argmax
                actions = a_t
            else:
                if self.teacher:
                    _, a_t = probs.max(1)
                else:
                    m = torch.distributions.Categorical(probs)  # sampling from distribution
                    a_t = m.sample()
                actions = [0 if _ended else a_t[i] for i, _ended in enumerate(ended)]

            """ perform next action """
            next_viewpoints, next_headings, next_viewpoint_idx, ended = Dynamic._next_viewpoint(obs, viewpoints, navigable_index, actions, ended)
            actions = [scan_id, next_viewpoints, next_headings]
            obs = env.step(actions)

            """ update trajectories """
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break

        """ Compute the loss for the whole rollout """
        losses = torch.stack(losses).to(device=self.device)
        rollout_loss = torch.mean(losses)

        return traj, rollout_loss

    def low_level_rollout(self, env):

        assert self.mode is not None, "This agent contains trainable modules! Please call either agent.train() or agent.eval() before rollout"
        assert self.mode in ['train', 'eval'], "Agent.mode expected to be in ['train', 'eval'], found %s" % self.mode

        obs = env.reset()
        ended = np.array([False] * len(obs))
        losses = []

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        instr = [ob['instructions'] for ob in obs]
        value = self._encode_instruction(instr)

        target_actions, target_idx, features = self._get_targets_and_features_low(obs)
        previous_action = to_one_hot([Dynamic.n_inputs() - 1] * len(obs), Dynamic.n_inputs())  # Action at t=0 is <start> for every agent

        for t in range(self.max_episode_len):

            image_features = torch.stack(
                [append_coordinates(features[i], ob['heading'], ob['elevation']) for i, ob in enumerate(obs)]
            ).to(device=self.device)

            pred, logits, _ = self.policy(image_features, value, previous_action, t == 0)

            """ Losses """
            step_loss = self.criterion(pred, target_idx.to(device=self.device))
            losses.append(step_loss)

            """ Performs steps """
            # Mask outputs where agent can't move forward
            probs = logits.clone().detach().to(device=torch.device('cpu'))
            for i, ob in enumerate(obs):
                if len(ob['navigableLocations']) <= 1:
                    probs[i, self.env_actions.index((1, 0, 0))] = 0.

            if self.mode == 'eval':
                _, a_t = probs.max(1)  # argmax
                actions = [self.env_actions[idx] for idx in a_t]
            else:
                m = torch.distributions.Categorical(probs)  # sampling from distribution
                a_t = m.sample()
                actions = [self.env_actions[idx] if target_actions[i] != (0, 0, 0) else (0, 0, 0) for i, idx in enumerate(a_t)]

            """ Next step """
            obs = env.step(actions)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    if actions[i] == (0, 0, 0):
                        ended[i] = True
                    else:
                        traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            if ended.all():
                break

            target_actions, target_idx, features = self._get_targets_and_features_low(obs)
            previous_action = to_one_hot(a_t, Dynamic.n_inputs())

        """ Compute the loss for the whole rollout """
        losses = torch.stack(losses).to(device=self.device)
        rollout_loss = torch.mean(losses)

        return traj, rollout_loss
