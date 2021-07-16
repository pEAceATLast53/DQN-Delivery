import copy
from dqn.modules.cnn import CNN
from dqn.modules.fc import FC1, FC2, FC3
from dqn.replay_buffer import ReplayBuffer

import torch
from torch.optim import RMSprop


class QLearner:
    def __init__(self, args, writer):
        self.args = args

        self.coord_fc = FC2(3, 128, 128)
        self.map_cnn = CNN(args)
        self.shared_fc = FC3(259, 512, 5)

        self.params = list(self.shared_fc.parameters())
        self.params += list(self.coord_fc.parameters())
        self.params += list(self.map_cnn.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_coord_fc = copy.deepcopy(self.coord_fc)
        self.target_map_cnn = copy.deepcopy(self.map_cnn)
        self.target_shared_fc = copy.deepcopy(self.shared_fc)

        self.replay_buffer = ReplayBuffer(args)
        self.writer = writer
        self.count = 0

    def train(self):
        self.count += 1
        batch_a, batch_r, batch_d, batch_obs_map, batch_obs_coord, batch_obs_prev_action, batch_obs_pos, \
            batch_obs_map_next, batch_obs_coord_next, batch_obs_prev_action_next, batch_obs_pos_next = self.replay_buffer.sample()

        coord_out = torch.sum(self.coord_fc(batch_obs_coord), 1)
        map_out = self.map_cnn(batch_obs_map.permute(0, 3, 1, 2))
        q_out = self.shared_fc(torch.cat([coord_out, map_out, batch_obs_prev_action, batch_obs_pos], -1))
        chosen_q_out = torch.gather(q_out, dim=-1, index=batch_a)

        with torch.no_grad():
            target_coord_out = torch.sum(self.target_coord_fc(batch_obs_coord_next), 1)
            target_map_out = self.target_map_cnn(batch_obs_map_next.permute(0, 3, 1, 2))
            target_q_out = batch_r + 0.99 * (1 - batch_d) * torch.max(self.shared_fc(torch.cat([target_coord_out, target_map_out, batch_obs_prev_action_next, batch_obs_pos_next], dim=-1)), dim=-1)[0]

        self.loss = ((chosen_q_out - target_q_out) ** 2).mean()
        
        self.optimiser.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimiser.step()

        self.writer.add_scalar('Loss', self.loss.item(), self.count)
        self.writer.add_scalar('Q Mean', chosen_q_out.mean().item(), self.count)

    def select_action(self, obs_coord, obs_map, obs_prev_action, obs_pos):
        with torch.no_grad():
            coord_out = torch.sum(self.coord_fc(obs_coord), 1)
            map_out = self.map_cnn(obs_map.permute(0, 3, 1, 2))
            q_out = self.shared_fc(torch.cat([coord_out, map_out, obs_prev_action, obs_pos], -1)).squeeze(0)
        return torch.argmax(q_out).item()

    def update_targets(self):
        self.target_coord_fc.load_state_dict(self.coord_fc.state_dict())
        self.target_map_cnn.load_state_dict(self.map_cnn.state_dict())
        self.target_shared_fc.load_state_dict(self.shared_fc.state_dict())

    def cuda(self):
        self.coord_fc.cuda()
        self.target_coord_fc.cuda()
        self.map_cnn.cuda()
        self.target_map_cnn.cuda()
        self.shared_fc.cuda()
        self.target_shared_fc.cuda()

    def save_models(self, path):
        torch.save(self.coord_fc.state_dict(), "{}/coord_fc.th".format(path))
        torch.save(self.map_cnn.state_dict(), "{}/map_cnn.th".format(path))
        torch.save(self.shared_fc.state_dict(), "{}/shared_fc.th".format(path))

    def load_models(self, path):
        self.coord_fc.load_state_dict(torch.load("{}/coord_fc.th".format(path), map_location=lambda storage, loc:storage))
        self.map_cnn.load_state_dict(torch.load("{}/map_cnn.th".format(path), map_location=lambda storage, loc:storage))
        self.shared_fc.load_state_dict(torch.load("{}/shared_fc.th".format(path), map_location=lambda storage, loc:storage))
