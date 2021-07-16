import collections, random, torch

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer_a = collections.deque(maxlen = buffer_size)
        self.buffer_r = collections.deque(maxlen = buffer_size)
        self.buffer_d = collections.deque(maxlen = buffer_size)
        self.buffer_obs_map = collections.deque(maxlen = buffer_size)
        self.buffer_obs_coord = collections.deque(maxlen = buffer_size)
        self.buffer_obs_prev_action = collections.deque(maxlen= buffer_size)
        self.buffer_obs_pos = collections.deque(maxlen = buffer_size)
        self.buffer_obs_map_next = collections.deque(maxlen = buffer_size)
        self.buffer_obs_coord_next = collections.deque(maxlen = buffer_size)
        self.buffer_obs_prev_action_next = collections.deque(maxlen= buffer_size)
        self.buffer_obs_pos_next = collections.deque(maxlen= buffer_size)
        
        self.batch_size = batch_size
        self.length = 0

    def store(self, a, r, d, obs_map, obs_coord, obs_prev_action, obs_pos, obs_map_next, obs_coord_next, obs_prev_action_next, obs_pos_next):
        self.buffer_a.append(a)
        self.buffer_r.append(r)
        self.buffer_d.append(d)
        self.buffer_obs_map.append(obs_map)
        self.buffer_obs_coord.append(obs_coord)
        self.buffer_obs_prev_action.append(obs_prev_action)
        self.buffer_obs_pos.append(obs_pos)
        self.buffer_obs_map_next.append(obs_map_next)
        self.buffer_obs_coord_next.append(obs_coord_next)
        self.buffer_obs_prev_action_next.append(obs_prev_action_next)
        self.buffer_obs_pos_next.append(obs_pos_next)
        self.length = len(self.buffer_a)

    def sample(self):
        batch_idx = random.sample(range(self.length), self.batch_size)

        a_list = [self.buffer_a[idx] for idx in batch_idx]
        r_list = [self.buffer_r[idx] for idx in batch_idx]
        d_list = [self.buffer_d[idx] for idx in batch_idx]
        obs_map_list = [self.buffer_obs_map[idx] for idx in batch_idx]
        obs_coord_list = [self.buffer_obs_coord[idx] for idx in batch_idx]
        obs_prev_action_list = [self.buffer_obs_prev_action[idx] for idx in batch_idx]
        obs_pos_list = [self.buffer_obs_pos[idx] for idx in batch_idx]
        obs_map_next_list = [self.buffer_obs_map_next[idx] for idx in batch_idx]
        obs_coord_next_list = [self.buffer_obs_coord_next[idx] for idx in batch_idx]
        obs_prev_action_next_list = [self.buffer_obs_prev_action_next[idx] for idx in batch_idx]
        obs_pos_next_list = [self.buffer_obs_pos_next[idx] for idx in batch_idx]

        return torch.stack(a_list).cuda().long(), torch.stack(r_list).cuda().float(), \
            torch.stack(d_list).cuda().long(), torch.stack(obs_map_list).cuda().float(), \
            torch.stack(obs_coord_list).cuda().float(), torch.stack(obs_prev_action_list).cuda().float(), \
            torch.stack(obs_pos_list).cuda().float(), \
            torch.stack(obs_map_next_list).cuda().float(), torch.stack(obs_coord_next_list).cuda().float(), \
            torch.stack(obs_prev_action_next_list).cuda().float(), torch.stack(obs_pos_next_list).cuda().float()