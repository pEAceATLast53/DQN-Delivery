import collections, random, torch

class ReplayBuffer():
    def __init__(self, args):
        self.buffer_a = collections.deque(maxlen = args.buffer_size)
        self.buffer_r = collections.deque(maxlen = args.buffer_size)
        self.buffer_d = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_map = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_coord = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_map_next = collections.deque(maxlen = args.buffer_size)
        self.buffer_obs_coord_next = collections.deque(maxlen = args.buffer_size)
        
        self.batch_size = args.batch_size
        self.device = args.device
        self.length = 0

    def store(self, a, r, d, obs_map, obs_coord, obs_map_next, obs_coord_next):
        self.buffer_a.append(a)
        self.buffer_r.append(r)
        self.buffer_d.append(d)
        self.buffer_obs_map.append(obs_map)
        self.buffer_obs_coord.append(obs_coord)
        self.buffer_obs_map_next.append(obs_map_next)
        self.buffer_obs_coord_next.append(obs_coord_next)
        self.length = len(self.buffer_a)

    def sample(self):
        batch_idx = random.sample(range(self.length), self.batch_size)

        a_list = [self.buffer_a[idx] for idx in batch_idx]
        r_list = [self.buffer_r[idx] for idx in batch_idx]
        d_list = [self.buffer_d[idx] for idx in batch_idx]
        obs_map_list = [self.buffer_obs_map[idx] for idx in batch_idx]
        obs_coord_list = [self.buffer_obs_coord[idx] for idx in batch_idx]
        obs_map_next_list = [self.buffer_obs_map_next[idx] for idx in batch_idx]
        obs_coord_next_list = [self.buffer_obs_coord_next[idx] for idx in batch_idx]

        return torch.stack(a_list).to(self.device).long(), torch.stack(r_list).to(self.device).float(), \
            torch.stack(d_list).to(self.device).long(), torch.stack(obs_map_list).to(self.device).float(), \
            torch.stack(obs_coord_list).to(self.device).float(), torch.stack(obs_map_next_list).to(self.device).float(), \
            torch.stack(obs_coord_next_list).to(self.device).float()