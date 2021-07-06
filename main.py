from arguments import args
from env.environment import Env
from env.world import World
from dqn.replay_buffer import ReplayBuffer
from dqn.q_learner import QLearner

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2, os, torch, random, joblib
from tensorboardX import SummaryWriter
import numpy as np

if torch.cuda.is_available():
    if args.device == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_dir = './results'
model_save_dir = save_dir + '/models/' + args.model_name
log_dir = save_dir + '/tb_logs/' + args.model_name

if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

world = World(args)
env = Env(world, reset_callback = world.reset, reward_callback = world.reward, observation_callback = world.observation, \
    info_callback = world.render, done_callback = world.done)

trainer = QLearner(args, writer)
trainer.cuda()

epsilon = args.init_epsilon

t_total = 0

for episode in range(args.num_episodes):
    episode_return = 0
    obs, _ = env.reset()
    obs_map = obs['map']
    obs_coord_dict = obs['dists']
    obs_coord = np.full((args.num_landmarks, 3), -1)
    for idx, l in enumerate(world.landmarks):
        if not l.generated or l.found:
            continue
        obs_coord[idx, 0] = obs_coord_dict[idx, 0]
        obs_coord[idx, 1] = np.cos(obs_coord_dict[idx, 1])
        obs_coord[idx, 2] = np.sin(obs_coord_dict[idx, 1])

    while True:
        t_total += 1
        if random.random() < epsilon:
            a = random.randint(0, 4)
        else:
            a = trainer.select_action(torch.Tensor(obs_coord).to(args.device).float().unsqueeze(0), \
                torch.Tensor(obs_map).to(args.device).float().unsqueeze(0))

        obs_next, r, d, _ = env.step(a)

        obs_map_next = obs['map']
        obs_coord_next_dict = obs['dists']
        obs_coord_next = np.full((args.num_landmarks, 3), -1)
        for idx, l in enumerate(world.landmarks):
            if not l.generated or l.found:
                continue
            obs_coord_next[idx, 0] = obs_coord_next_dict[idx, 0]
            obs_coord_next[idx, 1] = np.cos(obs_coord_next_dict[idx, 1])
            obs_coord_next[idx, 2] = np.sin(obs_coord_next_dict[idx, 1])

        trainer.replay_buffer.store(torch.Tensor([a]), torch.Tensor([r]), torch.Tensor([d]), torch.Tensor(obs_map), torch.Tensor(obs_coord), torch.Tensor(obs_map_next), torch.Tensor(obs_coord_next))

        if trainer.replay_buffer.length >= args.update_start and t_total % args.update_interval == 0:
            trainer.train()

        episode_return += r

        if epsilon > args.final_epsilon:
            epsilon -= (args.init_epsilon - args.final_epsilon) / args.epsilon_anneal_time

        if d:
            break

    if episode % args.target_update_interval == 0:
        trainer.update_targets()

    if episode % args.save_interval == 0:
        trainer.save_models(model_save_dir)

    trainer.writer.add_scalar('Episode Return', episode_return, episode+1)
    print("Episode : ", episode+1, "Return : ", episode_return, "Duration : ", world.time_t)