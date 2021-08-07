from arguments import args
from env.environment import Env
from env.world import World
from dqn.replay_buffer import ReplayBuffer
from dqn.q_learner import QLearner

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2, os, torch, random, joblib
import numpy as np

save_dir = './results'
model_save_dir = save_dir + '/models/' + args.model_name
log_dir = save_dir + '/tb_logs/' + args.model_name
args.log_dir = log_dir
args.mode = 'train'

if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

world = World(args)
env = Env(world, reset_callback = world.reset, reward_callback = world.reward, observation_callback = world.observation, \
    info_callback = world.render, done_callback = world.done)

trainer = QLearner(args)
trainer.cuda()

epsilon = args.init_epsilon

t_total = 0

for episode in range(args.num_episodes):
    episode_return = 0
    obs, _ = env.reset()
    obs_map = obs['map']
    obs_coord_raw = obs['dists']
    if args.distance_type == 'both':
        obs_coord = np.full((args.num_landmarks, 4), -1)
    else:
        obs_coord = np.full((args.num_landmarks, 3), -1)
    for idx, l in enumerate(world.landmarks):
        if not l.generated or l.found:
            continue
        lm_info = []
        lm_info.append(obs_coord_raw[idx, 0] * 0.01)
        if args.distance_type == 'both':
            lm_info.append(obs_coord_raw[idx, 1] * 0.01)
        lm_info.append(np.cos(obs_coord_raw[idx, -1]))
        lm_info.append(np.sin(obs_coord_raw[idx, -1]))
        obs_coord[idx, :] = np.array(lm_info)
    obs_prev_action = [obs['prev_action']]
    obs_pos = [world.agent.state.p_pos[0] * 0.01, world.agent.state.p_pos[1] * 0.01]

    while True:
        t_total += 1

        if random.random() < epsilon:
            a = random.randint(0, 4)
        else:
            a = trainer.select_action(torch.Tensor(obs_coord).cuda().float().unsqueeze(0), \
                torch.Tensor(obs_map).cuda().float().unsqueeze(0), torch.Tensor(obs_prev_action).cuda().float().unsqueeze(0), \
                torch.Tensor(obs_pos).cuda().float().unsqueeze(0))

        obs_next, r, d, _ = env.step(a)

        obs_map_next = obs_next['map']
        obs_coord_next_raw = obs_next['dists']
        if args.distance_type == 'both':
            obs_coord_next = np.full((args.num_landmarks, 4), -1)
        else:
            obs_coord_next = np.full((args.num_landmarks, 3), -1)
        for idx, l in enumerate(world.landmarks):
            if not l.generated or l.found:
                continue
            lm_info = []
            lm_info.append(obs_coord_next_raw[idx, 0] * 0.01)
            if args.distance_type == 'both':
                lm_info.append(obs_coord_next_raw[idx, 1] * 0.01)
            lm_info.append(np.cos(obs_coord_next_raw[idx, -1]))
            lm_info.append(np.sin(obs_coord_next_raw[idx, -1]))
            obs_coord_next[idx, :] = np.array(lm_info)
        obs_prev_action_next = [obs_next['prev_action']]
        obs_pos_next = [world.agent.state.p_pos[0] * 0.01, world.agent.state.p_pos[1] * 0.01]

        trainer.replay_buffer.store(torch.Tensor([a]).cpu(), torch.Tensor([r]).cpu(), torch.Tensor([d]).cpu(), \
            torch.Tensor(obs_map).cpu(), torch.Tensor(obs_coord).cpu(), torch.Tensor(obs_prev_action).cpu(), torch.Tensor(obs_pos).cpu(), \
            torch.Tensor(obs_map_next).cpu(), torch.Tensor(obs_coord_next).cpu(), torch.Tensor(obs_prev_action_next).cpu(), torch.Tensor(obs_pos_next).cpu())

        if trainer.replay_buffer.length >= args.update_start and t_total % args.update_interval == 0:
            trainer.train()

        episode_return += r

        if epsilon > args.final_epsilon:
            epsilon -= (args.init_epsilon - args.final_epsilon) / args.epsilon_anneal_time

        if d:
            break

        obs_map = obs_map_next
        obs_coord = obs_coord_next
        obs_prev_action = obs_prev_action_next
        obs_pos = obs_pos_next

    if episode % args.target_update_interval == 0:
        trainer.update_targets()

    if episode % args.save_interval == 0:
        trainer.save_models(model_save_dir)

    num_found_landmarks = 0
    for l in world.landmarks:
        if l.generated and l.found:
            num_found_landmarks += 1
            
    trainer.writer.add_scalar('Episode Return', episode_return, episode+1)
    trainer.writer.add_scalar('Number of Successful Orders', num_found_landmarks, episode+1)
    trainer.writer.add_scalar('Number of Collisions', world.collision_count, episode+1)
    print("Episode : ", episode+1, "Return : ", episode_return, "# of successful orders : ", num_found_landmarks, "# of collisions : ", world.collision_count)