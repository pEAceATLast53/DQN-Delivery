from env.environment import Env
from env.world import World
from dqn.replay_buffer import ReplayBuffer
from dqn.q_learner import QLearner
from arguments import args

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2, os, torch, random, joblib
from tensorboardX import SummaryWriter
import numpy as np

save_dir = './results'
model_load_dir = save_dir + '/models/' + args.model_name
render_save_dir = save_dir + '/render/' + args.model_name

assert os.path.isdir(model_load_dir), "Model path does not exist"
if not os.path.isdir(render_save_dir):
    os.mkdir(render_save_dir)

world = World(args)
env = Env(world, reset_callback = world.reset, reward_callback = world.reward, observation_callback = world.observation, \
    info_callback = world.render, done_callback = world.done)

args.mode = 'eval'
trainer = QLearner(args)
trainer.load_models(model_load_dir)
trainer.cuda()

for episode in range(args.num_episodes):
    episode_return = 0
    obs, info = env.reset()
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
    frames = [info]

    while True:
        a = trainer.select_action(torch.Tensor(obs_coord).cuda().float().unsqueeze(0), \
            torch.Tensor(obs_map).cuda().float().unsqueeze(0), torch.Tensor(obs_prev_action).cuda().float().unsqueeze(0), \
            torch.Tensor(obs_pos).cuda().float().unsqueeze(0))

        obs_next, r, d, info = env.step(a)
        frames.append(info)

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

        episode_return += r

        if d:
            break

        obs_map = obs_map_next
        obs_coord = obs_coord_next
        obs_prev_action = obs_prev_action_next
        obs_pos = obs_pos_next

    num_found_landmarks = 0
    for l in world.landmarks:
        if l.generated and l.found:
            num_found_landmarks += 1
    
    print("Episode : ", episode+1, "Return : ", episode_return, "# of successful orders : ", num_found_landmarks, "# of collisions : ", world.collision_count)

    out = cv2.VideoWriter(os.path.join(render_save_dir, str(episode) + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15, (frames[0].shape[1], frames[0].shape[0]))
    for aa in frames:
        out.write(aa)
    out.release()