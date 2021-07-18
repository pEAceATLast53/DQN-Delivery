import argparse

parser = argparse.ArgumentParser("DQN Delivery")

parser.add_argument('--map_name', default='dist_test.png')

parser.add_argument('--lidar_range', default=7)
parser.add_argument('--num_landmarks', default=8)
parser.add_argument('--distance_type', default='both', choices = ['euc', 'geo', 'both'])
parser.add_argument('--reward_type', default='dense', choices = ['sparse', 'dense'])

parser.add_argument('--success_reward', default=1)
parser.add_argument('--collision_penalty', default=0.1)
parser.add_argument('--time_penalty', default=0.01)

parser.add_argument('--max_episode_len', default=300)

args = parser.parse_args()

from env.environment import Env
from env.world import World
import random
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

world = World(args)
env = Env(world, reset_callback = world.reset, reward_callback = world.reward, observation_callback = world.observation, \
    info_callback = world.render, done_callback = world.done)
frames = []
_, info = env.reset()
frames.append(info)

while True:
    _, _, done, info = env.step(random.randint(0, 4))
    frames.append(info)
    if done:
        break

out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, world.map_img.shape[::-1])
for aa in frames:
    out.write(aa)
out.release()
