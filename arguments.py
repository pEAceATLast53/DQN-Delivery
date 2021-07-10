import argparse

parser = argparse.ArgumentParser("DQN Delivery")

parser.add_argument('--map_name', default='dist_test.png')

parser.add_argument('--lidar_range', default=7)
parser.add_argument('--num_landmarks', default=3)

parser.add_argument('--success_reward', default=1)
parser.add_argument('--collision_penalty', default=0.1)
parser.add_argument('--time_penalty', default=0.01)

parser.add_argument('--max_episode_len', default=100)
parser.add_argument("--num-episodes", type=int, default=500000, help="number of episodes")
parser.add_argument("--update_start", default=10000)
parser.add_argument("--save_interval", default=10000)

parser.add_argument('--device', default='cuda')

parser.add_argument('--model_name', default='3_100', type=str)

# RL hyperparameters
parser.add_argument('--target_update_interval', default=200, type=int)  
parser.add_argument('--update_interval', default=100) 
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--buffer_size', default=1000000, type=int)
parser.add_argument('--init_epsilon', default=1.0, type=float)
parser.add_argument('--final_epsilon', default=0.05, type=float)
parser.add_argument('--epsilon_anneal_time', default=50000, type=int)   
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--optim_alpha', default=0.99, type=float)           
parser.add_argument('--optim_eps', default=0.00001, type=float)          

args = parser.parse_args()