# DQN-Delivery
The robot is trained by DQN to deliver food to customers.  
To train, run
```
python main.py
```
To evalutate a trained model and save videos, run
```
python eval.py
```
The list of arguments can be found at arguments.py.

To simply test the environment, run
```
python test_env.py
```
## Environment
<img src="https://user-images.githubusercontent.com/86182918/124696488-25bca680-df20-11eb-82c4-00452757d20c.gif" width="500" height="300">
   The delivery robot(red cross) navigates around the grid map to deliver food to the customers. The orders are initiated at random time steps, on random grids throughout the episode. When an order is initiated, its grid location is marked by a blue cross in the picture above. The total number of orders in one episode is fixed. In order to succeed in a delivery, the robot has to go to the same grid as the corresponding blue cross. The goal is to train the robot to succeed in all the deliveries in each episode. As the robot navigates, it must avoid static obstacles. In the picture above, the black grids are static obstacles, while the white grids are free space.
   
   - A grid in which an active order takes place is called 'landmark' in the code.

## DQN
The robot is trained by DQN.
### Action
The robot moves one grid in the N, W, S, E direction, or stays on the same grid in each time step.
If the robot's action results in a collision, then the robot stays on the same grid.
### Observation
1. 15 x 15 Egocentric Local Map
   - 1st channel : Grid map indicating the static obstacles around the robot
   - 2nd channel : One-hot encodings indicating the locations of the orders that are within the range of the local map.
2. Distance to the orders.
   - --distance_type euc : Euclidean distance
   - --distance_type geo : Geodesic distance
   - --distance_type both : Both Euclidean and geodesic distances
3. Bearing angles to the orders.
4. Robot's current coordinate.
5. Robot's previous action.
### Reward
If the robot's action will result in a collision with a static obstacle, it receives a penalty of -0.1. It also receives a -0.01 penalty per each active order.
#### Sparse Reward
Every time the robot succeeds in a delivery, it receives +1.
#### Dense Reward
In addition to the sparse reward, the agent receives a progress reward, which is 0.1 * (previous_geodesic_distance_to_the_closest_order - current_geodesic_distance_to_the_closest_order).
## Current Progress
1. Training the agent only with the sparse reward failed. (--num_landmarks 2, --max_episode_len 100, --reward_type sparse, --distance_type euc, --lidar_range 7)
2. Including the dense reward enabled training. Below are some of the successful episodes. (--num_landmarks 2, --max_episode_len 100, --reward_type dense, --distance_type euc, --lidar_range 15)
   <img src="https://user-images.githubusercontent.com/86182918/126624153-4bedebe1-b127-428f-9a8e-0781c5941538.gif" width="500" height="300">
   <img src="https://user-images.githubusercontent.com/86182918/126624234-ce5ebb38-3268-4a4a-9177-defdeec6f428.gif" width="500" height="300">
