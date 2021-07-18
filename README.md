# DQN-Delivery
The robot is trained by DQN to deliver food to customers.  
## Environment
<img src="https://user-images.githubusercontent.com/86182918/124696488-25bca680-df20-11eb-82c4-00452757d20c.gif" width="500" height="300">
   The delivery robot(red cross) navigates around the grid map to deliver food to the customers. The orders are initiated at random time steps, on random grids throughout the episode. When an order is initiated, its grid location is marked by a blue cross in the picture above. The total number of orders in one episode is fixed. In order to succeed in a delivery, the robot has to go to the same grid as the corresponding blue cross. The goal is to train the robot to succeed in all the deliveries in each episode. As the robot navigates, it must avoid static obstacles. In the picture above, the black grids are static obstacles, while the white grids are free space.
   - A grid in which an order takes place is called 'landmark' in the code.

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
3. Bearing angles to the orders.
4. Robot's current coordinate.
5. Robot's previous action.
### Reward
If the robot's action will result in a collision with a static obstacle, it receives a penalty of -0.1. It also receives a -0.01 penalty per each remaining order that has been initiated.
#### Sparse Reward
Every time the robot succeeds in a delivery, it receives +1.
#### Dense Reward
In addition to the sparse reward, the agent receives a progress reward, which is 0.1 * (previous_geodesic_distance_to_the_closest_order - current_geodesic_distance_to_the_closest_order).
## Current Progress
- Training the agent only with the sparse reward failed.
- Including the dense reward enabled training. Below are some of the successful episodes.
   '''
   --num_landmarks 2  #Total number of orders in one episode
   --max_episode_len 100  #Episode length is 100 time steps
   --reward_type dense
   '''
   <img src="https://user-images.githubusercontent.com/86182918/126068284-b622657c-3099-41c0-a271-c46f4d83b894.gif" width="500" height="300">
   <img src="https://user-images.githubusercontent.com/86182918/126068335-0aa9fee9-18a8-45e5-b81b-b1a3415aad9c.gif" width="500" height="300">
  However, the success rate wasn't too high. The agent was not able to succeed in orders far away and block by walls.
