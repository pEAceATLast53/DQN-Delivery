# DQN-Delivery
The agent is trained by DQN to deliver food to customers.  
## Environment
<img src="https://user-images.githubusercontent.com/86182918/124696488-25bca680-df20-11eb-82c4-00452757d20c.gif" width="500" height="300">
   The delivery robot(red cross) navigates around the grid map to deliver food to the customers. The orders are initiated at random time steps, locations throughout the episode. When an order is initiated, the location of the customer is marked by a blue cross in the picture above. The total number of orders in one episode is fixed. In order to succeed in a delivery, the robot has to go to the same grid as the corresponding blue cross. The goal is to train the robot to succeed in all the deliveries in each episode. As the robot navigates, it must avoid static obstacles. In the picture above, the black grids are static obstacles, while the white grids are free space.

## DQN
The agent is trained by DQN.
### Action
The agent moves one grid in the N, W, S, E direction, or stays on the same grid in each time step.
If the agent's action results in the collision, then the agent stays on the same grid.
### Observation
1. 15 x 15 Egocentric Local Map - 1st channel : Grid map indicating the static obstacles around the agent, 2nd channel : Locations of the orders that are within the range of the local map.
2. Euclidean (--distance_type = 'euc') / Geodesic (--distance_type = 'geo') distances to the initiated orders.
3. Bearing angles to the initiated orders.
### Reward
If the agent's action will result in a collision with a static obstacle, the agent receives a penalty of -0.1. It also receives a -0.01 penalty per each remaining order that has already been initiated.
#### Sparse Reward
Every time the agent succeeds in a delivery, it receives +1.
#### Dense Reward
