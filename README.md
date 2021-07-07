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
The agent's observations are the relative coordinates of the initiated orders that the agent has not yet visited, and a 15x15 egocentric local map. The local map is comprised of 2 channels. The first channel is the grid map indicating the static obstacles arount the agent. The second channel shows the locations of the orders that are within the range of the local map.
### Reward
Every time the agent succeeds in a delivery, it receives +1. If the agent's action will result in a collision with a static obstacle, the agent receives a penalty of -1. It also receives a -0.01 penalty per each remaining order that has already been initiated.
