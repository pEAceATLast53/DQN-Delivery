# DQN-Delivery
The agent is trained by DQN to deliver food to customers.  
## Environment
<img src="https://user-images.githubusercontent.com/86182918/124696488-25bca680-df20-11eb-82c4-00452757d20c.gif" width="400" height="300">
   The delivery robot(red cross) navigates around the grid map to deliver food to the customers. The orders are initiated at random time steps, locations throughout the episode. When an order is initiated, the location of the customer is marked by a blue cross in the picture above. The total number of orders in one episode is fixed. In order to succeed in a delivery, the robot has to go to the same grid as the corresponding blue cross. The goal is to train the robot to succeed in all the deliveries in each episode. As the robot navigates, it must avoid static obstacles. In the picture above, the black grids are static obstacles, while the white grids are free space.
