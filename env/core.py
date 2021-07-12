import numpy as np

class EntityState(object):
    def __init__(self):
        self.p_pos = None

class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        self.next_pos = None

# action of the agent
class Action(object):
    def __init__(self):
        self.u = None
        self.prev_u = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.50
        # state
        self.state = EntityState()

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.found = False
        self.rewarded = False

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        self.reward = 0.0
