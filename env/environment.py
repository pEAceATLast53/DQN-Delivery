import numpy as np
import time
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import copy
import imp

class Env():
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None):
        self.world = world
        self.agent = self.world.agent
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.time = 0

    def step(self, action_n):
        self._set_action(action_n)
        self.world.step()
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return obs, reward, done, info

    def reset(self, scenario=None):
        # reset world
        if scenario is not None: 
            self.reset_callback(scenario=scenario)
        else:
            self.reset_callback()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # set env action for a particular agent
    def _set_action(self, action, time=None):
        self.agent.action.u = action
        self.agent.state.next_pos = self.world.state_transition()

    # get observation for a particular agent
    def _get_obs(self):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback()

    def _get_done(self):
        if self.done_callback is None:
            return False
        return self.done_callback()

    # get reward for a particular agent
    def _get_reward(self):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback()

    # info : frame for render
    def _get_info(self):
        if self.info_callback is None:
            return None
        return self.info_callback()