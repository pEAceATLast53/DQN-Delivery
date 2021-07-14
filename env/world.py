from env.core import Agent, Landmark
import cv2
import copy
import os
import math
import numpy as np
import time
import heapq
import sys

class ScenarioDustGeneration(object):
    def __init__(self, world):
        self.num_landmarks = len(world.landmarks)
        self.num_found_landmarks = 0
        self.max_num_landmarks = self.num_landmarks
        self.alive = [1] * len(world.landmarks)
        self.times = [0] * len(world.landmarks)
        
    def reset(self, world): 
        generated_landmarks = []
        if world.scenario is not None: 
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = world.scenario['landmark pose'][i]
                landmark.found = False
                landmark.rewarded = False
                landmark.id = i
                landmark.generated = world.scenario['landmark time'][i] == False
                if landmark.generated:
                    generated_landmarks.append(landmark.state.p_pos)
            self.num_landmarks = len(world.landmarks)
            self.num_found_landmarks = 0
            self.alive = [1] * len(world.landmarks)
            self.times = [0] * len(world.landmarks)
            return np.array(generated_landmarks)[:,0], np.array(generated_landmarks)[:,1]
        else:
            print('No dust scenario input given')
            return None

    def update_state(self, world):
        poses = []
        for i, landmark in enumerate(world.landmarks):
            if world.scenario['landmark time'][i] == world.time_t:
                landmark.generated = True

            if (not landmark.found) and (landmark.generated):
                if (landmark.state.p_pos == world.agent.state.p_pos).all():
                    landmark.found = True
                    self.num_found_landmarks += 1

                if landmark.found:
                    self.alive[landmark.id] = 0
                else:
                    self.times[landmark.id] += 1
                    poses.append(landmark.state.p_pos)

        if len(poses) > 0:
            return np.array(poses)[:,0], np.array(poses)[:,1]
        else:
            return [], []

class World:
    def __init__(self, args):
        file_path = '/'.join(__file__.split('/')[:-1])
        self.map_img = cv2.imread(os.path.join(file_path,args.map_name), cv2.IMREAD_GRAYSCALE)
        self.size_ratio = 1
        self.map_img[np.where(self.map_img < 225)] = 0
        self.map_img[np.where(self.map_img != 0)] = 255

        self.map_color = np.stack([self.map_img, self.map_img, self.map_img],2)
        map_ = copy.deepcopy(self.map_img)
        self.map = map_
        self.map_H, self.map_W = self.map.shape

        self.map_padding = 0
        if self.map_padding > 0:
            x, y = np.where(self.map != 255)
            for xx, yy in zip(x, y):
                self.map[xx - self.map_padding:xx + self.map_padding, yy - self.map_padding:yy + self.map_padding] = 0
        self.free_area, self.free_area_idx = self.get_free_areas()
        self.adj_mat = self.get_adjacency_matrix()
        self.geo_dist_table = np.full((self.map_H, self.map_W, self.map_H, self.map_W), -1)

        self.lidar_range = args.lidar_range
        self.distance_type = args.distance_type
        self.reward_type = args.reward_type

        self.action_mapping = {0: (0, 0), 1: (-1, 0), 2: (0, 1), 3: (1, 0),
                               4: (0, -1)}
        self.world_state = np.zeros([self.map_H, self.map_W, 2])
        self.world_state[:,:,0] = self.map > 0

        # set world properties
        num_landmarks = args.num_landmarks
        print('num_landmark', num_landmarks)

        # add agents
        self.agent = Agent()
        self.agent.collide = True
        self.agent.color = np.array([80, 105, 127])

        # add landmarks
        self.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(self.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.8
            landmark.color = np.array([239, 111, 108])  # red
            setattr(landmark, 'generated', False)

        self.SUCCESS_REWARD = args.success_reward
        self.COLLISION_PENALTY = (-1) * args.collision_penalty
        self.TIME_PENALTY = (-1) * args.time_penalty
        self.max_episode_len = args.max_episode_len

        self.dust_model = ScenarioDustGeneration(self)
        #self.reset()

    def sample_navigable_position(self, candidate_area=None):
        if candidate_area is None:
            candidate_area = self.free_area
        h, w = self.sample_navigable_point(candidate_area)
        return np.array([h,w])

    def sample_navigable_point(self, possible_points=None):
        # return pixel point
        if possible_points is not None:
            num_candidate = len(possible_points)
            sampled_point_idx = np.random.randint(num_candidate)
            return possible_points[sampled_point_idx]
        else:
            h = np.random.rand() * self.map_H# - self.origin_pixel[0]
            w = np.random.rand() * self.map_W# - self.origin_pixel[1]
            return np.array([h, w])

    def get_euc_dist(self, src, target):
        return ((src[0]-target[0])**2 + (src[1]-target[1])**2)**0.5

    def create_scenario(self):
        candidate_area = copy.deepcopy(self.free_area)
        candidate_area_idx = copy.deepcopy(self.free_area_idx)
        
        candi_h, candi_w = self.sample_navigable_position(candidate_area)
        candi_idx = candidate_area_idx[candi_h, candi_w]
        agent_pose = [candi_h, candi_w]
        candidate_area.remove([candi_h, candi_w])
        np.where(candidate_area_idx==candi_idx, -1, candidate_area_idx)

        landmark_poses = []
        for landmark in self.landmarks:
            candi_h, candi_w = self.sample_navigable_position(candidate_area)
            candi_idx = candidate_area_idx[candi_h, candi_w]
            landmark_poses.append([candi_h, candi_w])
            candidate_area.remove([candi_h, candi_w])
            np.where(candidate_area_idx==candi_idx, -1, candidate_area_idx)

        init_landmarks = 1 # landmarks which are generated in the first step
        limit_time_ratio = 0.8
        limit_time = int(self.max_episode_len * limit_time_ratio)
        landmark_times = np.random.choice(range(limit_time), len(self.landmarks)-init_landmarks, replace=False)
        landmark_times = np.concatenate(([0]*init_landmarks, landmark_times), 0)
        np.random.shuffle(landmark_times) 

        scenario = {'agent': agent_pose, 'landmark pose': landmark_poses, 'landmark time': landmark_times}
        return scenario  

    def reset(self, scenario=None):
        self.time_t = 0
        self.world_state[:,:,1] = 0

        if scenario is None:
            scenario = self.create_scenario()
        self.scenario = {'agent': np.array(scenario['agent']),
                        'landmark pose': np.array(scenario['landmark pose']), 
                        'landmark time': scenario['landmark time']
                        }

        self.agent.state.p_pos = self.scenario['agent']
        self.agent.reward = 0.0        
        
        alive_landmarks_h, alive_landmarks_w = self.dust_model.reset(self) # world.scenario will be used
        if len(alive_landmarks_h) > 0:
            self.world_state[alive_landmarks_h, alive_landmarks_w, 1] = 1.0

        self.agent.action.prev_u = -1

        gps = []
        geo_dists = []
        for l in self.landmarks:
            if self.distance_type == 'euc':
                dist = self.get_euc_dist(l.state.p_pos, self.agent.state.p_pos)
            if self.distance_type == 'geo':
                if self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]] == -1:
                    self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]], _ = \
                        self.get_dist(self.agent.state.p_pos, l.state.p_pos)
                    self.geo_dist_table[l.state.p_pos[0], l.state.p_pos[1], self.agent.state.p_pos[0], self.agent.state.p_pos[1]] = \
                        self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]]
                dist = self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]]
            if self.reward_type == 'dense':
                if self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]] == -1:
                    self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]], _ = self.get_dist(self.agent.state.p_pos, l.state.p_pos)
                    self.geo_dist_table[l.state.p_pos[0], l.state.p_pos[1], self.agent.state.p_pos[0], self.agent.state.p_pos[1]] = \
                        self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]]
                if l.generated and not l.found:
                    geo_dists.append(self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]])
            direction = math.atan2(*(l.state.p_pos - self.agent.state.p_pos))
            gps.append([dist, direction])
        self.agent.gps = np.array(gps) 
        if len(geo_dists) > 0:
            self.prev_min_dist = min(geo_dists)
        self.collision_count = 0           
  
    def step(self):
        self.world_state[:,:,1] = 0.0
        self.move_agent()
        alive_landmarks_h, alive_landmarks_w = self.dust_model.update_state(self)
        if len(alive_landmarks_h) > 0:
            self.world_state[alive_landmarks_h, alive_landmarks_w, 1] = 1.0
        gps = []
        geo_dists = []
        for l in self.landmarks:
            if self.distance_type == 'euc':
                dist = self.get_euc_dist(l.state.p_pos, self.agent.state.p_pos)
            if self.distance_type == 'geo':
                if self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]] == -1:
                    self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]], _ = self.get_dist(self.agent.state.p_pos, l.state.p_pos)
                    self.geo_dist_table[l.state.p_pos[0], l.state.p_pos[1], self.agent.state.p_pos[0], self.agent.state.p_pos[1]] = \
                        self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]]
                dist = self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]]
            if self.reward_type == 'dense':
                if self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]] == -1:
                    self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]], _ = self.get_dist(self.agent.state.p_pos, l.state.p_pos)
                    self.geo_dist_table[l.state.p_pos[0], l.state.p_pos[1], self.agent.state.p_pos[0], self.agent.state.p_pos[1]] = \
                        self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]]
                if l.generated and not l.found:
                    geo_dists.append(self.geo_dist_table[self.agent.state.p_pos[0], self.agent.state.p_pos[1], l.state.p_pos[0], l.state.p_pos[1]])
            direction = math.atan2(*(l.state.p_pos - self.agent.state.p_pos))
            gps.append([dist,direction])
        self.agent.gps = np.array(gps)
        if len(geo_dists) > 0:
            self.curr_min_dist = min(geo_dists)
        self.time_t += 1

    def reward(self):
        rew = 0
        if [self.agent.state.next_pos[0], self.agent.state.next_pos[1]] not in self.free_area:
            rew += self.COLLISION_PENALTY
        for l in self.landmarks:
            if (l.generated and l.found) and not l.rewarded:
                rew += self.SUCCESS_REWARD
                l.rewarded = True
            if l.generated and not l.found:
                rew += self.TIME_PENALTY
        if self.reward_type == 'dense':
            if abs(self.prev_min_dist - self.curr_min_dist) <= 1:
                rew += 0.1 * (self.prev_min_dist - self.curr_min_dist)
            self.prev_min_dist = self.curr_min_dist
        return rew

    def done(self):
        if self.time_t >= self.max_episode_len or self.dust_model.num_found_landmarks == self.dust_model.num_landmarks: 
            return True
        else: 
            return False

    def observation(self):
        map_obs = self.get_patch()
        self.agent.state.obs = {'map': map_obs, 'dists': self.agent.gps, 'prev_action': self.agent.action.prev_u}
        return self.agent.state.obs

    #Returns geodesic distance and shortest path between src and target by Dijkstra.
    def get_dist(self, src, target):
        if (src == target).all() : return 0.0, []
        src_idx = self.free_area_idx[src[0], src[1]]
        target_idx = self.free_area_idx[target[0], target[1]]
        parent_list = np.full(len(self.free_area), -1)
        distances = {free_node_idx: float('infinity') for free_node_idx in range(len(self.free_area))}
        distances[src_idx] = 0

        pq = [(0, src_idx)]
        while len(pq) > 0:
            curr_dist, curr_node_idx = heapq.heappop(pq)
            if curr_dist > distances[curr_node_idx]: continue
            if curr_node_idx == target_idx: break
            neighbor_ids = np.where(self.adj_mat[curr_node_idx, :] > 0)
            for neighbor_idx in neighbor_ids[0]:
                distance = curr_dist + self.adj_mat[curr_node_idx, neighbor_idx]
                if distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = distance
                    heapq.heappush(pq, (distance, neighbor_idx))
                    parent_list[neighbor_idx] = curr_node_idx
        assert parent_list[target_idx] > -1, "No path from source to target"

        shortest_path = []
        pt_idx = target_idx
        while pt_idx != src_idx:
            shortest_path.append(self.free_area[pt_idx])
            pt_idx = parent_list[pt_idx]
        shortest_path.append(self.free_area[src_idx])
        shortest_path = shortest_path[::-1]
        return distances[target_idx], shortest_path

    def state_transition(self):
        return [self.agent.state.p_pos[0] + self.action_mapping[self.agent.action.u][0], \
            self.agent.state.p_pos[1] + self.action_mapping[self.agent.action.u][1]]

    def move_agent(self):
        if [self.agent.state.next_pos[0], self.agent.state.next_pos[1]] in self.free_area:
            self.agent.state.p_pos = [self.agent.state.next_pos[0], self.agent.state.next_pos[1]]
        else:
            self.collision_count += 1
                                             
    def get_patch(self):
        h,w = self.agent.state.p_pos

        agent_patch = np.zeros([self.lidar_range*2+1,self.lidar_range*2+1,2])
        min_h = max(0, h-self.lidar_range)
        max_h = min(h+self.lidar_range, self.map_H-1)
        min_w = max(0, w-self.lidar_range)
        max_w = min(w+self.lidar_range, self.map_W-1)
        patch = copy.deepcopy(self.world_state[min_h:max_h+1, min_w:max_w+1])
        patch_size = agent_patch.shape[0]//2
        start_h = max(patch_size-h, 0)
        start_w = max(patch_size-w, 0)
        agent_patch[start_h:start_h + patch.shape[0],start_w:start_w + patch.shape[1]] = patch

        return agent_patch

    def get_free_areas(self):
        h, w = np.where(self.map == 255)
        free_area_idx = np.full(self.map.shape, -1)
        free_areas = []
        for hh, ww in zip(h, w):
            free_areas.append([hh,ww])
            free_area_idx[hh,ww] = len(free_areas) - 1
        return free_areas, free_area_idx

    def get_adjacency_matrix(self):
        num_free = len(self.free_area)
        adj_mat = np.zeros((num_free, num_free))
        for h in range(self.map_H):
            for w in range(self.map_W):
                curr_idx = self.free_area_idx[h,w]
                if curr_idx > -1:
                    #adj_mat[curr_idx, curr_idx] = 1
                    if w+1 < self.map_W:
                        adj_idx = self.free_area_idx[h, w+1]
                        if adj_idx > -1:
                            adj_mat[curr_idx, adj_idx] = 1
                            adj_mat[adj_idx, curr_idx] = 1
                    if h+1 < self.map_H:
                        adj_idx = self.free_area_idx[h+1, w]
                        if adj_idx > -1:
                            adj_mat[curr_idx, adj_idx] = 1
                            adj_mat[adj_idx, curr_idx] = 1
        return adj_mat

    def render(self):
        frame = copy.deepcopy(self.map_color)
        cv2.circle(frame, (self.agent.state.p_pos[1], self.agent.state.p_pos[0]), 2, (0, 0, 255), -1)
        for l in self.landmarks:
            if l.generated and not l.found:
                cv2.circle(frame, (l.state.p_pos[1], l.state.p_pos[0]), 1, (255, 0, 0), -1)
        return frame


