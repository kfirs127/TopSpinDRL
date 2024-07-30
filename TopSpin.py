import math

import gymnasium
from gymnasium import spaces
import numpy as np
import random
from BWAS import BWAS
import tensorflow as tf

class TopSpin:
    def __init__(self, n, k, state=None):
        self.n = n
        self.k = k
        self.state = state if state is not None else np.arange(1, n + 1)

    def rotate_clockwise(self):
        self.state = np.roll(self.state, 1)

    def rotate_counterclockwise(self):
        self.state = np.roll(self.state, -1)

    def flip(self):
        self.state = np.concatenate([self.state[:self.k][::-1], self.state[self.k:]])

    def get_neighbors(self):
        return [np.roll(self.state, 1), np.roll(self.state, -1),
                np.concatenate([self.state[:self.k][::-1], self.state[self.k:]])]

    def get_neighbors_topspin(self):
        return [(TopSpin(self.n, self.k, list(np.roll(self.state.copy(), 1))), 1),
                (TopSpin(self.n, self.k, list(np.roll(self.state.copy(), -1))), 1),
                (TopSpin(self.n, self.k, list(np.concatenate([self.state.copy()[:self.k][::-1], self.state.copy()[self.k:]]))), 1)]

    def is_solved(self, state=None):
        state = state if state is not None else self.state
        return np.all(state == np.arange(1, self.n + 1))

    def scramble(self, distance):
        for _ in range(distance):
            action_index = random.randint(0, 2)
            neighbors_states = [rs for rs in self.get_neighbors()]
            self.state = neighbors_states[action_index]

        # to make sure not to get the goal state.
        if self.is_solved():
            action_index = random.randint(0, 2)
            neighbors_states = [rs for rs in self.get_neighbors()]
            self.state = neighbors_states[action_index]

    def get_state(self):
        return self.state.copy()

    def __repr__(self):
        return f"TopSpin({list(self.state)})"

    def __eq__(self, other):
        if isinstance(other, TopSpin):
            return list(self.state) == list(other.state)
        return False

    def __hash__(self):
        return hash(tuple(self.state))



class TopSpinEnv(gymnasium.Env):
    def __init__(self, n, k, distance, rewardType, state=None, rewardNet=None, rewardOptimizer=None):
        super(TopSpinEnv, self).__init__()
        self.topspin = TopSpin(n, k, state)
        self.distance = distance
        self.rewardType = rewardType
        self.action_space = spaces.Discrete(3)  # 0: rotate_clockwise, 1: rotate_counterclockwise, 2: flip
        self.observation_space = spaces.Box(low=1, high=n, shape=(n,), dtype=np.int32)
        self.rewardNet = rewardNet
        self.rewardOptimizer = rewardOptimizer

    def reset(self, *, seed=None, options=None):
        self.topspin = TopSpin(self.topspin.n, self.topspin.k)
        self.topspin.scramble(self.distance)
        return self.topspin.get_state(), {}

    def step(self, action):
        reward = self.get_reward()
        if action == 0:
            self.topspin.rotate_clockwise()
        elif action == 1:
            self.topspin.rotate_counterclockwise()
        elif action == 2:
            self.topspin.flip()

        done = 1 if self.topspin.is_solved() else 0
        return self.topspin.get_state(), reward, done, {}, {}

    def get_reward(self):
        if self.rewardType == 0:
            return -1
        elif self.rewardType == 1:
            return self.get_euclidean_distance(self.topspin.state, np.arange(1, self.topspin.n + 1))
        elif self.rewardType == 2:
            topspin = TopSpin(self.topspin.n, self.topspin.k, self.topspin.state.copy())
            path, _ = BWAS(topspin, 5, 10)
            return -1000 if path is None else -len(path)
        elif self.rewardType == 3:
            state_tensor = tf.convert_to_tensor(self.topspin.state, dtype=tf.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)  # Add batch dimension
            return self.rewardNet(state_tensor)[0][0]

    def get_euclidean_distance(self, state, goal):
        distance = 0
        for i in range(len(state)):
            distance += (state[i] - goal[i]) ** 2
        return -math.sqrt(distance)

    def render(self, mode='human'):
        print(self.topspin.get_state())

    def get_k(self):
        return self.topspin.k

    def get_n(self):
        return self.topspin.n
