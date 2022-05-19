"""

	Replay Buffer for Deep Reinforcement Learning

"""

from collections import deque
import random
import numpy as np


class ReplayBuffer():
    def __init__(self, size_buffer, random_seed=8):
        self.__size_bf = size_buffer
        self.__length = 0
        self.__buffer = deque()
        random.seed(random_seed)
        np.random.seed(random_seed)


    @property
    def buffer(self):
        return self.__buffer


    def add(self, exp):
        if self.__length < self.__size_bf:
            self.__buffer.append(exp)
            self.__length += 1
        else:
            self.__buffer.popleft()
            self.__buffer.append(exp)

    def __len__(self):
        return self.__length

    def sample_batch(self, size_batch):

        if self.__length < size_batch:
            batch = random.sample(self.__buffer, self.__length)
        else:
            batch = random.sample(self.__buffer, size_batch)
        return batch

    def clear(self):
        self.__buffer.clear()
        self.__length = 0


