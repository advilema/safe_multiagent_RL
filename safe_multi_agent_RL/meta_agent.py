import numpy as np


class MetaAgent():
    def __init__(self, constraint_space, gamma, lr, thresholds, leq=True, start_learning_cycle=10, decay=1.0):
        self.constraint_space = constraint_space
        self.gamma = gamma
        self.thresholds = np.array(thresholds)
        self.leq = leq  # constraints lower equal (<=) thresholds
        self.batch_constraints = []
        self.constraint_values = []
        self.lambdas = np.zeros(sum(constraint_space))
        self.lr = lr
        self.learning_cycle = 0
        self.start_learning_cycle = start_learning_cycle
        self.decay = decay

    def act(self, constraint, reward):
        if self.learning_cycle >= self.start_learning_cycle:
            self.batch_constraints.append(constraint)
        sign = -1 if self.leq else 1
        modified_rew = sign * np.inner(self.lambdas, np.array(constraint)) + np.array(reward)
        return modified_rew.tolist()

    def step(self):
        if not len(self.batch_constraints):
            return
        discounts = [self.gamma ** i for i in range(len(self.batch_constraints) + 1)]
        batch_constraint_values = [sum([a * b for a, b in zip(discounts, constr)])
                                   for constr in np.array(self.batch_constraints).T]
        self.constraint_values.append(batch_constraint_values)
        self.batch_constraints = []

    def update(self):
        sign = 1 if self.leq else -1
        mean_values = np.array([np.mean(value) for value in np.array(self.constraint_values).T])
        self.lambdas += sign * self.lr * (mean_values - self.thresholds)
        self.lambdas = np.array([max(lam, 0.0) for lam in self.lambdas])
        self.lr = self.lr / self.decay
        self.constraint_values = []
        self.learning_cycle = 0

    def increment_learning_cycle(self):
        self.learning_cycle += 1
