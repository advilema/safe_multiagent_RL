import numpy as np
import matplotlib.pyplot as plt
import json
import os


class Buffer:
    def __init__(self, params, constrained=True, save_path=None):
        self.params = params
        self.save_path = save_path
        self.constrained = constrained
        self.lambdas = []

        self.rewards = []
        self.modified_rewards = []
        self.batch_constraints = []

        self.scores = []
        self.modified_scores = []
        self.constraints = []

    def append(self, reward, modified_reward, constraint):
        self.rewards.append(reward)
        self.modified_rewards.append(modified_reward)
        self.batch_constraints.append(constraint)

    def append_lambdas(self, lambdas):
        self.lambdas.append(lambdas)

    def step(self):
        discounts = [self.params.gamma ** i for i in range(len(self.rewards) + 1)]
        R = [sum([gamma_t * r for gamma_t, r in zip(discounts, agent_rewards)])
             for agent_rewards in np.array(self.rewards).T]
        modified_R = [sum([gamma_t * mod_r for gamma_t, mod_r in zip(discounts, mod_agent_rewards)])
                      for mod_agent_rewards in np.array(self.modified_rewards).T]

        self.scores.append(R)
        self.modified_scores.append(modified_R)
        self.constraints.append([sum(constr) for constr in np.array(self.batch_constraints).T])

        self.rewards = []
        self.modified_rewards = []
        self.batch_constraints = []

    def mean_score(self, n=100):
        return np.mean(np.array(self.scores).T[0, -n:]), np.mean(np.array(self.modified_scores).T[0, -n:]), \
               [np.mean(constr[-n:]) - thresh for constr, thresh in
                zip(np.array(self.constraints).T, self.params.thresholds)]

    def save_results(self):
        if self.save_path is None:
            if not os.path.isdir("results/"):
                os.mkdir("results/")
            path = "results/" + self.params.environment + "_s" + str(self.params.size) + "_n" + str(self.params.n_agents) + \
                   "_" + str(self.params.numpy_seed) + "-" + str(self.params.torch_seed) + '_'
            if not self.constrained:
                path += 'unconstr_'
            i = 0
            while True:
                if not os.path.isdir(path + str(i)):
                    self.save_path = path + str(i)
                    os.mkdir(self.save_path)
                    break
                i += 1

        n_batches = len(self.scores) + 1
        batch_scores, batch_modified_scores = self._get_batch_scores()

        # plot scores
        plt.figure()
        for ag in range(np.array(batch_scores).shape[1]):
            plt.plot(np.arange(1, n_batches, self.params.batch_size), np.array(batch_scores).T[ag])
        plt.plot([1, n_batches], [0, 0], 'r')  # plot maximum achievable score
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(self.save_path + '/scores' + '.png')
        plt.close()

        # plot constraints
        batch_constraints = np.array(self._get_batch_constraints())
        maximum_constr = np.array(self.params.max_t) - np.array(self.params.thresholds)
        minimum_constr = - np.array(self.params.thresholds)
        plt.figure()
        [plt.plot(np.arange(1, n_batches, self.params.batch_size), constr, label='agent ' + str(agent))
         for agent, constr in enumerate(batch_constraints.T)]
        plt.plot([1, n_batches], [0, 0], 'r')  # below this line the constraints are satisfied
        #plt.plot([1, n_batches], [maximum_constr, maximum_constr], color='black')
        #plt.plot([1, n_batches], [minimum_constr, minimum_constr], color='black')
        plt.ylabel('Constraints')
        plt.xlabel('Episode #')
        plt.legend()
        plt.savefig(self.save_path + '/constr.png')
        plt.close()

        # plot how much constraints are not respected
        constr_excess = np.maximum(batch_constraints, np.zeros(batch_constraints.shape))
        constr_excess = np.sum(constr_excess, axis=1)
        plt.figure()
        plt.plot(np.arange(1, n_batches, self.params.batch_size), constr_excess)
        plt.plot([1, n_batches], [0, 0], 'r')
        plt.ylabel('Constraint Excess')
        plt.xlabel('Episode #')
        plt.savefig(self.save_path + '/constr_excess.png')
        plt.close()

        """
        #plot constraints regret
        constr_integral = [constr_excess[0]]
        for i in range(1, len(constr_excess)):
            constr_integral.append(constr_integral[i-1]+constr_excess[i])
        constr_regret = [c/(i+1) for i, c in enumerate(constr_integral)]
        plt.figure()
        plt.plot(np.arange(1, n_batches, self.params.batch_size), constr_regret)
        plt.plot([1, n_batches], [0, 0], 'r')
        plt.ylabel('Constraint Regret')
        plt.xlabel('Episode #')
        plt.savefig(self.save_path + '/constr_regret.png')
        plt.close()
        """

        if self.constrained:
            # plot modified scores
            for agent in range(self.params.n_agents):
                plt.figure()
                plt.plot(np.arange(1, n_batches, self.params.batch_size), np.array(batch_scores).T[agent], label='original')
                plt.plot(np.arange(1, n_batches, self.params.batch_size), np.array(batch_modified_scores).T[agent], label='modified')
                plt.plot([1, n_batches], [0, 0], 'r')  # plot maximum achievable score
                plt.ylabel('Score')
                plt.xlabel('Episode #')
                plt.legend()
                plt.savefig(self.save_path + '/modified_scores_agent' + str(agent) + '.png')
                plt.close()

            # plot lambdas
            plt.figure()
            [plt.plot(np.arange(1, n_batches, self.params.batch_size * self.params.n_agents_learning_cycles), lam,
                      label="agent " + str(agent))
             for agent, lam in enumerate(np.array(self.lambdas).T)]
            plt.ylabel('Constraints')
            plt.xlabel('Episode #')
            plt.savefig(self.save_path + '/lambdas')
            plt.close()

        path_json = self.save_path + '/params.json'
        with open(path_json, 'w') as file:
            #json.dump(self.params._asdict(), file)
            json.dump(vars(self.params), file)

        np.save(self.save_path + '/constr.npy', self.constraints)
        np.save(self.save_path + '/scores.npy', self.scores)
        np.save(self.save_path + '/modified_scores.npy', self.modified_scores)
        if self.constrained:
            np.save(self.save_path + '/lambdas.npy', self.lambdas)
        return

    def _get_batch_scores(self):
        batch_scores = []
        batch_modified_scores = []
        j = 0
        batch_scores.append(self.scores[0])
        batch_modified_scores.append(self.modified_scores[0])
        for i in range(self.params.batch_size, len(self.scores), self.params.batch_size):
            batch_scores.append([np.mean(agent_scores[j:i]) for agent_scores in np.array(self.scores).T])
            batch_modified_scores.append(
                [np.mean(mod_agent_scores[j:i]) for mod_agent_scores in np.array(self.modified_scores).T])
            j = i
        return batch_scores, batch_modified_scores

    def _get_batch_constraints(self):
        batch_constraints = []
        j = 0
        batch_constraints.append(np.array(self.constraints[0]) - self.params.thresholds)
        for i in range(self.params.batch_size, len(self.constraints), self.params.batch_size):
            batch_constraints.append([np.mean(constr[j:i]) - thresh for constr, thresh in
                                      zip(np.array(self.constraints).T, self.params.thresholds)])
            j = i
        return batch_constraints
