import argparse


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unconstrained', action='store_true')
    parser.add_argument('--numpy_seed', type=int, default=0)
    parser.add_argument('--torch_seed', type=int, default=0)
    parser.add_argument('--environment', type=str, default='Space')
    parser.add_argument('--algo', type=str, default='ppo') #either reinforce, ac or ppo
    parser.add_argument('--n_meta_agent_learning_cycles', type=int, default=40)
    parser.add_argument('--n_agents_learning_cycles', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_t', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_lr', type=float, default=0.0015)
    parser.add_argument('--critic_lr', type=float, default=0.0005)
    parser.add_argument('--meta_lr', type=float, default=0.002)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--lambda_0', type=float, default=0)
    parser.add_argument('--size', type=int, default=3)
    parser.add_argument('--coarseness', type=int, default=None)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--n_landmarks', type=int, default=1)
    parser.add_argument('--weights', nargs="*", type=float, default=[1, 1, 1, 1, 1,1,1,1,1,1])
    parser.add_argument('--thresholds', nargs="*", type=float, default=[2])
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--checkpoints', action='store_true')
    params = parser.parse_args()
    return params
