import argparse


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unconstrained', type=bool, default=False)
    parser.add_argument('--numpy_seed', type=int, default=0)
    parser.add_argument('--torch_seed', type=int, default=0)
    parser.add_argument('--environment', type=str, default='ExploreContinuous')
    parser.add_argument('--n_meta_agent_learning_cycles', type=int, default=50)
    parser.add_argument('--n_agents_learning_cycles', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_t', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--meta_lr', type=float, default=0.003)
    parser.add_argument('--decay', type=float, default=0.003)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--weights', type=list, default=[1,2,3])
    parser.add_argument('--thresholds', type=list, default=[25, 25, 25])
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--print_every', type=int, default=1)
    params = parser.parse_args()
    return params