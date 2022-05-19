import argparse
import torch as th

def get_arg():
    parser = argparse.ArgumentParser(description='QMIX-VN')

    # Add config here
    # algorithm config
    # if mixer == None, use IQL where simple_agent will be used, else use cql agent
    parser.add_argument('--rl-model', default='cql',
        help='Reinforcement learning model, select(simple|cql)')
    parser.add_argument('--mixer', default='qmix',
        help='Mixer type, if None means only use local rewards. select(vdn|qmix|multi-qmix|None)')
    #parser.add_argument('--lr', type=float, default=0.0005,help='Learning rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', default="RMSprop",
        help='Optimizer type, including (RMSprop|Adam)')
    parser.add_argument('--optim-alpha', type=float, default=0.99,
        help='RMSprop optimizer alpha')
    parser.add_argument('--optim-eps', type=float, default=1e-5,
        help='RMSprop optimizer eps')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='Discounted factor for reinforcement learning')
    parser.add_argument('--grad-norm-clip', type=float, default=10,
        help='Reduce magnitude of gradients above this L2 norm')
    parser.add_argument('--target-update-interval', type=float, default=200,
        help='Update target network every target-update-interval updates')
    parser.add_argument('--epsilon-start', type=float, default=1.,
        help='Starting value for epsilon decay schedule')
    parser.add_argument('--epsilon-finish', type=float, default=0.01,
        help='Finishing value for epsilon decay schedule')
    parser.add_argument('--epsilon-scheduler', default='exp', 
        help='Epsilon decay scheduler, select (exp|linear), default is exp')
    parser.add_argument("--tau", type=float, default=0.9999,
        help='Update rate of delta for cql algorithm, refer to CMIX paper')
    parser.add_argument('--policy-disc', action='store_true', default=False,
        help='Use discrete policy in constrained RL algorithm')
    parser.add_argument('--weight-alpha', type=float, default=0.1,
        help='Weighted QMIX alpha parameter')
    parser.add_argument('--loss-beta', type=float, default=1.,
        help='Loss combination weight for cql learner') 

    # replay buffer config
    parser.add_argument('--buffer-size', type=int, default=1000,
        help='Buffer size of replay buffer (default 1000)')

    # running config
    parser.add_argument('--application', default='blocker',
            help='Application scenario, select (blocker|vn), i.e. blocker game, vehicular network')
    parser.add_argument('--log-dir', default='test',
        help='Directory to save model logs')
    parser.add_argument('--env-path', default=None,
        help='Environment instance path, if None setup a new environment instance')
    parser.add_argument('--max-env-t', type=int, default=20,
        help='Maximum number of iteration step of the experiments in training epoch')
    parser.add_argument('--training-episodes', type=int,default=1,
        help='Number of training episodes')
    parser.add_argument('--training-epochs', type=int,default=10000,
        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1000,
        help='Sample batch size from replay buffer')
    parser.add_argument('--model-load-path', default=None,
        help='Model parameters loading path, if None do not load model from checkpoint')


    #from here our arguments
    parser.add_argument('--unconstrained', action='store_true')
    parser.add_argument('--numpy_seed', type=int, default=0)
    parser.add_argument('--torch_seed', type=int, default=0)
    parser.add_argument('--environment', type=str, default='Congestion')
    #parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_t', type=int, default=20)
    #parser.add_argument('--gamma', type=float, default=0.99)
    #parser.add_argument('--lr', type=float, default=0.003)
    #parser.add_argument('--meta_lr', type=float, default=0.002)
    parser.add_argument('--decay', type=float, default=1.0)
    #parser.add_argument('--lambda_0', type=float, default=0)
    parser.add_argument('--size', type=int, default=3)
    parser.add_argument('--coarseness', type=int, default=6)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--thresholds', nargs="*", type=float, default=[1.5, 1.5, 1.5])
    parser.add_argument('--shuffle', action='store_true')




    args = parser.parse_args()

    return args
