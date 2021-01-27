import argparse
import torch
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # checkpoint
    parser.add_argument(
        '--name',
        type=str,
        default='gail',
        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument(
        '--which_epoch',
        type=str, default='latest',
        help='which epoch to load? set to latest to use latest cached model')

    # behavioral cloning
    parser.add_argument(
        '--bc-batch-size',
        type=int,
        default=64,
        help='behavioral cloning batch size (default: 64)')
    parser.add_argument(
        '--bc-epoch',
        type=int,
        default=100,
        help='number of behavioral cloning epochs (default: 100)')

    # gail / ppo
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')

    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--results-root',
        default='./results/',
        help='directory to store gail results (gym env snapshots)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='gpu id to use'
    )
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    # network architecture
    parser.add_argument(
        '--adjust-scale',
        action='store_true',
        default=False,
        help='add a non-linearity to the final layer of generator, giving some boost in performance in sog-gail')
    parser.add_argument(
        '--wasserstein',
        action='store_true',
        default=False,
        help='use wasserstein loss for discriminator as advised by infogail paper')

    # infogail
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=3,
        help='dim of latent codes')
    parser.add_argument(
        '--bc-pretrain',
        action='store_true',
        default=False,
        help='pretrain the generator with behavioral cloning before training')
    parser.add_argument(
        '--infogail',
        action='store_true',
        default=False,
        help='use infogail model (additional posterior network)')
    parser.add_argument(
        '--infogail-coef',
        type=float,
        default=0.1,
        help='mutual entropy lower bound coefficient (default: 0.1)')

    # sog
    parser.add_argument(
        '--sog-gail',
        action='store_true',
        default=False,
        help='use sog-gail model (additional sog module)')
    parser.add_argument(
        '--sog-gail-coef',
        type=float,
        default=0.01,
        help='sog-gail term coefficient (default: 0.01)')
    parser.add_argument(
        '--shared-code',
        action='store_true',
        default=False,
        help='solve for a "shared" latent code for expert trajectories, in sog-gail model')
    parser.add_argument(
        '--block_size',
        type=int,
        default=2,
        help='size of coordinate search blocks')
    parser.add_argument(
        '--n_rounds',
        type=int,
        default=1,
        help='number of coordinate search rounds')
    parser.add_argument(
        '--latent_optimizer',
        type=str,
        default='bcs',
        help='method to find best latent code: e.g. "bcs" for block coorindate search, or "ohs" for one-hot-search.')

    # custom envs
    parser.add_argument(
        '--env-name',
        type=str,
        default='Circles-v0',
        help='environment to train')
    parser.add_argument(
        '--radii',
        type=str, default='-10,10,20',
        help='a list of radii to be sampled uniformly at random for "Circles-v0" environment. a negative sign implies that the circle is to be drawn downwards. you may also input expressions such as "np.linspace(-10,10,100)".')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    args.results_dir = os.path.join(args.results_root, args.env_name.split('-')[0].lower(), args.name)

    if args.env_name == 'Circles-v0':
        args.radii = eval(args.radii)
        # maximum action magnitude in Circles-v0 environment
        args.max_ac_mag = max(map(abs, args.radii)) * 0.075

    args.vanilla = not (args.infogail or args.sog_gail)
    if args.infogail and args.sog_gail:
        raise ValueError('cannot raise --infogail and --sog-gail flags concurrently.')
    # args.adjust_scale |= args.sog_gail
    # args.wasserstein |= args.infogail

    # TODO Arash: separate away train/test options

    return args
