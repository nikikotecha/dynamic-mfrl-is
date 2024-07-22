"""Self Play
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env3rundivproduct import MultiAgentInvManagementDiv

from archive.run import play
from algo import spawn_ai
from algo import tools


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs('./data', exist_ok=True)

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'iql'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=5, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=1, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--max_steps', type=int, default=10, help='set the max steps')
    parser.add_argument('--cuda', type=bool, default=False, help='use the cuda')

    args = parser.parse_args()

    # Initialize the environment
    env = MultiAgentInvManagementDiv(config = {})


    handles = env.group_handles
    handles_dict = env.group_handles_dict
    # Get the number of agents in each group
    print("Handles: ", handles)
    num_agents_group_0 = env.get_num(handles[0], handles_dict)
    num_agents_group_1 = env.get_num(handles[1], handles_dict)

    print(f"Number of agents in group 0: {num_agents_group_0}")
    print(f"Number of agents in group 1: {num_agents_group_1}")

    log_dir = os.path.join(BASE_DIR, 'data/tmp/{}'.format(args.algo))
    render_dir = os.path.join(BASE_DIR, 'data/render/{}'.format(args.algo))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.algo))

    start_from = 0

    models = [spawn_ai(args.algo, env, handles[0], args.algo + '-me', args.max_steps, args.cuda), spawn_ai(args.algo, env, handles[1], args.algo + '-opponent', args.max_steps, args.cuda)]
    runner = tools.Runner(env, handles, args.max_steps, models, play,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, render_dir=render_dir, train=True, cuda=args.cuda)

    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
