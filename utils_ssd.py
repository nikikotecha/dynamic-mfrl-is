import argparse
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

#from sequential_social_dilemma_games.social_dilemmas.envs.cleanup import CleanupEnvMultiType, CleanupEnvReward
#from sequential_social_dilemma_games.social_dilemmas.envs.env_creator import get_env as ssd_get_env
#from sequential_social_dilemma_games.social_dilemmas.envs.harvest import HarvestEnvReward
#from sequential_social_dilemma_games.utility_funcs import make_video_from_image_dir
from env3rundivproduct import MultiAgentInvManagementDiv
from parsed_args_ssd import config_args

# def make_dirs(args: argparse.Namespace) -> (str, str, str, str):
#     path = "results_ssd/" + args.setting_name
#     if path is None:
#         path = os.path.abspath(os.path.dirname(__file__)) + "/results_ssd" + args.setting_name
#         if not os.path.exists(path):
#             os.makedirs(path)
#     image_path = os.path.join(path, "frames/")
#     if not os.path.exists(image_path):
#         os.makedirs(image_path)
#     video_path = os.path.join(path, "videos/")
#     if not os.path.exists(video_path):
#         os.makedirs(video_path)
#     saved_path = os.path.join(path, "saved/")
#     if not os.path.exists(saved_path):
#         os.makedirs(saved_path)
#
#     return path, image_path, video_path, saved_path


def make_dirs(args: argparse.Namespace, folder_location: str = 'current_folder') -> (str, str, str, str):
    if folder_location == 'current_folder':
        path = './results_ssd/' + args.setting_name
    elif folder_location == 'upper_folder':
        path = '../results_ssd/' + args.setting_name
    else:
        raise ValueError(f'There is no folder option: {folder_location}.')
    if path is None:
        path = os.path.abspath(os.path.dirname(__file__)) + "/results_ssd" + args.setting_name
        if not os.path.exists(path):
            os.makedirs(path)
    image_path = os.path.join(path, "frames/")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    video_path = os.path.join(path, "videos/")
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    saved_path = os.path.join(path, "saved/")
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    return path, image_path, video_path, saved_path


"""def make_video(is_train: bool, epi_num: int, fps: int, video_path: str, image_path: str):
    if is_train:
        video_name = "trajectory_train_episode_" + str(epi_num)
    else:
        video_name = "trajectory_test_episode_" + str(epi_num)
    make_video_from_image_dir(video_path, image_path, fps=fps, video_name=video_name)
    # Clean up images.
    for single_image_name in os.listdir(image_path):
        single_image_path = os.path.join(image_path, single_image_name)
        try:
            if os.path.isfile(single_image_path) or os.path.islink(single_image_path):
                os.unlink(single_image_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (single_image_path, e))"""


def get_env(args: argparse.Namespace) -> Union[MultiAgentInvManagementDiv]:
    config = config_args()
    env = MultiAgentInvManagementDiv(config)
    return env


def get_explore_params(explore_params: dict, i: int, args: argparse.Namespace) -> dict:
    def get_decayed_epsilon(prev_decayed_eps: float, i: int, args: argparse.Namespace) -> float:
        if args.epsilon_decay_ver == 'linear':
            decayed_eps = args.epsilon * (1 - 0.98 * i / args.num_episodes)
        elif args.epsilon_decay_ver == 'exponential':
            decayed_eps = max(prev_decayed_eps * 0.9999, 0.01)
        else:
            raise ValueError("The version of epsilon decay is not matched with current implementation.")
        return decayed_eps

    def get_decayed_beta(prev_decayed_beta: float, i: int, args: argparse.Namespace) -> float:
        decayed_beta = None
        if args.beta_decay_ver == 'linear':
            # y = [1, 0.2, 0.1]  # v0
            # x = [0, int(args.num_episodes * 0.8), int(args.num_episodes)]  # v0
            # y = [1, 0.01, 0.01]  # v1
            # x = [0, int(args.num_episodes * 0.8), int(args.num_episodes)]  # v1
            # y = [args.beta, 0.3, 0.01, 0.01]  # v2
            # x = [0, int(args.num_episodes * 0.4), int(args.num_episodes * 0.8), int(args.num_episodes)]  # v2
            # y = [args.beta, 0.3, 0.01, 0.01]  # v3
            # x = [0, int(args.num_episodes * 0.2), int(args.num_episodes * 0.5), int(args.num_episodes)]  # v3
            # y = [args.beta, 0.3, 0.01, 0.001]  # v4
            # x = [0, int(args.num_episodes * 0.2), int(args.num_episodes * 0.5), int(args.num_episodes)]  # v4
            y = [args.beta, 0.1, 0.01, 0.001]  # v5
            x = [0, int(args.num_episodes * 0.2), int(args.num_episodes * 0.5), int(args.num_episodes)]  # v5
            min_v = y[0]
            if i == 0:
                decayed_beta = min_v
            else:
                for t, x_t in enumerate(x):
                    if i <= x_t:
                        interval = (y[t] - y[t - 1]) / (x_t - x[t - 1])
                        decayed_beta = interval * (i - x[t - 1]) + y[t - 1]
                        break
        elif args.beta_decay_ver == 'exponential':
            decayed_beta = max(prev_decayed_beta * 0.9999, 0.01)
        else:
            raise ValueError("The version of beta decay is not matched with current implementation.")
        return decayed_beta

    epsilon = explore_params['epsilon']
    beta = explore_params['beta']

    decayed_epsilon = get_decayed_epsilon(epsilon, i, args)
    decayed_beta = get_decayed_beta(beta, i, args)

    decayed_explore_params = {'epsilon': decayed_epsilon, 'beta': decayed_beta}

    return decayed_explore_params


def save_data(args, env, networks, explore_params, episode_trained, time_trained, outcomes, path, name):
    """
    Save several data.

    Parameters
    ----------
    args: argparse.Namespace
    env: CleanupEnvReward | CleanupEnvMultiType | HarvestEnvReward
    episode_trained: int
    explore_params: dict
    time_trained: float
    outcomes: dict
    networks: networks_ssd.Networks
    path: str
    name: str
    """
    actor_params: list = []
    actor_opt_params: list = []
    critic_params: list = []
    critic_opt_params: list = []
    psi_params: list = []
    psi_opt_params: list = []
    for agent_type in range(args.num_types):
        actor_param = networks.actor[agent_type].state_dict() if args.mode_ac else None
        actor_opt_param = networks.actor_opt[agent_type].state_dict() if args.mode_ac else None
        psi_param = networks.psi[agent_type].state_dict() if args.mode_psi else None
        psi_opt_param = networks.psi_opt[agent_type].state_dict() if args.mode_psi else None
        critic_param = networks.critic[agent_type].state_dict() if not args.mode_psi else None
        critic_opt_param = networks.critic_opt[agent_type].state_dict() if not args.mode_psi else None
        actor_params.append(actor_param)
        actor_opt_params.append(actor_opt_param)
        psi_params.append(psi_param)
        psi_opt_params.append(psi_opt_param)
        critic_params.append(critic_param)
        critic_opt_params.append(critic_opt_param)
    torch.save({
        'args': args,
        'env': env,
        'actor': actor_params,
        'actor_opt': actor_opt_params,
        'psi': psi_params,
        'psi_opt': psi_opt_params,
        'critic': critic_params,
        'critic_opt': critic_opt_params,
        'explore_params': explore_params,
        'episode_trained': episode_trained,
        'time_trained': time_trained,
        'outcomes': outcomes,
    }, path + name)


def draw_or_save_plt(col_rews, col_rews_test, objs, objs_test, i=0, mode='draw', filename=''):
    """
    Draw or save plt using collective rewards and objective values.
    220915:
    Function is updated to reflect multi-type outcomes.

    If col_rews is a 1D numpy.ndarray, it works as a previous version.
    If col_rew is a 2D numpy ndarray, it means that each row contains collective rewards for each type.
    This function adds all rows to build a single value.

    Parameters
    ----------
    col_rews: numpy.ndarray
    col_rews_test: numpy.ndarray
    objs: numpy.ndarray
    objs_test: numpy.ndarray
    i: int
    mode: str
    filename: str
    """
    def get_figure_components(inputs: np.ndarray, i: int) -> (np.ndarray, np.ndarray, np.ndarray):
        rew = inputs[:i + 1]
        moving_avg_len = 20
        means, stds = [np.zeros(rew.size) for _ in range(2)]
        for j in range(rew.size):
            if j + 1 < moving_avg_len:
                rew_part = rew[:j + 1]
            else:
                rew_part = rew[j - moving_avg_len + 1:j + 1]
            means[j] = np.mean(rew_part)
            stds[j] = np.std(rew_part)
        return rew, means, stds

    if col_rews.ndim == 1:
        pass
    elif col_rews.ndim == 2:
        col_rews = np.sum(col_rews, axis=0)
        col_rews_test = np.sum(col_rews_test, axis=0)
    else:
        raise ValueError

    # Set axis.
    x_axis = np.arange(i+1)
    y_axis_lim_rew = np.max(col_rews[:i + 1]) + 100
    y_axis_lim_rew_test = np.max(col_rews_test[:i + 1]) + 100
    y_axis_lim_obj = np.max(objs[:i + 1]) + 1
    y_axis_lim_obj_test = np.max(objs_test[:i + 1]) + 1

    # Build figure.
    plt.figure(figsize=(16, 14))

    plt.subplot(2, 2, 1)
    outs, means, stds = get_figure_components(col_rews, i)
    plt.plot(x_axis, means, label='Moving avg. of collective rewards', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Collective rewards')
    plt.ylim([0, y_axis_lim_rew])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Collective rewards per episode', fontsize=20)
    plt.title('Collective rewards (train)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.subplot(2, 2, 2)
    outs, means, stds = get_figure_components(col_rews_test, i)
    plt.plot(x_axis, means, label='Moving avg. of collective rewards', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Collective rewards')
    plt.ylim([0, y_axis_lim_rew_test])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Collective rewards per episode', fontsize=20)
    plt.title('Collective rewards (test)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.subplot(2, 2, 3)
    outs, means, stds = get_figure_components(objs, i)
    plt.plot(x_axis, means, label='Moving avg. of designer objectives', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Designer objectives')
    plt.ylim([0, y_axis_lim_obj])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Designer objectives per episode', fontsize=20)
    plt.title('Designer objectives (train)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.subplot(2, 2, 4)
    outs, means, stds = get_figure_components(objs_test, i)
    plt.plot(x_axis, means, label='Moving avg. of designer objectives', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Designer objectives')
    plt.ylim([0, y_axis_lim_obj_test])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Designer objectives per episode', fontsize=20)
    plt.title('Designer objectives (test)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    if mode == 'draw':
        plt.show()
    elif mode == 'save':
        plt.savefig(filename)
    else:
        raise ValueError