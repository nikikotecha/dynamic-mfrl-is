import random
import sys
import time

import numpy as np

from networks import Networks
from parsed_args_ssd import args
#from sequential_social_dilemma_games.social_dilemmas.envs.cleanup import CleanupEnvMultiType, CleanupEnvReward
#from sequential_social_dilemma_games.social_dilemmas.envs.harvest import HarvestEnvReward
from env3rundivproduct import MultiAgentInvManagementDiv
import utils_all
import utils_ssd

"""
Notes: You will run this 'main_ssd.py' file but you should change settings in 'parsed_args_ssd.py'
"""


GREEDY_BETA = 0.001


def roll_out(networks, env, args, init_set, epi_num, explore_params, paths, is_draw=False, is_train=True):
    """
    Run the simulation over epi_length and get samples from it.

    Parameters
    ----------
    networks: Networks
    env: CleanupEnvMultiType | CleanupEnvReward | HarvestEnvReward
    args: argparse.Namespace
    init_set: dict
        init_set = {'obs': dict, 'm_act': list of dict}
    epi_num: int
    explore_params: dict
    paths: list
    is_draw: bool
    is_train: bool

    Returns
    ----------
    samples: list
        List of samples which are tuples.
        Length of samples will be the epi_length.
        ex. [(obs, act, rew, m_act, fea), (obs, act, rew, m_act, fea), ...]
    init_set: dict
    collective_reward: list[float]
        Collective reward of this episode.
    collective_feature: list[np.ndarray]
        Collective feature of this episode.
        This is used for calculating total_incentives and total_penalties.
        ex. np.array([x, y])
    """
    image_path, video_path, saved_path = paths

    epi_length = args.episode_length
    fps = args.fps
    num_types = env.num_types

    #agent_ids = list(env.node_names.keys())
    agent_ids = env.node_names
    agent_types = list(env.agent_types.values())  # env.agents_types = {agent_id: int}
    prev_steps = 0  # epi_num * epi_length
    samples = [None] * epi_length
    collective_reward = [0 for _ in range(num_types)]
    collective_feature = [np.zeros(np.prod(env.observation_space.shape)) for _ in range(num_types)]

    obs = init_set['obs']  # Initial observations.
    prev_m_act = init_set['m_act']  # Initial previous mean actions which is only used for Boltzmann policy.
    # Run the simulation (or episode).
    for i in range(epi_length):
        # Select actions.
        epsilon = explore_params['epsilon']
        beta = explore_params['beta']
        if args.mode_ac:
            rand_prob = np.random.rand(1)[0]
            if is_train and rand_prob < epsilon:
                act = {agent_id: np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=env.action_space.shape) for agent_id in agent_ids}                
                low = env.action_space.low
                high = env.action_space.high
                pdf_value = 1 / (high - low)
                if pdf_value == 0:
                    pdf_value = 0 + 1e-8 #ensures non zero, undefined value

                act_probs = {agent_id: np.log(pdf_value) for agent_id in agent_ids}
            else:
                act, act_probs = networks.get_actions(obs, prev_m_act, GREEDY_BETA, is_target=False)  # beta will not do anything here.
        else:  # Boltzmann policy using Q value based on critic or psi.
            if is_train:
                act, act_probs = networks.get_actions(obs, prev_m_act, beta, is_target=False)
            else:
                act, act_probs = networks.get_actions(obs, prev_m_act, GREEDY_BETA, is_target=False)

        # Save the image.
        if is_draw:
            if i == 0:
                print("Run the episode with saving figures...")
            filename = image_path + "frame" + str(prev_steps + i).zfill(9) + ".png"
            env.render(filename=filename, i=prev_steps + i, epi_num=epi_num, act_probs=act_probs)

        # Step
        obs, act, rew, m_act, n_obs, fea = env.step(act)

        # Add one-transition sample to samples if is_train=True.
        if is_train:
            samples[i] = (obs, act, act_probs, rew, m_act, n_obs, fea, beta)
            print("act_probs: ", act_probs)

        # Update collective_reward and collective_feature for each type.
        for idx, agent_id in enumerate(agent_ids):
            agent_type = agent_types[idx]
            collective_reward[agent_type] += rew[agent_id]
            collective_feature[agent_type] += fea[agent_id]

        sys.stdout.flush()

        # Update obs and prev_m_act for the next step.
        obs = n_obs
        prev_m_act = m_act

        # Save the last image.
        if is_draw and i == epi_length - 1:
            act_probs = {agent_ids[j]: "End" for j in range(len(agent_ids))}
            filename = image_path + "frame" + str(prev_steps + i + 1).zfill(9) + ".png"
            env.render(filename=filename, i=prev_steps + i + 1, epi_num=epi_num, act_probs=act_probs)

    # Save the video.
    #utils_ssd.make_video(is_train, epi_num, fps, video_path, image_path) if is_draw else None

    # Reset the environment after roll_out.
    
    init_set: dict = env.reset()

    return samples, init_set, collective_reward, collective_feature


if __name__ == "__main__":
    # Seed setting.
    utils_all.set_random_seed(args.random_seed)

    # Build the environment.
    env = utils_ssd.get_env(args)
    init_set = env.reset()

    # Build networks
    networks = Networks(env, args)

    # Initial exploration probability.
    epsilon = args.epsilon
    beta = args.beta
    explore_params = {'epsilon': epsilon, 'beta': beta}

    # Build paths for saving images.
    path, image_path, video_path, saved_path = utils_ssd.make_dirs(args)
    paths = [image_path, video_path, saved_path]

    # Metrics
    collective_rewards, collective_rewards_test = [np.zeros([args.num_types, args.num_episodes]) for _ in range(2)]
    total_penalties, total_penalties_test = [np.zeros(args.num_episodes) for _ in range(2)]
    total_incentives, total_incentives_test = [np.zeros(args.num_episodes) for _ in range(2)]
    objectives, objectives_test = [np.zeros(args.num_episodes) for _ in range(2)]
    time_start = time.time()

    # Save current setting(args) to txt for easy check
    utils_all.make_setting_txt(args, path)

    # Buffer
    buffer = []

    # Run
    for i in range(args.num_episodes):
        # Option for visualization.
        #is_draw = (True and args.mode_draw) if (i == 0 or (i + 1) % args.save_freq == 0) else False
        is_draw = False
        # Decayed exploration parameters.
        explore_params = utils_ssd.get_explore_params(explore_params, i, args)

        # Run roll_out function (We can get 1,000 (epi_length) samples and collective reward of this episode).
        samples, init_set, collective_reward, collective_feature = roll_out(networks=networks,
                                                                            env=env,
                                                                            args=args,
                                                                            init_set=init_set,
                                                                            epi_num=i,
                                                                            explore_params=explore_params,
                                                                            paths=paths,
                                                                            is_draw=is_draw,
                                                                            is_train=True,
                                                                            )
        buffer += samples
        for agent_type in range(args.num_types):
            collective_rewards[agent_type, i] = collective_reward[agent_type]
        if 'cleanup' in args.env:
            collective_feature = sum(collective_feature)  # Aggregate collective_features of each type.
            total_penalties[i] = collective_feature[0] * args.lv_penalty
            total_incentives[i] = collective_feature[1] * args.lv_incentive
            objectives[i] = sum(collective_rewards[:, i]) + total_penalties[i] - total_incentives[i]
        elif 'harvest' in args.env:
            collective_feature = sum(collective_feature)  # Aggregate collective_features of each type.
            total_penalties[i] = -collective_feature[1] * args.lv_penalty
            total_incentives[i] = collective_feature[2] * args.lv_incentive
            objectives[i] = sum(collective_rewards[:, i]) + total_penalties[i] - total_incentives[i]
        else:
            objectives[i] = sum(collective_rewards[:, i])
        buffer = buffer[-args.buffer_size:]

        # Update networks
        if (i + 1) % args.update_freq == 0:
            k_samples = random.choices(buffer, k=args.K)
            networks.update_networks(k_samples)

        # Update target networks
        if (i + 1) % args.update_freq_target == 0:
            networks.update_target_networks()

        # Print status
        update = "O" if (i + 1) % args.update_freq == 0 else "X"
        print(f"Process : {i}/{args.num_episodes}, "
              f"Time : {time.time() - time_start:.2f}, "
              f"Collective reward (all types) : {sum(collective_rewards[:, i]):.2f}, "
              f"Objective : {objectives[i]:.2f}, "
              f"Update : {update}, "
              f"Train")

        # Test
        if args.mode_test:
            samples, init_set, collective_reward, collective_feature = roll_out(networks=networks,
                                                                                env=env,
                                                                                args=args,
                                                                                init_set=init_set,
                                                                                epi_num=i,
                                                                                explore_params=explore_params,
                                                                                paths=paths,
                                                                                is_draw=is_draw,
                                                                                is_train=False,
                                                                                )

            for agent_type in range(args.num_types):
                collective_rewards_test[agent_type, i] = collective_reward[agent_type]
            if 'cleanup' in args.env:
                collective_feature = sum(collective_feature)  # Aggregate collective_features of each type.
                total_penalties_test[i] = collective_feature[0] * args.lv_penalty
                total_incentives_test[i] = collective_feature[1] * args.lv_incentive
                objectives_test[i] = sum(collective_rewards_test[:, i]) + total_penalties_test[i] - total_incentives_test[i]
            elif 'harvest' in args.env:
                collective_feature = sum(collective_feature)  # Aggregate collective_features of each type.
                total_penalties_test[i] = -collective_feature[1] * args.lv_penalty
                total_incentives_test[i] = collective_feature[2] * args.lv_incentive
                objectives_test[i] = sum(collective_rewards_test[:, i]) + total_penalties_test[i] - total_incentives_test[i]
            else:
                objectives_test[i] = sum(collective_rewards_test[:, i])

            # Print status
            print(f"Process : {i}/{args.num_episodes}, "
                  f"Time : {time.time() - time_start:.2f}, "
                  f"Collective reward (all types) : {sum(collective_rewards_test[:, i]):.2f}, "
                  f"Objective : {objectives_test[i]:.2f}, "
                  f"Test")

        if (i + 1) % args.draw_freq == 0 and args.mode_draw:
            utils_ssd.draw_or_save_plt(collective_rewards,
                                       collective_rewards_test,
                                       objectives,
                                       objectives_test,
                                       i=i,
                                       mode='draw',
                                       )

        # Save several things
        if (i + 1) % args.save_freq == 0:
            time_trained = time.time() - time_start
            filename = str(i).zfill(9) + '.tar'
            filename_plt = saved_path + 'outcomes_' + str(i).zfill(9) + '.png'
            utils_ssd.draw_or_save_plt(collective_rewards,
                                       collective_rewards_test,
                                       objectives,
                                       objectives_test,
                                       i=i,
                                       mode='save',
                                       filename=filename_plt,
                                       )
            utils_ssd.save_data(args=args,
                                env=env,
                                networks=networks,
                                explore_params=explore_params,
                                episode_trained=i,
                                time_trained=time_trained,
                                outcomes={'collective_rewards': collective_rewards,
                                          'collective_rewards_test': collective_rewards_test,
                                          'total_penalties': total_penalties,
                                          'total_penalties_test': total_penalties_test,
                                          'total_incentives': total_incentives,
                                          'total_incentives_test': total_incentives_test,
                                          'objectives': objectives,
                                          'objectives_test': objectives_test,
                                          },
                                path=saved_path,
                                name=filename)