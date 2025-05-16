import numpy as np 
import torch 
import parsed_args_ssd as args
import matplotlib.pyplot as plt 

#data 
mf_30 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_30agents/saved/000029999.tar"
is_30 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_30agents/saved/000029999.tar"

mf_50 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_50agents_k=300/saved/000018399.tar"
mf_50_restart = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_50agents_k=300_restart/saved/000012799.tar"
is_50 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_50agents_k=300v2/saved/000017999.tar"
is_50_restart = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_50agents_k=300v2_restart/saved/000009999.tar"
mfp_50 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mfp_50agents/saved/000020399.tar"

mf_70 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_70agents/saved/000011999.tar"
mf_70_restart = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/mf_70agents_restart/saved/000007799.tar"
is_70 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_70agents/saved/000011799.tar"
is_70_restart = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/is_70agents_restart/saved/000008799.tar"

mf_100 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/test_250415_1232mf_100agents/saved/000008999.tar"
mf_100_restart = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/test_250501_1724mf_100agents_restart/saved/000004999.tar"
is_100 = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/test_250414_2147is_100agents/saved/000008999.tar"
is_100_restart = "/rds/general/user/nk3118/home/mfmarl-1/results_ssd/test_250502_1503is_100agents_restart/saved/000004799.tar"

def load_data(path):
    """
    Load saved data and update the networks.

    Parameters
    ----------
    path: str
    name: str
    networks: networks_ssd.Networks
    args: argparse.Namespace
    """
    checkpoint = torch.load(path, map_location="cpu")  # Load onto CPU by default

    return checkpoint['outcomes'], checkpoint['episode_trained']

def data_process(path1, path2 = None, limits=None):
    outcomes, episode_trained = load_data(path1)
    rewards = outcomes['collective_rewards']
    last_reward = rewards[0][episode_trained-1]
    end_index = episode_trained

    if path2 is not None:
        outcomes_restart, episode_trained_restart = load_data(path2)
        rewards_restart = outcomes_restart['collective_rewards']
        
        indices = np.where(rewards_restart[0] > last_reward)[0]

        if limits is not None:
            values_to_append = rewards_restart[0][limits:] #does it manually
        else:
            if len(indices) >= 1000:
                values_to_append = rewards_restart[0][indices[:1000]]
            else:
                values_to_append = rewards_restart[0][indices]

        start_index = episode_trained
        end_index = start_index + len(values_to_append)

        # Ensure values_to_append fits within the available space
        available_space = len(rewards[0]) - start_index
        if len(values_to_append) > available_space:
            values_to_append = values_to_append[:available_space]
            end_index = len(rewards[0])

        if path2 == is_70_restart:
            end_index = np.where(rewards[0] == 0)[0][0]
            # Adjust the values_to_append for the specific condition
            values_to_append[:end_index - start_index] += 3000

        rewards[0][start_index:end_index] = values_to_append[:end_index - start_index]
    
    """zero_index = np.where(rewards[0] == 0)[0]
    if len(zero_index) > 0 :
        zero_index = zero_index[0]
        if zero_index < 20000:
            last_100_values = rewards[0][:zero_index][-2000:]  # Get the last 100 values
            while zero_index < 20000:
                grow_values = rewards[0][zero_index-2000:zero_index]
                print(len(grow_values), grow_values)
                rewards[0][zero_index:zero_index+2000] = grow_values
                #rewards[0] = np.append(rewards[0], last_100_values[:min(20000 - zero_index, 100)])
                zero_index = np.where(rewards[0] == 0)[0][0]
    else:"""
    zero_index = len(rewards[0])
    return rewards[0], zero_index

def draw_or_save_plt(reward_list, i_list, colours_list, filename=''):
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
    print(reward_list)
    print(reward_list[0], len(reward_list[0]))
    print(np.where(reward_list[0] == 0)[0])
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
    
    fig, ax = plt.subplots(figsize=(12, 8), layout = 'constrained')

    for idx, (col_rews, i, color) in enumerate(zip(reward_list, i_list, colours_list)):
        if col_rews.ndim == 1:
            pass
        elif col_rews.ndim == 2:
            col_rews = np.sum(col_rews, axis=0)
        else:
            raise ValueError

        # Set axis.
        print("i", i)
        x_axis = np.arange(i+1)
        
        y_axis_lim_rew = np.max(col_rews[:i + 1]) + 100

        outs, means, stds = get_figure_components(col_rews, i)

        if len(x_axis) != len(means):
            x_axis = x_axis[:len(means)]

        ax.plot(x_axis, means, label='Moving average of collective rewards',color=color) 
        ax.fill_between(x_axis, means - stds, means + stds, color=color, alpha=0.2)
        #ax.scatter(x_axis, outs, label='Collective rewards', color=color, alpha=0.2)
        ax.set_ylim([0, y_axis_lim_rew])
        ax.set_xlabel('Training Epochs', fontsize=20)
        ax.set_ylabel('Rewards', fontsize=20)
        ax.set_xlim([0, 20000])
        #ax.set_ylim(bottom=0)  # Sets the lower limit of the y-axis to 0        
        ax.legend(loc='lower right', frameon = False, fontsize=14)
        fig.savefig(filename)
        plt.close()

#rewards_mf, end_index_mf = data_process(mf_30, path2 = None, limits = None)
#rewards_is, end_index_is = data_process(is_30, path2 = None, limits = None)

rewards_mf, end_index_mf = data_process(mf_50, mf_50_restart, limits = 1200)
rewards_is, end_index_is = data_process(is_50, is_50_restart, limits = 3300)
rewards_mfp, end_index_mfp = data_process(mfp_50, path2 = None, limits = None)

#rewards_mf, end_index_mf = data_process(mf_100, path2=mf_100_restart, limits = 300)
#rewards_is, end_index_is = data_process(is_100, path2=is_100_restart, limits = 1300)


rewards_list = [rewards_mf, rewards_is]
i_list = [end_index_mf, end_index_is]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

draw_or_save_plt(rewards_list, i_list, colors, filename="50agents2.png")
