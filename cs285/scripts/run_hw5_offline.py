import pickle
import time
import argparse
import h5py
from cs285.agents.cql_agent import CQLAgent

from cs285.agents.dqn_agent import DQNAgent
import cs285.env_configs
from cs285.envs import Pointmass

import os
import time
import random

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.agents import agents
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

from scripting_utils import make_logger, make_config

OBSERVATION_SHAPE = (1602, )
NUM_ACTIONS = 4

# generate 2 random integers between 0 and 4


class ReplayBuffer:
    def __init__(self, data_dict):
        
        self.observations = data_dict["observations"]
        self.actions = data_dict["actions"]
        self.next_observations = np.append(data_dict["observations"][1:],
                                           data_dict["observations"][-1:], axis=0)
        self.rewards = data_dict["rewards"]
        # set self.dones to true if either the value at terminals or timeouts is true
        self.dones = np.logical_or(data_dict["terminals"], data_dict["timeouts"])
        self.size = len(self.dones)

        assert len(self.observations) == len(self.next_observations)


    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
        }

    def __len__(self):
        return self.size
    

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    #env = config["make_env"]() # gym.make(config["env_name"]) <-- TODO Hier ändern
    #discrete = isinstance(env.action_space, gym.spaces.Discrete) <-- entweder Action Space oder Agents anpassen

    #assert discrete, "DQN only supports discrete action spaces"

    #agent_cls = agents[config["agent"]]
    # cql_agent = CQLAgent
    # agent = agent_cls(
    #     env.observation_space.shape,
    #     env.action_space.n,
    #     **config["agent_kwargs"],
    # )
    agent = CQLAgent(OBSERVATION_SHAPE, NUM_ACTIONS, **config["agent_kwargs"])

    #ep_len = env.spec.max_episode_steps or env.max_episode_steps

    # Open the HDF5 file
    with h5py.File(os.path.join(args.dataset_dir, f"{config['dataset_name']}.h5"), 'r') as hdf:
        # Create an empty dictionary
        retrieved_dict = {}
        # Iterate over datasets in the HDF5 file and add them to the dictionary
        for key in hdf.keys():
            retrieved_dict[key] = hdf[key][()]
    
    dataset = ReplayBuffer(retrieved_dict)
    dataset.actions = np.random.randint(0, NUM_ACTIONS, size=(dataset.size,))
    #with open(os.path.join(args.dataset_dir, f"{config['dataset_name']}."), "rb") as f:

    print(config["batch_size"])
    for step in tqdm.trange(config["training_steps"], dynamic_ncols=True):
        # Train with offline RL
        batch = dataset.sample(config["batch_size"]) # <-- sampling process anpassen!

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            for k, v in metrics.items():
                logger.log_scalar(v, k, step)
        
        # if step % args.eval_interval == 0:
        #     # Evaluate
        #     trajectories = utils.sample_n_trajectories(
        #         env,
        #         agent,
        #         args.num_eval_trajectories,
        #         ep_len,
        #     )
        #     returns = [t["episode_statistics"]["r"] for t in trajectories]
        #     ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

        #     logger.log_scalar(np.mean(returns), "eval_return", step)
        #     logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

        #     if len(returns) > 1:
        #         logger.log_scalar(np.std(returns), "eval/return_std", step)
        #         logger.log_scalar(np.max(returns), "eval/return_max", step)
        #         logger.log_scalar(np.min(returns), "eval/return_min", step)
        #         logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
        #         logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
        #         logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

        #     env_pointmass: Pointmass = env.unwrapped
        #     logger.log_figures(
        #         [env_pointmass.plot_trajectory(trajectory["next_observation"]) for trajectory in trajectories],
        #         "trajectories",
        #         step,
        #         "eval"
        #     )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str,
                        default="experiments/offline/pointmass_easy_cql.yaml")#required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    # new -------------------------------------------------------------------------------
    parser.add_argument("--observation_shape", type=int, nargs="+", default=[2])
    parser.add_argument("--num_actions", type=int, default=4) # TODO: Hier anpassen!
    # -----------------------------------------------------------------------------------

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--dataset_dir", type=str, default="datasets/", )#required=True)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "final_project_cql_"  # keep for autograder

    config = make_config(args.config_file) # debunk this line

    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
