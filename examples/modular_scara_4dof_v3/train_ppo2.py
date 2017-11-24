import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from baselines import bench, logger

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
tf.Session(config=config).__enter__()

def make_env():
    env = gym.make('GazeboModularScara4DOF-v3')
    env.render()
    print(logger.get_dir())
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    return env
env = DummyVecEnv([make_env])
env = VecNormalize(env)

initial_observation = env.reset()
print("Initial observation: ", initial_observation)
# env.render()
seed = 0
set_global_seeds(seed)
policy = MlpPolicy
ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
    lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
    ent_coef=0.0,
    lr=3e-4,
    cliprange=0.2,
    total_timesteps=1e6, save_interval=1)

# def policy_fn(name, ob_space, ac_space):
#     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
#         hid_size=64, num_hid_layers=2)
#
# pposgd_simple.learn(env, policy_fn,
#                     max_timesteps=1e6,
#                     timesteps_per_actorbatch=2048,
#                     clip_param=0.2, entcoeff=0.0,
#                     optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
#                     optim_batchsize=64, lam=0.95, schedule='linear', save_model_with_prefix='4dof_ppo1_test_H')
