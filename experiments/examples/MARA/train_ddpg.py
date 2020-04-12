
import multiprocessing
import tensorflow as tf
from  importlib import import_module

import os
import sys
import time
from datetime import datetime
import gym
import gym_gazebo2
import tensorflow as tf
import multiprocessing

from importlib import import_module
from baselines import bench, logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


ncpu = multiprocessing.cpu_count()

config = tf.ConfigProto(allow_soft_placement = True,
                        intra_op_parallelism_threads= ncpu,
                        inter_op_parallelism_threads = ncpu,
                        log_device_placement=False)

config.gpu_options.allow_growth = True

tf.Session(config=config).__enter__()

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # try baselines package first
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # else from rl_alg
        alg_module = import_module('.'.join(['rl_algs', alg, submodule]))

    return alg_module

def get_learn_function(alg, submodule=None):
    # return learn function as object
    return get_alg_module(alg, submodule).learn

# def get_learn_function_defaults(alg, env_type):
#     try:
#         alg_defaults = get_alg_module(alg, 'defaults')
#         kwargs = getattr(alg_defaults, env_type)()
#         except(ImportError, AttributeError):
#             kwargs={}
#         return kwargs

def make_env():
    env = gym.make(alg_kwargs['env_name'])
    env.set_episode_size(alg_kwargs['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env


#init dictionary, but ddpg has no defaults like ppo2
env_type = 'mara_ddpg'
alg_kwargs={}
alg_kwargs['env_name']='MARA-v0' 
alg_kwargs['nsteps']= 1024
alg_kwargs['transfer_path']=None
alg_kwargs['network']='mlp'

#create folders
timedate = datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
logdir = '/tmp/ros2learn/'+alg_kwargs['env_name']+'/ddpg/'+timedate

# Generate tensorboard file
format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
logger.configure(os.path.abspath(logdir), format_strs)

with open(logger.get_dir()+ "/parameters.txt", 'w') as out:
    out.write(
        'nsteps = '+str(alg_kwargs['nsteps'])+'\n'
        + 'network = '+str(alg_kwargs['network'])+'\n'
        )
    

env=DummyVecEnv([make_env])

learn = get_learn_function('ddpg')
transfer_path = alg_kwargs['transfer_path']

# Remove unused parameters for training
alg_kwargs.pop('env_name')
alg_kwargs.pop('nsteps')
alg_kwargs.pop('transfer_path')



if transfer_path is not None:
    # Do transfer learning
    _ = learn(env=env, load_path = transfer_path, **alg_kwargs)
else:
    _ = learn(env=env, **alg_kwargs)

env.dummy().gg2().close()
os.kill(os.getpid(), 9)
