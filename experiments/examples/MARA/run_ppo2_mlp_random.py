import os
import sys
import time
import gym
import gym_gazebo2
import numpy as np
import multiprocessing
import tensorflow as tf
import write_csv as csv_file

from importlib import import_module
from baselines import bench, logger
from baselines.ppo2 import model as ppo2
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.policies import build_policy

ncpu = multiprocessing.cpu_count()

if sys.platform == 'darwin':
    ncpu //= 2

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)

config.gpu_options.allow_growth = True

tf.Session(config=config).__enter__()

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def constfn(val):
    def f(_):
        return val
    return f

def make_env():
    env = gym.make(defaults['env_name'])
    env.set_episode_size(defaults['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/ppo2/defaults
# defaults = get_learn_function_defaults('ppo2', 'mara_mlp')
defaults = dict(
        num_layers = 8,
        num_hidden = 256,
        layer_norm = False,
        # activation = tf.nn.relu,
        nsteps = 1024,
        nminibatches = 4, #batchsize = nevn * nsteps // nminibatches
        lam = 0.95,
        gamma = 0.99,
        noptepochs = 10,
        log_interval = 1,
        ent_coef = 0.0,
        lr = lambda f: 3e-3 * f,
        cliprange = 0.25,
        vf_coef = 1,
        max_grad_norm = 0.5,
        seed = 0,
        value_network = 'copy',
        network = 'mlp',
        total_timesteps = 1e8,
        save_interval = 10,
        # env_name = 'MARA-v0',
        # env_name = 'MARARandomTarget-v0',
        env_name = 'MARARandomTarget2DEnv-v0',
        #env_name = 'MARAReal-v0',
        #env_name = 'MARAOrient-v0',
        # env_name = 'MARACollision-v0',
        # env_name = 'MARACollisionOrient-v0',
        transfer_path = None,
        # transfer_path = '/tmp/ros2learn/MARA-v0/ppo2_mlp/2019-02-19_12h47min/checkpoints/best',
        trained_path = './models/MARARandomTarget2DEnv-v0/ppo2_mlp/2020-03-29_00h18min/checkpoints/best'
    )

# Create needed folders
try:
    logdir = defaults['trained_path'].split('checkpoints')[0] + 'results' + defaults['trained_path'].split('checkpoints')[1]
except:
    logdir = '/tmp/ros2learn/' + defaults['env_name'] + '/ppo2_mlp_results/'
finally:
    logger.configure( os.path.abspath(logdir) )
    csvdir = logdir + "/csv/"

csv_files = [csvdir + "det_obs.csv", csvdir + "det_acs.csv", csvdir + "det_rew.csv" ]
if not os.path.exists(csvdir):
    os.makedirs(csvdir)
else:
    for f in csv_files:
        if os.path.isfile(f):
            os.remove(f)

env = DummyVecEnv([make_env])

set_global_seeds(defaults['seed'])

if isinstance(defaults['lr'], float):
    defaults['lr'] = constfn(defaults['lr'])
else:
    assert callable(defaults['lr'])
if isinstance(defaults['cliprange'], float):
    defaults['cliprange'] = constfn(defaults['cliprange'])
else:
    assert callable(defaults['cliprange'])

alg_kwargs ={ 'num_layers': defaults['num_layers'], 'num_hidden': defaults['num_hidden'] }
policy = build_policy(env, defaults['network'], **alg_kwargs)

nenvs = env.num_envs
ob_space = env.observation_space
ac_space = env.action_space
nbatch = nenvs * defaults['nsteps']
nbatch_train = nbatch // defaults['nminibatches']

make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                                nbatch_train=nbatch_train,
                                nsteps=defaults['nsteps'], ent_coef=defaults['ent_coef'], vf_coef=defaults['vf_coef'],
                                max_grad_norm=defaults['max_grad_norm'])

model = make_model()

if defaults['trained_path'] is not None:
    model.load(defaults['trained_path'])

obs = env.reset()
loop = True
while loop:
    actions = model.step_deterministic(obs)[0]
    obs, reward, done, _  = env.step_runtime(actions)

    print("Reward: ", reward)
    print("ee_translation[x, y, z]: ", obs[0][6:9])
    print("ee_orientation[w, x, y, z]: ", obs[0][9:13])

    csv_file.write_obs(obs[0], csv_files[0], defaults['env_name'])
    csv_file.write_acs(actions[0], csv_files[1])
    csv_file.write_rew(reward, csv_files[2])

    # if np.allclose(obs[0][6:9], np.asarray([0., 0., 0.]), atol=0.005 ): # lock if less than 5mm error in each axis
    #     env.step_runtime(obs[0][:6])
    #     loop = False
