""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""
from sensor_msgs.msg import CameraInfo, Image
import gym
import gym_gazebo2
import time
env = gym.make('MARACamera-v0')
#env = gym.make('MARAOrient-v0')
#env = gym.make('MARACollision-v0')
#env = gym.make('MARACollisionOrient-v0')
#env = gym.make('MARACollisionOrientRandomTarget-v0')

while True:
    # take a random action
    observation, reward, done, info = env.step(env.action_space.sample())

