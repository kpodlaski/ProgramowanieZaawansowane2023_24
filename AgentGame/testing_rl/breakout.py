import gymnasium as gym
import math
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#env = gym.make("BreakoutDeterministic-v4", render_mode='human')



def screen_and_convolution():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    print("Obs space ", n_observations)
    print("state space ", state.shape)
    # Initialize the environment and get its state
    init_state, info = env.reset()
    env.render()
    img = np.zeros((100,144))
    for t in count():
        action = random.randrange(n_actions)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # we cut all unimportant screen elements, blocks in first level are above 93 row.
        # we stuck with increasing importance
        img = img/2+observation[93:193,8:-8,0]
        if done:
            print("Duration:",t+1)
            return
        if t>0 and t%4==0:
            # no stacking
            # img = observation[93:193,8:-8,0]
            img = img
            input_tensor = torch.tensor(img.reshape(1,1,100,144), dtype=torch.float32)
            pool = nn.MaxPool2d(kernel_size=(3,4))
            output = pool(input_tensor)
            plt.imshow(img)
            plt.title("orig frames:{}-{}".format(t-4,t))
            plt.show()
            print("Orig_shape",img.shape)
            plt.imshow(output.squeeze())
            plt.title("smaller frames:{}-{}".format(t-4,t))
            plt.show()
            print("Smaller shape:",output.shape)
            #np.savetxt("../out_files/img.csv", img, delimiter=' ', header='', fmt="%.0f")
            img = np.zeros((100, 144))
        if t>12:
            break

def screen_and_histograms():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    print("Obs space ", n_observations)
    print("state space ", state.shape)
    # Initialize the environment and get its state
    init_state, info = env.reset()
    env.render()
    img = np.zeros((100,144))
    for t in count():
        action = random.randrange(n_actions)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # we cut all unimportant screen elements, blocks in first level are above 93 row.
        # we stuck with increasing importance
        # 93:189 -- image above palette
        # 189:193 -- palette
        img = img/2+observation[93:193,8:-8,0]
        if done:
            print("Duration:",t+1)
            return
        if t>0 and t%4==0:
            # no stacking
            ball_x = img[:-4, :].sum(axis=0)/255/5
            ball_y = img[:-4, :].sum(axis=1)/255/5
            palette= img[-4:, :].sum(axis=0)/255/6
            network_input = np.hstack((ball_x,ball_y,palette))
            print("ball_x", ball_x.shape)
            print("ball_y shape:", ball_y.shape)
            print("palette shape:", palette.shape)
            print("net_input shape:", network_input.shape)
            print("img shape:",img.shape)
            np.savetxt("../out_files/img.csv", img, delimiter=' ', header='', fmt="%.0f")
            np.savetxt("../out_files/ball_x.csv", ball_x.reshape(1,-1), delimiter=' ', header='', fmt="%.2f")
            np.savetxt("../out_files/ball_y.csv", ball_y.reshape(-1,1), delimiter=' ', header='', fmt="%.2f")
            np.savetxt("../out_files/palette.csv", palette.reshape(1,-1), delimiter=' ', header='', fmt="%.2f")

            #np.savetxt("img.csv", img, delimiter=' ', header='', fmt="%.0f")
            img = np.zeros((100, 144))
        if t>12:
            break

def rom_data():
    env = gym.make("Breakout-ramNoFrameskip-v4", render_mode='human')
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    print("Obs space ", n_observations)
    print("state space ", state.shape)
    # Initialize the environment and get its state
    init_state, info = env.reset()
    env.render()
    avg = np.zeros(state.shape)
    obs = []
    action = 1
    for t in count():
        observation, reward, terminated, truncated, _ = env.step(action)
        action =0
        done = terminated or truncated
        # we cut all unimportant screen elements, blocks in first level are above 93 row.
        # we stuck with increasing importance
        avg = avg+observation
        obs.append(observation)
        if done:
            print("Duration:",t+1)
            return
        if t>0 and t%4==0:
            avg =avg/4
            for o in obs:
                print("o")
                print(o)
                print("o-avg")
                print(o-avg)
                print("--------------")
                plt.imshow((o-avg).reshape(8,-1))
                plt.show()
            avg = np.zeros(avg.shape)
            obs = []
        if t>12:
            break

print('Complete')
#screen_and_convolution()
screen_and_histograms()
#rom_data()