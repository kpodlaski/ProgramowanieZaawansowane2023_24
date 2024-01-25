import numpy as np
import random

init_state  = [[0,0,1],
               [1,0,0],
               [0,1,0],
               [1,0,0],
               [0,0,0],
               [0, 1, 0],
               [0, 1, 0],
               [0, 1, 0],
               [1, 0, 0],
               ]
class Game():

    def __init__(self):
        self.state = np.array(init_state)
        self.position = 1
        self.cols = self.state.shape[1]
        self.score = 0

    def action(self, action):
        last_row = self.state[-1,:]
        if last_row[self.position]>0:
            return self.score;
        self.position = (self.position + action)%self.cols
        new_row = np.zeros((1,self.cols))
        new_barier = random.randrange(0,self.cols)
        if new_barier>0:
            new_row[0,new_barier-1]=1
        self.state = np.vstack((new_row,self.state[:-1,:]))
        self.score+=1
        player_row = np.zeros((1,self.cols))
        player_row [0,self.position]=2
        return np.vstack((self.state,player_row))

    def reset(self):
        self.state = np.array(init_state)
        self.position = 1
        self.score = 0