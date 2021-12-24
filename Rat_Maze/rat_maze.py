"""
Name:  Hoai Cao
Email: hoai.cao64@myhunter.cuny.edu
Resources:  https://www.kaggle.com/alexisbcook/one-step-lookahead
            https://www.samyzaf.com/ML/rl/qmaze.html
            https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pickle
import cv2
from PIL import Image, ImageEnhance
from matplotlib import style

style.use("ggplot")


#initializing the size of maze
SIZE = 10 # a 5x5 maze

NUM_EPISODES = 15000
MOVE_PENALTY = -1
WALL_PENALTY = -9999
REVISIT_PENALTY = -100
#CAUGHT_PENALTY = -9999
CHEESE_REWARD = 100
epsilon = 0.9
EPS_DECAY = 0.999
SHOW_EVERY = 1000

#WALL_LOCATION = [(0,1),(3,0),(3,1),(3,3),(3,4)]
#WALL_LOCATION = [(0,1),(1,3),(1,4),(1,6),(2,0),(2,1),(2,2),(2,6),(3,4),(3,5),(4,1),(4,2),(4,3),(5,1),(6,3)]
WALL_LOCATION = [(0,1),(1,5),(2,5),(3,0),(3,1),(3,3),(3,4),(3,6),(4,2),(4,4),(4,6),(4,7),(4,8),(5,2),(5,4),\
                (7,6),(7,7),(7,8),(7,9),(8,1),(8,2),(8,3),(8,4),(8,5),(9,7)]

CHEESE_LOCATION = (SIZE-1,SIZE-1)

start_q_table = None #None or file name if having a pre-trained agent (existing q-table)

LEARNING_RATE = 0.1
DISCOUNT = 0.95

RAT_ = 1
CHESSE_ = 2
WALL_ = 3
CAT_ = 4
TRACE_ = 5

d = {1: (255,175,0),
     2: (0,255,0),
     3: (0,0,255),
     4: (255,255,255),
     5: (128,128,128)}


#creating a maze using numpy array
maze5 = np.array([
    [ 1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.],
    [ 0.,  0.,  1.,  0.,  0.],
    [ 1.,  1.,  1.,  1.,  1.]
])
maze7 =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

maze10 = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
    [ 1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
])

class Rat:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited_cells = []

    def __str__(self):
        return f"{self.x},{self.y}"

    def action(self, act):
        #act = 0,1,2 or 3
        if act == 0:
            self.move(x = 0,y = -1)
        elif act == 1:
            self.move(x = 1, y = 0)
        elif act == 2:
            self.move(x = 0, y = 1)
        elif act == 3:
            self.move(x = -1, y = 0)


    def move(self,x,y):
        self.x += x
        self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1
        self.visited_cells.append((self.x,self.y))

    def get_reward(self, x, y):
        if (x,y) == CHEESE_LOCATION:
            return CHEESE_REWARD
        elif (x,y) in WALL_LOCATION:
            return WALL_PENALTY
        elif (x,y) in self.visited_cells:
            return REVISIT_PENALTY
        else:
            return MOVE_PENALTY

    def reset(self):
        self.x = 0
        self.y = 0
        self.visited_cells = []
            
rat = Rat(0,0)
#cat = Rat(SIZE-1, 0)

#print(rat)
#rat.action(2)
#print(rat)

if start_q_table is None:
    q_table = {}
    for x in range(1-SIZE, SIZE):
        for y in range(1-SIZE, SIZE):
            q_table[(x,y)] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

#print(q_table[(3, 4)])

episode_rewards = []

for ep in range(NUM_EPISODES):
    rat = Rat(0,0)
    
    if ep % SHOW_EVERY == 0:
        print(f"on #{ep}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        one_step_obs = (rat.x, rat.y)
        if np.random.random() > epsilon:
            #one-step lookahead:
            action = np.argmax(q_table[one_step_obs])
        
        else:
            action = np.random.randint(0,4)

        rat.action(action)
        reward = rat.get_reward(rat.x, rat.y)

        new_obs = (rat.x, rat.y)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[one_step_obs][action]

        if reward == CHEESE_REWARD:
            new_q = CHEESE_REWARD
            #isFinished
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[one_step_obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[rat.x][rat.y] = d[RAT_]  # sets the rat location tile to blue color
            env[CHEESE_LOCATION[0]][CHEESE_LOCATION[1]] = d[CHESSE_]  # sets the cheese tile to green
            for wall in WALL_LOCATION:                  #set wall to red
                env[wall[0]][wall[1]] = d[WALL_]

            
            img = Image.fromarray(env, 'RGB')  # reading to rgb.
            img = img.resize((300, 300), resample=Image.NEAREST)  # resizing
            cv2.imshow("image", np.array(img))  # show it!
            if reward == CHEESE_REWARD: #or reward == WALL_PENALTY or reward == REVISIT_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == CHEESE_REWARD:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)


