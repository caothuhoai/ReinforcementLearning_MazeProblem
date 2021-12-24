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
from PIL import Image
from matplotlib import style

style.use("ggplot")


#initializing the size of maze
SIZE = 10 # a 5x5 maze

NUM_EPISODES = 50000
MOVE_PENALTY = -1
WALL_PENALTY = -9999
REVISIT_PENALTY = -200
CAUGHT_PENALTY = -3000
CHEESE_REWARD = 5000
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 5000

#WALL_LOCATION = [(0,1),(3,0),(3,1),(3,3),(3,4)]
#WALL_LOCATION = [(0,1),(1,3),(1,4),(1,6),(2,0),(2,1),(2,2),(2,6),(3,4),(3,5),(4,1),(4,2),(4,3),(5,1),(6,3)]
#WALL_LOCATION = [(0,1),(1,5),(2,5),(3,0),(3,1),(3,3),(3,4),(3,6),(4,2),(4,4),(4,6),(4,7),(4,8),(5,2),(5,4),\
#                (7,6),(7,7),(7,8),(7,9),(8,1),(8,2),(8,3),(8,4),(8,5),(9,7)]

#wall locations with cat:
WALL_LOCATION = [(1,1),(1,2),(1,4),(1,6),(1,8),(3,0),(3,2),(3,5),(3,6),(3,8),(3,9),\
                 (5,1),(5,3),(5,5),(5,7),(5,9),(7,3),(3,4),(7,5),(7,8),(9,3),(9,5),(9,6)]

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

#create Cat() class that chases the Rat(), use the __sub__ operator to get the relative position of the rat, which is the obs of Cat()

class Rat:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        #elf.target = target
        self.visited_cells = []

    def __str__(self):
        return f"{self.x},{self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

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


    def move(self,x=False,y=False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1,2)
        else:
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
cat = Rat(SIZE-1, 0)

#print(rat)
#rat.action(2)
#print(rat)

if start_q_table is None:
    q_table = {}
    for x1 in range(1-SIZE, SIZE):
        for y1 in range(1-SIZE, SIZE):
            for x2 in range(1-SIZE, SIZE):
                for y2 in range(1-SIZE, SIZE):
                    q_table[((x1,y1),(x2,y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

#print(q_table[(3, 4)])

episode_rewards = []

Ntimes_caught = 0
Ntimes_get_cheese = 0

for ep in range(NUM_EPISODES):
    rat = Rat(0,0)
    cat = Rat(SIZE-1, 0)
    cheese = Rat(SIZE-1,SIZE-1,)
    
    if ep % SHOW_EVERY == 0:
        print(f"on #{ep}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        print(f"#times getting caught: {Ntimes_caught}. #times getting cheese: {Ntimes_get_cheese}")
        #print(len(rat.visited_cells))
        #print(list(set(rat.visited_cells)))
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (rat - cheese, rat - cat)
        #two_step_obs = [(rat.x-1,rat.y),(rat.x+1, rat.y),(rat.x, rat.y-1),(rat.x, rat.y+1)]
        if np.random.random() > epsilon:
            #one-step lookahead:
            action = np.argmax(q_table[obs])

            #2-step-lookhead:
            #rewards = dict(zip(np.argmax(np.argmax(q_table[one_step_obs]), [rat.get_reward(obs[0],obs[1]) for obs in two_step_obs])))
            #max_reward = [key for key in rewards.keys() if rewards[key] == max(rewards.values())]
            #action = random.choice(max_reward)

            #two_step_actions = []
            #max_action = 0
            #for obs in two_step_obs:

        
                #two_step_actions.append(q_table[obs])
            #    m_action = np.argmax(q_table[obs])
            #    if m_action > max_action:
            #        max_action = m_action
        

        else:
            action = np.random.randint(0,4)

        rat.action(action)
        cat.move()

        if rat.x == cat.x and rat.y == cat.y:
            reward = CAUGHT_PENALTY
        else:
            reward = rat.get_reward(rat.x, rat.y)

        new_obs = (rat - cheese, rat - cat)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == CHEESE_REWARD:
            new_q = CHEESE_REWARD
            #isFinished
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            #print(len(rat.visited_cells))
            #print(list(set(rat.visited_cells)))
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[rat.x][rat.y] = d[RAT_]  # sets the rat location tile to blue color
            env[cheese.x][cheese.y] = d[CHESSE_]  # sets the cheese tile to green
            env[cat.x][cat.y] = d[CAT_]
            for wall in WALL_LOCATION:                  #set wall to red
                env[wall[0]][wall[1]] = d[WALL_]

            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently
            img = img.resize((400, 400), resample=Image.NEAREST)  # resizing
            cv2.imshow("image", np.array(img))
            if reward == CHEESE_REWARD or reward == CAUGHT_PENALTY:
                if cv2.waitKey(300) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == CHEESE_REWARD:
            Ntimes_get_cheese += 1
            break
        elif reward == CAUGHT_PENALTY:
            Ntimes_caught += 1
            break
    #rat.reset()

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    #without EPS_DECAY, the rewards don't increase very much later in the training.
    #with EPS_DECAY, the rewards increase steadily toward the end of training.
    #however, the number of times getting the cheese is going down to 0 at the end.
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)


