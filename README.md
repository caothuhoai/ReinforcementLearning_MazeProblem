# ReinforcementLearning_MazeProblem
Using RL to solve a Maze and more than that.

### 1. Rat and Cheese Maze:
- Agent Rat learns to solve the maze to get his cheese in the shortest path possble to save his energy. He learns from practice, getting rewards/punishments so he knows what's best to do.
    
    - Agent Rat learns from his mistakes. He learns that he doesn't want to go through red cells (walls) because it hurts him badly. He also doesn't want to wander around too much, because he's losing energy doing so.

    <img src="Rat_Maze/images/learning_rat.gif" width="550" height="300" />

    - After quite a few times studying the Maze (about 10000 episodes), agent Rat finds out the best way to get his reward.

    <img src="Rat_Maze/images/trained_rat.gif" width="550" height="300" />

### 2. Rat, Cat and Cheese Maze:
- Our agent Rat now has an enermy who wants to catch him as he searches for the cheese in the Maze. As agent Rat wants the cheese, the Cat will look for the agent. Agent Rat will learn that as much as he wants the cheese, he needs to avoid the Cat at all cost (or in other words, the penalty of getting caught is greater than the delayed reward).
- There are 3 cases that can happen here:
    - Either agent Rat safely finds the cheese:

    <img src="Rat_Cat_Maze/images/getCheese.gif" width="550" height="300" />

    - Or the Cat catches agent Rat before that:

    <img src="Rat_Cat_Maze/images/getCaught.gif" width="550" height="300" />

    - Or none of them get anything (a tie game):

    <img src="Rat_Cat_Maze/images/tieGame.gif" width="550" height="300" />








Resources:
 - https://www.kaggle.com/alexisbcook/one-step-lookahead
 - https://www.samyzaf.com/ML/rl/qmaze.html
 - https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/

## License

    Copyright [2022] [Hoai Cao]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
