import sys
from environment import MountainCar
import numpy as np
#import matplotlib.pyplot as plt

class Mountain():
  def __init__(self, mode, episodes, max_iterations, epsilon, gamma, learning_rate):
    self.environment = MountainCar(mode)
    self.n_states = self.environment.state_space
    self.n_actions = self.environment.action_space

    self.weights = np.zeros((self.n_states, self.n_actions)) #2 or 2048, 3
    self.bias = 0.0
    self.episodes = episodes
    self.max_iterations = max_iterations
    self.epsilon = epsilon
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.return_list = []

  def train(self):

    for i in range(self.episodes):
      state = self.environment.reset()
      flag = False
      count = 0
      reward = 0
      while (not flag) and count < self.max_iterations:
        #randomize
        a1 = np.random.choice(self.n_actions)
        if np.random.random() > self.epsilon:
          a1 = get_max(state, self.weights, self.bias, i, count, self.n_actions)

        q1 = get_q(state, a1, self.weights, i, count)
        q1 += self.bias
        
        next_state, ret, flag = self.environment.step(a1)

        reward += ret

        max_q = get_max(next_state, self.weights, self.bias, i, count, self.n_actions)
        #print(max_q)

        q2 = get_q(next_state, max_q, self.weights, i, count)
        q2 += self.bias

        grad = (q1 - (ret + q2 * self.gamma))
        #print(state)
        #print(state.items())
        for index in state.keys():
          self.weights[index, a1] = self.weights[index, a1] - (state[index] * self.learning_rate * grad)
        self.bias = self.bias - self.learning_rate * grad

        state = next_state
        count += 1

      self.return_list.append(reward)
      #print(self.return_list)
    return self.weights, self.bias, self.return_list

def write_output(weight_out, returns_out, bias, weights, reward_list):
  with open(weight_out, 'w') as f:
    f.write(str(bias) + '\n')
    for state in weights:
      for action in state:
        f.write(str(action) + '\n')
  print("Created weight.out")
  with open(returns_out, 'w') as f:
    for ret in reward_list:
      f.write(str(ret) + '\n')
  print("Created returns.out")

def get_q(state, action, weights, n, count):
  q_val = 0
  for index in state.keys():
    q_val += state[index] * weights[index, action]
  return q_val

def get_max(state, weights, bias, n, count, n_actions):
  q_val = [0]*n_actions
  for i in range(n_actions):
    temp_q = get_q(state, i, weights, n, count)
    temp_q += bias
    q_val[i] = temp_q
  max_val = max(q_val)
  max_index = q_val.index(max_val)
  #print(q_val)
  return max_index

def main(args):
  print("Loading...\n")
  mode = args[1] # raw or tile, mode to run environment in
  weight_out = args[2] # output file, weights of linear model
  returns_out = args[3] # output file, returns of the agent
  episodes = args[4] # episodes program should train agent, each episode is a sequence of states, actions, rewards | ends if in terminal state or max episode length reached
  max_iterations = args[5] # terminates curr episode
  epsilon = args[6] # coefficient epsilon-greedy strat
  gamma = args[7] # coefficient discount factor
  learning_rate = args[8] # alpha of Q-learning algorithm
  
  mode = str(mode)
  episodes = int(episodes)
  max_iterations = int(max_iterations)
  epsilon = float(epsilon)
  gamma = float(gamma)
  learning_rate = float(learning_rate)

  #initialize class Mountain
  mountain = Mountain(mode, episodes, max_iterations, epsilon, gamma, learning_rate)

  print("Training...\n")
  #train
  weights, bias, reward_list = mountain.train()

  print("Creating output files...\n")
  #write output files
  write_output(weight_out, returns_out, bias, weights, reward_list)

if __name__ == "__main__":
  main(sys.argv)
