# Report: Project 2: Continous Control"

We will train a DeepRL agent to solve a Unity Environment.

## Unity Environment

+ Set-up: Double-jointed arm which can move to target locations.
+ Goal: The agents must move it's hand to the goal location, and keep it there.
+ Agents: The environment contains 20 agents linked to a single Brain.
+ Agent Reward Function (independent):
  + 0.1 Each step agent's hand is in goal location.
+ Brains: One Brain with the following observation/action space.
  + Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  + Vector Action space: 
      + Size of 4, corresponding to torque applicable to two joints.
      + Continuous: Every entry in the action vector should be a number between -1 and 1.
+ Reset Parameters: Two, corresponding to goal size, and goal movement speed.
+ Benchmark Mean Reward: 30

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.


~~~~
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
          goal_speed -> 1.0
          goal_size -> 5.0
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
~~~~

~~~~
Number of agents: 20
Size of each action: 4
There are 20 agents. Each observes a state with length: 33
The state for the first agent looks like: 
[ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
~~~~

## Learning Algorithm

I implement an artificial agent, termed "Deep Deterministic Policy Gradient"(DDPG)

DDPG is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

+ DDPG is an off-policy algorithm.
+ DDPG can only be used for environments with continuous action spaces.
+ DDPG can be thought of as being deep Q-learning for continuous action spaces.
+ DDPG can be implemented with parallelization

DDPG is closely connected to Q-learning algorithms, and it concurrently learns a Q-function and a policy which are updated to improve each other.

Algorithms like DDPG and Q-Learning are off-policy, so they are able to reuse old data very efficiently. They gain this benefit by exploiting Bellman’s equations for optimality, which a Q-function can be trained to satisfy using any environment interaction data (as long as there’s enough experience from the high-reward areas in the environment).


### DDPG Pseudocode
![dpg-pseudocode](./img/ddpg-pseudocode.png "dpg-pseudocode")
(Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html#the-policy-learning-side-of-ddpg)

### Hyper Parameters
#### DDPG Parameters

+ BUFFER_SIZE = int(1e6)        # replay buffer size
+ BATCH_SIZE = 1024             # minibatch size
+ GAMMA = 0.99                  # discount factor
+ TAU = 1e-3                    # for soft update of target parameters
+ LR_ACTOR = 1e-3               # learning rate of the actor 
+ LR_CRITIC = 1e-3              # learning rate of the critic before: 3e-4
+ WEIGHT_DECAY = 0.0000         # L2 weight decay
+ EPSILON = 1.0                 # noise factor
+ EPSILON_DECAY = 1e-6          # decay of noise factor

#### Neural Network Model Architecture & Parameters
For this project we use these models:

~~~~
Actor Model:
  (bn0): BatchNorm1d(33, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)

Critic Model:
  (bn0): BatchNorm1d(33, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fcs1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
~~~~

### Training

#### Plot of Rewards

Environment solved in 135 episodes. Average Score: 30.18. 
A plot of rewards per episode is included to illustrate that:

+ the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.

![rewards.png](./img/rewards.png "rewards.PNG")


### Ideas for Future Work

Future ideas for improving the agent's performance.

+ Try new algorithm such as [Hierarchical Actor Critic (HAC)](https://arxiv.org/abs/1712.00948.pdf). HAC enables agents to learn to break down problems involving continuous action spaces into simpler subproblems belonging to differenttime scales. The ability to learn at different resolutions in time may help overcome one of the main challenges in deep reinforcement learning — sample efficiency.
+ Try new algorithm such as [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495.pdf)  
    + [Video HER: Vanilla DDPG vs DDPG video](https://www.youtube.com/watch?time_continue=130&v=Dz_HuzgMxzo )
