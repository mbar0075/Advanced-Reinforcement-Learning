# Advanced Reinforcement Learning

<p align="justify">

This repository contains all the code and findings from the research study `"Solving Control Problems using Reinforcement Learning"`.

## Introduction

Reinforcement Learning (RL) involves learning how to translate events into actions to maximize a numerical reward signal. RL doesn't dictate actions but relies on trial-and-error to determine which actions yield the highest rewards. It focuses on long-term reward maximization rather than immediate gains, operating within an interaction loop where agents navigate environments based on policies, transitioning states, collecting rewards, and making observations. RL models are often expressed as Markov Decision Processes (MDPs), providing a framework for sequential decision-making problems, and Deep Reinforcement Learning (DRL) integrates deep learning techniques to handle complex environments. Differentiating from traditional machine learning, RL's goal-oriented approach involves agents gaining experience over time by interacting with environments and receiving feedback. Various RL models, such as Value-Based, Policy-Based, and Actor-Critic, offer different decision-making paradigms. This overview draws insights from research in solving control problems through RL methodologies.

## Background

### Value Based Methods

Value-based methods in Reinforcement Learning (RL) assess state or state-action values using functions like *`V (s)`* or *`Q (s, a)`* to maximize expected returns. These methods leverage algorithms like Q-learning to converge towards optimal value functions.

#### Value Iteration Algorithm

Value iteration utilizes the Bellman optimality equation to update state values, leading to a deterministic policy.

#### Q-Learning Algorithm

Q-learning approximates action-value functions by iteratively updating values based on environment interactions.

### Deep Q-Learning Methods

Deep Q Network (DQN) extends Q-learning using deep neural networks to handle larger state spaces. It utilizes loss functions and network updates for learning.

#### Double Deep Q Network (DDQN)

DDQN mitigates overestimation bias by employing two separate networks for action selection and value estimation.

#### Dueling Deep Q Network (Dueling-DQN)

Dueling-DQN separates state value estimation and action advantages for efficient learning.

### Policy Based Methods

Policy-based methods directly compute action probabilities from states, focusing on learning parameterized policies to maximize returns.

### Actor-Critic Methods

Actor-critic combines value-based and policy-based methods, using an actor for action selection and a critic for action evaluation.

#### Proximal Policy Optimisation (PPO)

PPO optimizes policy within a trust region, ensuring stability while updating the policy.

#### Deep Deterministic Policy Gradient (DDPG)

DDPG is an actor-critic algorithm designed for continuous action spaces, utilizing neural networks for policy and value estimation.

## Methodology
This research study examined the effectiveness of the algorithms in approximating the `Lunar Lander problem` through a sequence of experiments. The problem involves optimizing rocket trajectory in a `pre-built gymnasium environment`, following Pontryagin’s maximum principle (engines should fire at full throttle). The goal is to maneuver the Lunar Lander between two flag poles, ensuring both side rockets make ground contact. The algorithms were rigorously tested until meeting the predefined success criteria of reaching an average reward of `195`.

<p align='center'><img src="Assets\LunarLander.png" alt="Lunar Lander problem"/></p>

The observation space for this problem is an 8-dimensional vector:
1. `X position`
2. `Y position`
3. `X velocity`
4. `Y velocity`
5. `Angle`
6. `Angular velocity`
7. `Left leg touching the ground`
8. `Right leg touching the ground`

At each step, RL agents received rewards based on proximity to the landing pad, lander velocities, and alignment. The episode ended with significant reward adjustments: crashes incurred a -100 penalty, while safe landings earned +100 points. A crash or inactivity terminated the episode.

### Experiment 1
Experiment 1 tested agents in a `discrete action space`:
1. `Do nothing`
2. `Fire left orientation engine`
3. `Fire main engine`
4. `Fire right orientation engine`

Additionally, the algorithm parameters for this experiment were:

<p align="center">

| `Alpha` | `Gamma` | `Loss`     | `Batch` | `Optimizer` |
|-------|-------|----------|-------|-----------|
| 0.001 | 0.99  | SmoothL1 | 128   | Adam      |

</p>

#### Standard / Normal DQN
Standard DQN comprised a 64-neuron linear layer followed by a Tanh activation function and an output layer matching the required number of output actions. 

#### Double DQN
This architecture separated action selection from value estimation using two identical networks to the Standard DQN.

#### Dueling DQN
Refined the Standard DQN by separating value estimation and advantage functions for efficient learning.

#### Double Dueling DQN
Combined the enhancements of Double DQN and Dueling DQN.

### Experiment 2
Experiment 2 tested agents in a `continuous action space`:
1. `Thrust`: (-1.0, 1.0) for the main engine
2. `Rotation`: (-1.0, 1.0) for the Lunar Lander

#### PPO with GAE
Utilized two network policies—an actor and a critic—and employed GAE for refined decision making. Algorithm parameters:

<p align="center">

| `Model`   | `Alpha`  | `Gamma` | `Loss`     | `Batch` | `Optimizer` |
|---------|--------|-------|----------|-------|-----------|
| PPO+GAE | 0.0003 | 0.999 | SmoothL1 | 64    | AdamW     |

</p>

#### DDPG
Constructed with actor and critic networks and employed Ornstein-Uhlenbeck action noise generation. Algorithm parameters:

<p align="center">

| `Model` | `Alpha`  | `Gamma` | `Loss`     | `Batch` | `Optimizer` |
|-------|--------|-------|----------|-------|-----------|
| DDPG  | 0.0003 | 0.99  | SmoothL1 | 64    | AdamW     |

</p>

## Testing
Below are the videos of the trained agents from Experiment 1 and Experiment 2 in the Lunar Lander environment for a single episode.

### Experiment 1
#### Standard / Normal DQN
  
https://github.com/mbar0075/Advanced-Reinforcement-Learning/assets/103250564/476b6a10-c9ad-4bb2-8074-e915e6856af1

#### Double DQN


https://github.com/mbar0075/Advanced-Reinforcement-Learning/assets/103250564/69c915be-c58c-48f5-97a4-19a429282ff2


#### Dueling DQN


https://github.com/mbar0075/Advanced-Reinforcement-Learning/assets/103250564/74e6db92-6014-403c-bf48-149b0e5f33e0


#### Double Dueling DQN


https://github.com/mbar0075/Advanced-Reinforcement-Learning/assets/103250564/883f0e30-96f6-4a3f-97e3-45d5018bf58e



### Experiment 2
#### PPO with GAE


https://github.com/mbar0075/Advanced-Reinforcement-Learning/assets/103250564/123dadff-66e8-4c11-b330-13f7644ce3c5


#### DDPG


https://github.com/mbar0075/Advanced-Reinforcement-Learning/assets/103250564/120891b0-d2f1-4cc6-b39b-3e7e1259357a



## Results & Discussion

### Experiment 1

The analysis of Experiment 1 revealed the performance of various RL algorithms in solving the Lunar Lander problem. Dueling DQN emerged as the top-performing algorithm, showcasing the fastest convergence among the experimented variants. Double Dueling DQN closely followed, outperforming Double DQN and Standard DQN.

<table>
  <tr>
    <td align="center">
      <img src="Code\dqn_results\avg_rewards.png" alt="Average Rewards"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Code\dqn_results\episode_lengths.png" alt="Cumulative Episode Lengths" width="100%" height="auto" />
    </td>
  </tr>
</table>

Detailed examination showcased that Double Dueling DQN achieved the highest maximum episode average rewards, closely followed by Dueling DQN and Standard DQN, with Double DQN trailing. Additionally, there was a gradual increase in cumulative episode lengths per episode. These trends highlighted the efficacy of Dueling DQN and its variations in achieving faster convergence and higher rewards compared to other models.

<table>
  <tr>
    <td align="center">
      <img src="Code\dqn_results\max_episode_avg_rewards.png" alt="Max Average Rewards"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Code\dqn_results\training_time.png" alt="Training Time" width="100%" height="auto" />
    </td>
  </tr>
</table>

In terms of completion time, Double DQN exhibited the quickest completion, followed by Dueling, Double Dueling, and Standard DQN in ascending order. Overall, Dueling DQN emerged as the top-performing RL algorithm in the Lunar Lander environment, with Double Dueling DQN showing promising performance between Dueling and Double DQN. Notably, Double DQN performed notably better than Standard DQN, albeit with minor discrepancies.

### Experiment 2

The outcomes from Experiment 2 presented insights into the performance of PPO with GAE and DDPG algorithms. PPO with GAE utilized eight concurrent environments, exhibiting high performance across all environments. Notably, Environment 1 displayed the best average rewards per episode.

<p align='center'><img src="Code\actor_critic_results\average_rewards_per_episode_ppogae2.png" alt="Average Rewards per Environment"/></p>

Comparing the top-performing environment of PPO with GAE to the results of DDPG, it was observed that PPO with GAE converged faster, employing a stochastic policy, whereas DDPG, utilizing a deterministic policy, showed more stable updates despite a longer convergence time.

<table>
  <tr>
    <td align="center">
      <img src="Code\actor_critic_results\average_rewards_per_episode_ppo_ddpg.png" alt="Max Average Rewards - Actor Critic"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Code\actor_critic_results\cumulative_episode_lengths_ppo_ddpg.png" alt="Training Time Lengths - Actor Critic" width="100%" height="auto" />
    </td>
  </tr>
</table>

Additionally, the cumulative episode durations per episode showed distinct patterns between PPO with GAE and DDPG. PPO with GAE demonstrated a linear rise, while DDPG displayed relatively smaller changes, possibly due to computational techniques, policy differences, or the environment's nature.

<table>
  <tr>
    <td align="center">
      <img src="Code\actor_critic_results\max_episode_avg_rewards.png" alt="Max Average Rewards - Actor Critic"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Code\actor_critic_results\training_duration.png" alt="Training Time Duration" width="100%" height="auto" />
    </td>
  </tr>
</table>

In terms of maximum episode average rewards, PPO with GAE achieved higher rewards due to parallelization-based training, with DDPG following closely. Despite the longer training time for PPO with GAE in multiple environments, DDPG, trained in a single environment, achieved competitive performance.

## Conclusion

The experiments detailed in this research offered valuable insights into the nuances of reinforcement learning algorithms in the Lunar Lander environment. In Experiment 1, within a discrete action space, Dueling DQN emerged as the top-performing algorithm, showcasing relatively rapid convergence. It was followed by Double Dueling DQN, Double DQN, and Standard DQN. Throughout the evaluation, the modified architectures consistently outperformed the original Standard DQN architecture.

Moreover, all DQNs concluded their environments much faster than the actor-critic methods in Experiment 2. This was an expected outcome due to the complexity introduced by continuous action spaces, expanding the state-action dimension and exposing the agent to additional noise during training.

Experiment 2 revealed distinct dynamics between PPO with GAE and DDPG. PPO demonstrated quicker convergence and higher maximum rewards due to parallelization. However, it required a longer training period compared to DDPG, which was trained in a single environment. These observations highlight the trade-offs between different algorithms in terms of convergence speed, rewards, and training duration.

</p>