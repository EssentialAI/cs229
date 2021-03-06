<!-- #region -->
<span class = 'nital'>You can navigate through this page quickly by using the navigation list on the right-hand side of this page.</span>

### [1.0] Introduction

Let's say you want to build a robot to pickup empty cans in a room. If you follow a rule-based approach, then the robot might follow the following steps:
1. Localize itself with the room by building a map of the entire room.
2. Run a local object detection algorithm to identify empty cans inside the room.
3. Then a cloud connection...

You see that such a simple solution becomes complex very easily. 
There is another method to solve this problem. Reinforcement Learning.

Using Reinforcement Learning, the agent (robot) can learn to collect as many cans as possible through trail and error from scratch. <span class = 'hital'>For the agent to learn progressively, we can use "the number of empty cans collected" as the reward.</span> This approach (Reinforcement Learning) would enable the agent to dynamically adapt to changes in the room such as moving furniture around, changing the color of the cans etc., because the agent can simply learn to maximize the reward through trail and error.
```{admonition} Note
:class: idea
Hold on a second! Before we continue any further, one  must understand what an Agent, Environment, Reward are. The above example explains the robust, adaptive nature of RL. This course teaches Reinforcement Learning from basics. Let's get started.
```

#### [1.1] What is Reinforcement Learning?

<span class = 'hital'>Reinforcement Learning is a field that deals with building models that mimic human behaviour to learn.</span> To completely understand how Reinforcement Learning (RL) models work, let's try and understand how humans learn.

Humans learn in a "cause and effect" fashion. For example, we humans do not know how to code right from birth. Learning to code is an interactive process. We make errors in the code, debug them, progressively gain experience about the consequences of actions and what actions to perform to achieve maximum success. <span class = 'nital'>Learning from the interaction is a foundational idea underlying nearly all theories of learning and intelligence.</span>

```{admonition} What is Reinforcement Learning?
:class: tip

Reinforcement Learning is a field that aims to build models that try to achieve 'goal-directed learning from interaction.' <span class = 'hital'>Reinforcement learning (RL) tells you how to make the best decisions, sequentially, within a context, to maximize a real-life measure of success.</span>

Learning by 'reinforcement' combines two tasks. The first is exploring new situations. The second is using that experience to make better decisions.
```
```{admonition} Note
:class: idea

The characteristics of 'Trial-and-Error based search' and 'Delayed Reward' are the two most important distinguishing features of Reinforcement Learning.
```
<span class = 'nital'>Differences between Machine Learning and Reinforcement Learning are discussed in detail below.</span>

#### [1.2] Why Reinforcement Learning?

One key feature of reinforcement learning is that it explicitly considers the *whole* problem of a goal-directed agent interacting with an uncertain environment. This is in contrast to many approaches that consider subproblems without addressing how they might fit into a larger picture.

Reinforcement learning is part of a decades-long trend within artificial intelligence and machine learning toward greater integration with statistics, optimization, and other mathematical subjects.

For example, the ability of some reinforcement learning methods to learn with parameterized approximators addresses the classical ???curse of dimensionality??? in operations research and control theory. More distinctively, reinforcement learning has also interacted strongly with psychology and neuroscience, with substantial benefits going both ways. Of all the forms of machine learning, reinforcement learning is the closest to the kind of learning that humans and other animals do, and many of the core algorithms of reinforcement learning were originally inspired by biological learning systems.


#### [1.3] Machine Learning
```{admonition} Note
:class: idea

[Source](https://rl-book.com/) I consider ML a child of data science, which is an overarching scientific field that investigates the data generated by phenomena. I dislike the term artificial intelligence (AI) for a similar reason; it is hard enough to
define what intelligence is, let alone specify how it is achieved.
```
<span class = 'hital'>Supervised Machine Learning.</span>
In this type of Machine Learning, models learn to generalize from numerous training images provided with labels. In Supervised Machine Learning, the data is split into training and test datasets. Each example is a description of a situation together with a specification, the label, of the correct action the system should take to that situation, which is often to identify a category to which the situation belongs. 

The objective of this kind of learning is for the system to extrapolate or generalize its responses so that it acts correctly in situations not present in the training set.

<span class = 'hital'>Unsupervised Machine Learning.</span>
In this type of Machine Learning, models try to figure out patterns in the data without labels. Unsupervised learning is aimed at finding structures hidden in collections of unlabelled data.

#### [1.4] Is Reinforcement Learning same as Machine Learning?

Although one might be tempted to think of reinforcement learning as a kind of unsupervised learning (because it does not rely on examples of correct behavior), reinforcement learning tries to maximize a reward signal instead of trying to find a hidden structure.

Reinforcement learning takes a different track compared to Machine Learning, starting with a complete, interactive, goal-seeking agent. All reinforcement learning agents have explicit goals, can sense aspects of their environments, and can choose actions to influence their environments.

When reinforcement learning involves planning, it has to address the interplay between planning and real-time action selection, as well as the question of how environment models are acquired and improved.

<!-- #### [0.1.5] Reinforcement Learning Fundamentals
<span style="color:blue;">Agent:</span>
An agent is a software program that learns to make intelligent decisions. For instance, a chess player can be considered an agent since the player learns to make the best moves (decisions) to win the game. Similarly, Mario in a Super Mario Bros video game can be considered an agent since Mario explores the game and learns to make the best moves in the game.

<span style="color:blue;">Environment:</span>
The environment is the world of the agent. The agent interacts within the environments. For example, a chessboard is called the environment for the chess player agent.

<span style="color:blue;">State and Action:</span>
A state is a position or a moment in the environment that the agent can be in. There can be many `positions in the chess board environment` that we discussed earlier. All these positions on the chess board are cosidered to be the state. The movement of the chess player agent (forward, backwward, right, and left) are known as actions. (A state is denoted by $s$, and an action is denoted by $a$).

<span style="color:blue;">Reward:</span>
As we discussed earlier, the agent interacts with the environments by performing actions. Every action has a reward associated to it. This reward is a numerical value, _(+1 or -1 for example)_ that denotes if the agent performed an optimal action or not.

The goal of the RL agent is to `maximize this reward` by performing an optimal set of actions.

<span style="color:blue;">Policy:</span>
A policy defines the learning agent???s way of behaving at a given time.

<span style="color:blue;">Value Functions:</span>
Whereas the reward signal indicates what is good in an immediate sense, a value function specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Rewards are in a sense primary, whereas values, as predictions of rewards, are secondary. Without rewards there could be no values, and the only purpose of estimating values is to achieve more reward. Rewards are basically given directly by the environment, but values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime.

<span style="color:blue;">Model-based and Model-free methods:</span>
Methods for solving reinforcement learning problems that use models and planning are called model-based methods, as opposed to simpler model-free methods that are explicitly trial-and-error learners???viewed as almost the opposite of planning. -->
<!-- #endregion -->
