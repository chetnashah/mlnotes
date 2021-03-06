
An agent finds itself in an environment, and tries to achieve
a goal in that environment.

The agent gets its input/observation+rewards via sensors.
The agent interacts via its output using actuators.
Combined together we say the agent is in a sensori-motor reality.

To make sense of high dimensional sensory input - Deep learning is used.

The agent builds model(approximation) of the environment,
and builds a plan/policy for the future and acts based on it.

Biological systems like dopamine in human brain have similar behavior.

### Charechteristics of Reinforcement Learning

* No supervisor, only a reward signal. Learn from rewards.

* Feedback is delayed, not instantaneous (so may be some sort of time-memory relation)

* Time really matters (sequential, non i.i.d data)

* Agent's actions affect subsequent data is receives, and thus also subsequent rewards.

* The two main charechteristics or RL - 
    1. trial and error search
    2. delayed rewards 

### Reward hypothesis

All goals can be described by maximisation of expected cumulative numerical reward. Agent's job is to reach goal or in other words maximise cumulative numerical reward.

#### When exactly do we get rewards?

It essentially boils down to a modelling choice: for some problems, the reward function might be easier to define on the (state,action) pairs, while for others, the tuple (state,action,state) might be more appropriate. There's even a third option that only defines the reward on the current state (this can also be found in some references).

I do think the definition of the reward function R(s,a) on the (state, action) pair is the most common, however. But the core learning algorithms remain the same whatever your exact design choice for the reward function.

Also usually there are two steps in decision making, first agent takes an action, and then the environment probabilistically takes it to some state.

### Observations vs States

Observations are raw input streams of data,
State is an internal representation of observations etc. that agent cares about, also known as the agent state.
Also known as information that may charechterize our future behavior when used as input to our policy.
Formally, State is a function of history.

#### Information/Markov State

An information/markov state contains all useful information from the history. The markov property being
future is independent of past, given the present.

### Observability

#### Full Observability

Also known as perfect information,
Agent State = environment state.
MDP can be applied

#### Partial observability (hidden information)

Agent indirectly observes environment, e.g. robot with camera vision which does not know its absolute location, so it has to localize.
or e.g. a poker agent which can only see public cards
In partial observability, agent state != environment state
POMDP can be applied (Partially observable MDP)

### Components of an RL agent

#### Policy

Agent's behaviour function, Given a state, tells us what action to take.
Simplest policy : 

#### Value function

Tells us how good each state and/or action is, i.e a goodness is total expected future reward. Note: A reward signal indicates how good it is in immediate sense, but value function specifies goodness in the long run.
Value function given the state & the policy we are going to follow, tells us the total expected future reward.

It is hard to determine value compared to reward, rewards are basically given to us by environment, where as values are estimated and re-estimated by sequences of observations an agent makes over its entire lifetime.



#### Model

Agent's representation of the environment

### Exploration vs Exploitation

Exploration: finding out more information about environment via experimentation (some randomness/trial and error) without losing too much reward along the way.

Exploitation: exploits known information to maximize reward.

### epsilon-greedy methods

Greedy action selection always exploits current knowledge to maximize immediate
reward; it spends no time at all sampling apparently inferior actions to see if they might really be better.
A simple alternative is to behave greedily most of the time, but every once in a while, say with small
probability ε, instead select randomly from among all the actions with equal probability, independently
of the action-value estimates. We call methods using this near-greedy action selection rule ε-greedy
methods

### What is return Gt for a given MDP sample?

Return - It is sum of discounted rewards for states in the sample.

### Why are markov processes discounted in RL?

Ther is a certain factor of uncertaininity i.e we do not have a perfect model, and we dont
want to consider too much of the future if is uncertain. You really would have to believe your model to keep discount factors ~1 and believe things are going to work out in the future, even if they are not being done so now.

Also it is mathematically convinient, and avoids infinite returns.

### What is a value function of a given state?

It is long term value of being in a state. i.e the expected return starting from state s, until you terminate. We have expected return since the environment is stochastic.

#### MDPs and terminology

* Set of States : Not just a single state, current state, but set of states.

* Transitions/Model/Dynamics: Physics of the environment, tells us all possible actions I can take in a given state

* Rewards: Reward that we get on entering a state.

* Actions: Set of possible actions, not all actions are possible in all states, but some are possible in some (only Transition rules can tell you what is valid when).

**Policy** - Given a state, tells us what action to take. ( state -> action )

**Utility** - Sum of discounted rewards


### Policy search

 If the space of policies is suﬃciently small, or can be structured so that good policies are common or easy to ﬁnd—or if a lot of time is available for the search—then evolutionary methods(simmulated annealing, genetic algorithms) can be eﬀective. In addition, evolutionary methods have advantages on problems in which the learning agent cannot sense the complete state of its environment. 

 is example illustrates the diﬀerences between evolutionary methods and methods that learn value functions. To evaluate a policy an evolutionary method holds the policy ﬁxed and plays many games against the opponent, or simulates many games using a model of the opponent. The frequency of wins gives an unbiased estimate of the probability of winning with that policy, and can be used to direct the next policy selection. But each policy change is made only after many games, and only the ﬁnal outcome of each game is used: what happens during the games is ignored. For example, if the player wins, then all of its behavior in the game is given credit, independently of how speciﬁc moves might have been critical to the win. Credit is even given to moves that never occurred! Value function methods, in contrast, allow individual states to be evaluated. In the end, evolutionary and value function methods both search the space of policies, but learning a value function takes advantage of information available during the course of play. 

### How we navigate state space?

s1, a1, r1,
s2, a2, r2,
s3, a3, r3.

### Practice

Use open AI gym.
Use robots.
Use games.


### Game Theory

In a 2-player, zero-sum deterministic game
of perfect information,
Minimax = Maximin, there always exists an optimal pure strategy for each player.

