
An agent finds itself in an environment, and tries to achieve
a goal in that environment.

The agent gets its input/observation+rewards via sensors.
The agent interacts via its output using actuators.
Combined together we say the agent is in a sensori-motor reality.

To make sense of high dimensional sensory input - Deep learning is used.

The agent builds model(approximation) of the environment,
and builds a plan/policy for the future and acts based on it.

Biological systems like dopamine in human brain have similar behavior.

#### MDPs and terminology

* Set of States : Not just a single state, current state, but set of states.

* Transitions/Model/Dynamics: Physics of the environment, tells us all possible actions I can take in a given state

* Rewards: Reward that we get on entering a state.

* Actions: Set of possible actions, not all actions are possible in all states, but some are possible in some (only Transition rules can tell you what is valid when).

**Policy** - Given a state, tells us what action to take. ( state -> action )

**Utility** - Sum of discounted rewards
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