from ppo import PPO
from deep_nn_policy import DeepNeuralNetworkPolicy
from deep_value_function import DeepValueFunction
from tabular_value_function import TabularValueFunction
from gridworld import GridWorld
from plot import Plot

gridworld = GridWorld()

# Instantiate the critic
critic = DeepValueFunction(state_space=len(gridworld.get_initial_state()), hidden_dim=16)

# Instantiate the actor
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
actor = DeepNeuralNetworkPolicy(state_space, action_space)

learner = PPO(mdp=gridworld, actor=actor, critic=critic)
rewards = learner.execute(200)

print(gridworld.value_function_to_string(critic))
print(gridworld.stochastic_policy_to_string(actor))

Plot.plot_cumulative_rewards(["PPO"], [rewards], smoothing_factor=0.8)
