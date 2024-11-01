from deep_nn_policy import DeepNeuralNetworkPolicy
from q_actor_critic import QActorCritic
from deep_q_function import DeepQFunction
from gridworld import GridWorld
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from qlearning import QLearning

mdp = GridWorld()
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

# Instantiate the critic
critic = DeepQFunction(state_space, action_space)

# Instantiate the actor
actor = DeepNeuralNetworkPolicy(state_space, action_space)

#  Instantiate the actor critic agent
learner = QActorCritic(mdp, actor, critic)
episode_rewards = learner.execute(episodes=1000)
mdp.visualise_q_function(critic)
mdp.visualise_stochastic_policy(actor)
