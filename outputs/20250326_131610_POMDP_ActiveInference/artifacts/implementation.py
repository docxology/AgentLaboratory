import numpy as np
from enum import Enum

# Define control actions
class Action(Enum):
    COOL = 1
    NOTHING = 2
    HEAT = 3

# Define latent states
class State(Enum):
    VERY_COLD = 1
    COLD = 2
    COMFORTABLE = 3
    WARM = 4
    HOT = 5

# Define observation levels
class Observation(Enum):
    VERY_COLD = 1
    COLD = 2
    SLIGHTLY_COLD = 3
    COMFORTABLE = 4
    SLIGHTLY_WARM = 5
    WARM = 6
    HOT = 7
    VERY_HOT = 8
    EXTREME_HOT = 9
    OUT_OF_RANGE = 10

# Initialize transition matrix
num_states = len(State)
num_actions = len(Action)

# Transition model as a 3D NumPy array
transition_matrix = np.zeros((num_states, num_states, num_actions))

# Example of defining transition probabilities for action COOL
transition_matrix[State.VERY_COLD.value - 1, State.COLD.value - 1, Action.COOL.value - 1] = 0.8
transition_matrix[State.VERY_COLD.value - 1, State.VERY_COLD.value - 1, Action.COOL.value - 1] = 0.2

# Define other transitions similarly...
# e.g. transition_matrix[State.COLD.value - 1, State.COMFORTABLE.value - 1, Action.NOTHING.value - 1] = 0.7

# Initialize observation model as a NumPy array
num_observations = len(Observation)
observation_matrix = np.zeros((num_states, num_observations))

# Example probabilities for observations given states
observation_matrix[State.VERY_COLD.value - 1, Observation.VERY_COLD.value - 1] = 0.9
observation_matrix[State.VERY_COLD.value - 1, Observation.COLD.value - 1] = 0.1

# Define other observation probabilities similarly...
# e.g. observation_matrix[State.COLD.value - 1, Observation.COLD.value - 1] = 0.8

def reward_function(state: State, action: Action) -> float:
    if state == State.COMFORTABLE and action == Action.NOTHING:
        return 10  # High reward for maintaining comfort
    elif action == Action.COOL:
        return -5  # Cost for cooling
    elif action == Action.HEAT:
        return -5  # Cost for heating
    else:
        return -1  # Small penalty for other actions

def variational_free_energy(observations: int, prior_beliefs: np.ndarray) -> float:
    log_likelihood = np.sum(np.log(observation_matrix[:, observations]))
    kl_divergence = np.sum(prior_beliefs * np.log(prior_beliefs / np.mean(prior_beliefs)))

    vfe = log_likelihood - kl_divergence
    return vfe

def expected_free_energy(current_beliefs: np.ndarray) -> np.ndarray:
    expected_rewards = np.zeros(num_actions)
    
    for action in range(num_actions):
        for next_state in range(num_states):
            expected_rewards[action] += transition_matrix[:, next_state, action] * reward_function(State(next_state + 1), Action(action + 1))
    
    return expected_rewards  # Return expected rewards for each action

def main():
    # Initialize prior beliefs (uniform distribution)
    prior_beliefs = np.ones(num_states) / num_states

    # Simulate some observations
    observations_sequence = [np.random.choice(num_observations) for _ in range(10)]

    for observation in observations_sequence:
        # Update beliefs using variational free energy
        vfe = variational_free_energy(observation, prior_beliefs)

        # Calculate expected free energy for action selection
        efe = expected_free_energy(prior_beliefs)

        # Select action that minimizes expected free energy
        action = np.argmin(efe)
        print(f"Action taken: {Action(action + 1).name}, Variational Free Energy: {vfe:.2f}, Expected Free Energy: {efe[action]:.2f}")

if __name__ == "__main__":
    main()