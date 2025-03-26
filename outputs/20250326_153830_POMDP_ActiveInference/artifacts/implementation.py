import numpy as np
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define Control States
class ControlState(Enum):
    COOL = 0
    NOTHING = 1
    HEAT = 2

# Define Latent States
class LatentState(Enum):
    VERY_COLD = 0
    COLD = 1
    COMFORTABLE = 2
    WARM = 3
    VERY_HOT = 4

# Define Observation Levels
class ObservationLevel(Enum):
    VERY_COLD = 0
    COLD = 1
    COOL = 2
    SLIGHTLY_COOL = 3
    COMFORTABLE = 4
    SLIGHTLY_WARM = 5
    WARM = 6
    HOT = 7
    VERY_HOT = 8
    EXTREMELY_HOT = 9

# Transition Model
T = np.array([
    [0.1, 0.7, 0.2, 0, 0],  # Transitions from VERY_COLD
    [0, 0.2, 0.6, 0.2, 0],  # Transitions from COLD
    [0, 0, 0.3, 0.4, 0.3],  # Transitions from COMFORTABLE
    [0, 0, 0, 0.3, 0.7],    # Transitions from WARM
    [0, 0, 0, 0.1, 0.9]     # Transitions from VERY_HOT
])

# Observation Model
O = np.array([
    [0.8, 0.2, 0, 0, 0],    # Observation probabilities for VERY_COLD
    [0, 0.7, 0.3, 0, 0],    # Observation probabilities for COLD
    [0, 0, 0.5, 0.4, 0.1],  # Observation probabilities for COMFORTABLE
    [0, 0, 0, 0.6, 0.4],    # Observation probabilities for WARM
    [0, 0, 0, 0.2, 0.8]     # Observation probabilities for VERY_HOT
])

# Reward Function
def reward_function(state: int, action: int) -> float:
    """
    Calculate the reward based on the current state and action taken.
    
    Parameters:
    - state: The current latent state (integer).
    - action: The action taken (integer).
    
    Returns:
    - Reward (float).
    """
    # Example: Define energy consumption and comfort deviation
    energy_consumption = np.array([1, 0, 2])  # Energy cost for COOL, NOTHING, HEAT
    comfort_deviation = np.abs(state - 2)  # Assuming 2 (COMFORTABLE) is the ideal state
    return energy_consumption[action] - comfort_deviation

# Variational Free Energy Calculation
def variational_free_energy(observations: int, prior: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate the Variational Free Energy and posterior distribution over latent states.
    
    Parameters:
    - observations: The observed state (integer).
    - prior: The prior distribution over latent states (numpy array).
    
    Returns:
    - VFE (float), posterior distribution (numpy array).
    """
    # Calculate the posterior distribution over latent states
    posterior = np.zeros(len(LatentState))
    for s in range(len(LatentState)):
        likelihood = O[s, observations]
        posterior[s] = prior[s] * likelihood
    posterior /= np.sum(posterior)  # Normalize

    # Calculate VFE
    vfe = -np.sum(posterior * np.log(posterior + 1e-10))  # Avoid log(0)
    return vfe, posterior

# Expected Free Energy Calculation
def expected_free_energy(current_state: int, observations: int) -> int:
    """
    Calculate the Expected Free Energy for each action and select the best action.
    
    Parameters:
    - current_state: The current latent state (integer).
    - observations: The observed state (integer).
    
    Returns:
    - Selected action (integer).
    """
    efe_values = []
    for action in ControlState:
        expected_reward = 0
        for next_state in range(len(LatentState)):
            # Calculate expected reward for each next state given the action
            reward = reward_function(next_state, action.value)
            expected_reward += T[current_state, next_state] * reward
        efe_values.append(expected_reward)
    return np.argmin(efe_values)  # Return the action that minimizes EFE

# Main function to demonstrate the model's behavior
def main():
    # Initial prior distribution over latent states
    prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform prior
    current_state = 2  # Start in the COMFORTABLE state

    # Simulate the environment
    for step in range(100):  # Run for 100 time steps
        # Simulate an observation (for example, from a sensor)
        observation = np.random.choice(range(len(ObservationLevel)))  # Random observation
        logging.info(f"Step {step}: Observation = {ObservationLevel(observation).name}")

        # Update belief about the state using VFE
        vfe, posterior = variational_free_energy(observation, prior)
        logging.info(f"Step {step}: VFE = {vfe:.4f}, Posterior = {posterior}")

        # Select action using EFE
        action = expected_free_energy(current_state, observation)
        logging.info(f"Step {step}: Selected Action = {ControlState(action).name}")

        # Update the current state based on the action (for simplicity, assume deterministic)
        current_state = np.random.choice(range(len(LatentState)), p=T[current_state])

if __name__ == "__main__":
    main()