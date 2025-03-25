import numpy as np
from datasets import load_dataset
import random

# Load an external HuggingFace dataset (here we use a small subset of the AG News dataset)
external_dataset = load_dataset("ag_news", split="train[:1%]")
print("External dataset sample:")
print(external_dataset[0])

# Parameters for synthetic thermal homeostat simulation
num_steps = 100  # number of simulation time steps
latent_states = [0, 1, 2, 3, 4]  # 0: very-cold, ..., 4: hot
actions = ["cool", "nothing", "heat"]
observation_levels = np.arange(10)  # discrete observation levels: 0 to 9

# Simulation parameters
# Probability of state transition based on action: simplified physics-inspired dynamics
# if action is "heat": move up with probability 0.8, stay with 0.2; "cool": move down with probability 0.8, stay with 0.2; "nothing": small drift.
state_transition_prob = {
    "heat": {"up": 0.8, "stay": 0.2},
    "cool": {"down": 0.8, "stay": 0.2},
    "nothing": {"drift": 0.4, "stay": 0.6}
}

# Prior preference target: aim for a moderate temperature corresponding to latent state 2.
target_state = 2

# Initialize arrays to store simulation data
times = []
latent_state_history = []
action_history = []
observation_history = []
ambient_temp_history = []  # external factor: ambient temperature

current_state = 2  # start in 'moderate' state

for t in range(num_steps):
    times.append(t)
    
    # Randomly select an action for simulation (in a real experiment, this would be based on computing EFE)
    chosen_action = random.choice(actions)
    action_history.append(chosen_action)
    
    # Apply transition dynamics based on chosen action
    if chosen_action == "heat":
        if current_state < max(latent_states):
            # With probability 0.8 increase the latent state, else remain same
            if random.random() < state_transition_prob["heat"]["up"]:
                current_state += 1
        # else remain at max
    elif chosen_action == "cool":
        if current_state > min(latent_states):
            if random.random() < state_transition_prob["cool"]["down"]:
                current_state -= 1
    elif chosen_action == "nothing":
        # drift: possibility to drift up or down slightly if not at boundaries (simulate noise)
        if current_state > min(latent_states) and current_state < max(latent_states):
            drift = random.choice([-1, 0, 1])
            current_state = min(max(current_state + drift, min(latent_states)), max(latent_states))
    latent_state_history.append(current_state)
    
    # Observation model: Map latent state to a discrete observation level out of 10 with some added noise.
    # We assume each latent state corresponds roughly to 2 observation levels.
    base_obs = current_state * 2  # 0,2,4,6,8 ideally
    noise = np.random.randint(-1, 2)  # noise in {-1, 0, 1}
    observation = int(np.clip(base_obs + noise, 0, 9))
    observation_history.append(observation)
    
    # Ambient temperature: simulate an external temperature effect as a random fluctuation around 20 degrees Celsius.
    ambient_temp = 20 + np.random.normal(0, 1)   # mean 20, std 1
    ambient_temp_history.append(ambient_temp)

# Prepare the synthetic dataset as a dictionary
synthetic_data = {
    "time": times,
    "latent_state": latent_state_history,
    "action": action_history,
    "observation": observation_history,
    "ambient_temperature": ambient_temp_history,
    "target_state": [target_state] * num_steps  # constant target state for reference
}

print("Synthetic thermal homeostat dataset sample:")
sample_idx = np.random.choice(num_steps, 5, replace=False)
for idx in sorted(sample_idx):
    print({key: synthetic_data[key][idx] for key in synthetic_data})