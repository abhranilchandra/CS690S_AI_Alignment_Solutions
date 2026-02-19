import gymnasium as gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cpu')

# This will be useful for implementing BCO
def collect_random_interaction_data(num_iters):
    states = []
    next_states = []
    actions = []

    env = gym.make('MountainCar-v0')

    for i in range(num_iters):
        obs, _ = env.reset()    
        done = False
        while not done:
            a = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            states.append(obs)
            next_states.append(next_obs)
            actions.append(a)
            obs = next_obs
    
    env.close()

    return np.array(states), np.array(next_states), np.array(actions)


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='rgb_array') 
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos


# --- ADDED CODE: PROGRAMMATIC EXPERT ---
# Added to simulate near perfect demonstrations to compare against the noisy human demos.
def collect_expert_demos(num_demos):
    print("Collecting perfect algorithmic demonstrations...")
    env = gym.make("MountainCar-v0")
    sas_pairs = []
    
    for _ in range(num_demos):
        obs, _ = env.reset()
        done = False
        while not done:
            # Expert policy: perfectly push in the direction of velocity to build momentum
            position, velocity = obs
            if velocity > 0:
                action = 2 # Accelerate Right
            elif velocity < 0:
                action = 0 # Accelerate Left
            else:
                action = 0 # Initial kick to the left
                
            next_obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            sas_pairs.append((obs, action, next_obs))
            obs = next_obs
            
    return sas_pairs
# --- END ADDED CODE ---


def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s,a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).long().to(device)

    return obs_torch, acs_torch, obs2_torch


# --- ADDED CODE: INVERSE DYNAMICS NETWORK & TRAINING ---
# Added the network architecture and training loop to learn the transition dynamics
# (i.e., mapping state transitions back to the actions that caused them).
class InverseDynamicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def inverse_dynamics(obs, next_obs, num_train_iters, num_data_steps):
    print(f"Collecting {num_data_steps} random interaction steps for Inverse Dynamics...")
    rand_obs, rand_next_obs, rand_acts = collect_random_interaction_data(num_data_steps)
    
    rand_obs_t = torch.from_numpy(rand_obs).float().to(device)
    rand_next_obs_t = torch.from_numpy(rand_next_obs).float().to(device)
    rand_acts_t = torch.from_numpy(rand_acts).long().to(device)

    inv_model = InverseDynamicsNetwork().to(device)
    optimizer = Adam(inv_model.parameters(), lr=0.01)
    loss_criterion = nn.CrossEntropyLoss()

    print("Training Inverse Dynamics Model...")
    for i in range(num_train_iters):
        optimizer.zero_grad()
        pred_logits = inv_model(rand_obs_t, rand_next_obs_t)
        loss = loss_criterion(pred_logits, rand_acts_t)
        
        if i % 100 == 0:
            print(f"ID Iteration {i}, Loss: {loss.item():.4f}")
            
        loss.backward()
        optimizer.step()

    inv_model.eval()
    with torch.no_grad():
        est_logits = inv_model(obs, next_obs)
        estimated_acts = torch.argmax(est_logits, dim=1)

    return estimated_acts
# --- END ADDED CODE ---


def train_policy(obs, acs, nn_policy, num_train_iters):
    pi_optimizer = Adam(nn_policy.parameters(), lr=0.1)
    loss_criterion = nn.CrossEntropyLoss()
    
    for i in range(num_train_iters):
        pi_optimizer.zero_grad()
        pred_action_logits = nn_policy(obs)
        loss = loss_criterion(pred_action_logits, acs) 
        if i % 20 == 0:  # ADDED: Print to monitor BC loss
            print("iteration", i, "bc loss", loss.item())
        loss.backward()
        pi_optimizer.step()


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)  
        self.fc2 = nn.Linear(8, 3)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human') 
    else:
        env = gym.make("MountainCar-v0") 

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs, _ = env.reset()
        while not done:
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))
    
    # --- ADDED CODE: Return mean so we can compare the policies later ---
    return np.mean(policy_returns)
    # --- END ADDED CODE ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 5, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_inv_dyn_iters', default = 1000, type=int, help="number of iterations to train inverse dynamics model")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    
    # --- ADDED CODE: New Arguments ---
    # Allow the user to select human, expert, or compare mode from the terminal
    parser.add_argument('--mode', type=str, default='compare', choices=['human', 'expert', 'compare'], 
                        help="Choose demonstration type: human, expert, or compare (runs both)")
    
    # NEW ARGUMENT FOR PART 7 MODIFICATION
    parser.add_argument('--num_random_interactions', default=25000, type=int, 
                        help="Number of random interaction steps to collect for Inverse Dynamics training")
    # --- END ADDED CODE ---
    
    args = parser.parse_args()

    # --- ADDED CODE: Pipeline wrapper ---
    # We wrap the professor's original main block sequence in a function so that 
    # we can call it twice (once for human, once for expert) during the 'compare' mode.
    def run_bco_pipeline(demo_source):
        if demo_source == 'human':
            print("\n*** Please use the Pygame window to provide human demonstrations! ***")
            demos = collect_human_demos(args.num_demos)
        else:
            demos = collect_expert_demos(args.num_demos)

        # Process demos (Original Professor Code)
        obs, ground_truth_acts, next_obs = torchify_demos(demos)

        # Train ID model and estimate actions (Added Part 7 Code)
        # UPDATED CALL: Passing both training iters and data collection steps
        estimated_acts = inverse_dynamics(obs, next_obs, args.num_inv_dyn_iters, args.num_random_interactions)
        
        # Output ID Accuracy
        if len(ground_truth_acts) > 0:
            correct_preds = (estimated_acts == ground_truth_acts).sum().item()
            accuracy = correct_preds / len(ground_truth_acts)
            print(f"\n[INFO] Inverse Dynamics Accuracy on {demo_source.capitalize()} Demo Data: {accuracy * 100:.2f}%\n")

        # Train policy WITHOUT ground truth actions (Original Professor Code)
        pi = PolicyNetwork()
        train_policy(obs, estimated_acts, pi, args.num_bc_iters)

        # Evaluate learned policy (Original Professor Code)
        avg_return = evaluate_policy(pi, args.num_evals)
        return avg_return
    # --- END ADDED CODE ---


    # --- ADDED CODE: Execution Logic ---
    if args.mode in ['human', 'expert']:
        run_bco_pipeline(args.mode)
        
    elif args.mode == 'compare':
        print(f"\n{'='*60}")
        print("PHASE 1: TRAINING BCO ON HUMAN DEMONSTRATIONS")
        print(f"{'='*60}")
        human_avg = run_bco_pipeline('human')
        
        print(f"\n{'='*60}")
        print("PHASE 2: TRAINING BCO ON EXPERT DEMONSTRATIONS")
        print(f"{'='*60}")
        expert_avg = run_bco_pipeline('expert')
        
        # Final comparison printout
        print(f"\n\n{'*'*60}")
        print("FINAL BCO PERFORMANCE COMPARISON")
        print(f"{'*'*60}")
        print(f"Human Demonstrations Average Return:  {human_avg:.2f}")
        print(f"Expert Demonstrations Average Return: {expert_avg:.2f}")
        print(f"{'*'*60}")
        if expert_avg > human_avg:
            print("Conclusion: As expected, the perfect programmatic expert generated\n"
                  "cleaner, optimal data, leading the BCO agent to a higher score.")
        else:
            print("Conclusion: The human demonstrator performed exceptionally well!")
    # --- END ADDED CODE ---
