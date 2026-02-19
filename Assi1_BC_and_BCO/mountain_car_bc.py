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


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='rgb_array') 
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos


# --- ADDED CODE: PROGRAMMATIC EXPERTS ---
def collect_programmatic_demos(num_demos, style='good'):
    """
    Generates demonstrations algorithmically.
    style='good': Uses optimal momentum strategy (Left then Right).
    style='bad':  Constantly pushes Right (Action 2), effectively failing.
    """
    print(f"Collecting {num_demos} {style} programmatic demonstrations...")
    env = gym.make("MountainCar-v0")
    sas_pairs = []
    
    for _ in range(num_demos):
        obs, _ = env.reset()
        done = False
        while not done:
            if style == 'good':
                # Expert policy: Build momentum
                position, velocity = obs
                if velocity > 0:
                    action = 2 # Right
                elif velocity < 0:
                    action = 0 # Left
                else:
                    action = 0 # Kick start
            else:
                # Bad policy: Just push right (Action 2) blindly
                action = 2
                
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
    
    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).long().to(device)

    return obs_torch, acs_torch, obs2_torch


def train_policy(obs, acs, nn_policy, num_train_iters):
    pi_optimizer = Adam(nn_policy.parameters(), lr=0.1)
    #action space is discrete so our policy just needs to classify which action to take
    #we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()
    
    # run BC using all the demos in one giant batch
    for i in range(num_train_iters):
        #zero out automatic differentiation from last time
        pi_optimizer.zero_grad()
        #run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = nn_policy(obs)
        #now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_action_logits, acs) 
        if i % 20 == 0: # Added print for monitoring
            print("iteration", i, "bc loss", loss.item())
        #back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        #perform update on policy parameters
        pi_optimizer.step()



class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This layer has 2 inputs corresponding to car position and velocity
        self.fc1 = nn.Linear(2, 8)  
        #This layer has three outputs corresponding to each of the three discrete actions
        self.fc2 = nn.Linear(8, 3)  



    def forward(self, x):
        #this method performs a forward pass through the network, applying a non-linearity (ReLU) on the 
        #outputs of the first layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    

#evaluate learned policy
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
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the 
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))
    
    # --- ADDED CODE: Return mean for comparison ---
    return np.mean(policy_returns)
    # --- END ADDED CODE ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    
    # --- ADDED CODE: New Arguments for Bad Demos and Mode ---
    parser.add_argument('--num_bad_demos', default=0, type=int, help="number of BAD demonstrations to mix in")
    parser.add_argument('--mode', type=str, default='compare', choices=['human', 'expert', 'compare'], 
                        help="Choose demonstration type: human, expert, or compare (runs both)")
    # --- END ADDED CODE ---

    args = parser.parse_args()

    # --- ADDED CODE: Pipeline Logic ---
    def run_bc_pipeline(demo_source):
        demos = []
        
        # 1. Collect Data
        if demo_source == 'human':
            total_demos_needed = args.num_demos + args.num_bad_demos
            print(f"\n*** Please provide {total_demos_needed} demonstrations in Pygame ***")
            if args.num_bad_demos > 0:
                print(f"(Note: Please perform {args.num_demos} GOOD demos and {args.num_bad_demos} BAD demos)")
            demos = collect_human_demos(total_demos_needed)
            
        else: # Expert
            # Collect Good Demos
            if args.num_demos > 0:
                demos.extend(collect_programmatic_demos(args.num_demos, style='good'))
            # Collect Bad Demos
            if args.num_bad_demos > 0:
                demos.extend(collect_programmatic_demos(args.num_bad_demos, style='bad'))

        # 2. Process Data
        obs, acs, _ = torchify_demos(demos)

        # 3. Train Policy
        pi = PolicyNetwork()
        train_policy(obs, acs, pi, args.num_bc_iters)

        # 4. Evaluate
        return evaluate_policy(pi, args.num_evals)

    # Execution Flow
    if args.mode in ['human', 'expert']:
        run_bc_pipeline(args.mode)
        
    elif args.mode == 'compare':
        print(f"\n{'='*60}")
        print("PHASE 1: TRAINING BC ON HUMAN DEMONSTRATIONS")
        print(f"{'='*60}")
        human_avg = run_bc_pipeline('human')
        
        print(f"\n{'='*60}")
        print("PHASE 2: TRAINING BC ON EXPERT DEMONSTRATIONS")
        print(f"{'='*60}")
        expert_avg = run_bc_pipeline('expert')
        
        print(f"\n\n{'*'*60}")
        print("FINAL BC PERFORMANCE COMPARISON")
        print(f"{'*'*60}")
        print(f"Human Demonstrations Average Return:  {human_avg:.2f}")
        print(f"Expert Demonstrations Average Return: {expert_avg:.2f}")
        print(f"{'*'*60}")
    # --- END ADDED CODE ---
