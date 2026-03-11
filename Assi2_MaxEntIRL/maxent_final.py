import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import math
import os
import sys
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
try:
    from matplotlib.cbook import MatplotlibDeprecationWarning
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
except ImportError:
    pass


# CUSTOM LOGGER FOR RESULTS

class Logger(object):
    def __init__(self, filename="maxent_irl_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def build_trans_mat_gridworld():
    trans_mat = np.zeros((26,4,26))
    for s in range(24):
        if s < 20: trans_mat[s,0,s+5] = 1
        else: trans_mat[s,0,s] = 1
    for s in range(24):
        if s >= 5: trans_mat[s,1,s-5] = 1
        else: trans_mat[s,1,s] = 1
    for s in range(24):
        if s%5 > 0: trans_mat[s,2,s-1] = 1
        else: trans_mat[s,2,s] = 1
    for s in range(24):
        if s%5 < 4: trans_mat[s,3,s+1] = 1
        else: trans_mat[s,3,s] = 1
    for a in range(4): trans_mat[24,a,25] = 1  
    return trans_mat

def build_state_features_gridworld():
    sf = np.zeros((26,4))  
    sf[0:6,0] = 1; sf[6:10,1] = 1; sf[10:12,0] = 1; sf[12,2] = 1
    sf[13:24,0] = 1; sf[24,3] = 1
    return sf

def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index):
    n_states, n_actions = np.shape(trans_mat)[0], np.shape(trans_mat)[1]
    rewards = np.dot(state_features, r_weights)
    Z = np.zeros(n_states)
    Z[term_index] = 1.0
    Za = np.zeros((n_states, n_actions))
    for i in range(horizon):
        expected_next_Z = np.einsum('san,n->sa', trans_mat, Z)
        Za = expected_next_Z * np.exp(rewards)[:, None]
        Z = np.sum(Za, axis=1)
        Z[term_index] = 1.0 
    with np.errstate(divide='ignore', invalid='ignore'):
        policy = np.nan_to_num(Za / Z[:, None]) 
    return policy

def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
    n_states = np.shape(trans_mat)[0]
    D = np.zeros((n_states, horizon + 1))
    D[:, 0] = start_dist
    for t in range(horizon):
        P_sa = D[:, t][:, None] * policy
        D[:, t+1] = np.einsum('sa,san->n', P_sa, trans_mat)
    return np.sum(D, axis=1)

def train_and_track(trans_mat, state_features, demos, lr, epochs=500, horizon=15):
    n_states, n_features = np.shape(state_features)
    feat_exp = np.zeros(n_features)
    for d in demos:
        for s in d: feat_exp += state_features[s]
    feat_exp /= len(demos)
    
    start_dist = np.zeros(n_states)
    for d in demos: start_dist[d[0]] += 1
    start_dist /= len(demos)
    
    r_weights = np.zeros(n_features)
    grad_norms = []
    
    for i in range(epochs):
        pol = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, 25)
        svf = calcExpectedStateFreq(trans_mat, horizon, start_dist, pol)
        exp_feats = np.dot(state_features.T, svf)
        grad = feat_exp - exp_feats
        grad_norms.append(np.linalg.norm(grad))
        r_weights += lr * grad
        
    return r_weights, grad_norms, pol, svf, feat_exp

def value_iteration(trans_mat, rewards, gamma=0.99, eps=1e-5):
    """
    Computes the optimal deterministic policy given the learned rewards.
    """
    n_states, n_actions, _ = trans_mat.shape
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    
    while True:
        delta = 0
        for s in range(n_states):
            if s == 25: # Terminal state
                continue
            
            v = V[s]
            Q = np.zeros(n_actions)
            for a in range(n_actions):
                expected_v = 0
                for s_next in range(n_states):
                    prob = trans_mat[s, a, s_next]
                    if prob > 0:
                        expected_v += prob * (rewards[s_next] + gamma * V[s_next])
                Q[a] = expected_v
                
            V[s] = np.max(Q)
            policy[s] = np.argmax(Q)
            delta = max(delta, abs(v - V[s]))
            
        if delta < eps:
            break
            
    return policy

if __name__ == '__main__':
    # Initialize the logger to save all outputs to a text file
    sys.stdout = Logger("maxent_irl_log.txt")

    trans_mat = build_trans_mat_gridworld()
    sf = build_state_features_gridworld() 
    demos = [[4,9,14,19,24,25],[3,8,13,18,19,24,25],[2,1,0,5,10,15,20,21,22,23,24,25],[1,0,5,10,11,16,17,22,23,24,25]]

    # 1. Full LR Ablation Setup
    lrs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}
    best_lr, best_final_grad = None, float('inf')

    f_exp = np.zeros(np.shape(sf)[1])
    for d in demos:
        for s in d: f_exp += sf[s]
    f_exp /= len(demos)
    
    print("--- Common Configurations ---")
    print(f"Empirical Feature Expectations: \n{f_exp}\n")

    print("--- Ablation Detailed Results ---")
    for lr in lrs:
        w, g, pol, svf, _ = train_and_track(trans_mat, sf, demos, lr)
        results[lr] = {'weights': w, 'grads': g, 'svf': svf}
        final_grad = g[-1]
        
        if not np.isnan(final_grad) and final_grad < best_final_grad and max(g) < 20: 
            best_final_grad, best_lr = final_grad, lr
            
        print(f"======================================================")
        print(f"LR: {lr:4.2f} | Final Grad Norm (Error): {final_grad:.4f}")
        print(f"Learned Reward Weights: \n{w}")
        
        reward_fxn = [np.dot(w, sf[s_i]) for s_i in range(25)]
        reward_fxn_grid = np.reshape(reward_fxn, (5,5))
        grid_feats = np.argmax(sf[:25], axis=1).reshape(5,5)
        
        print(f"\nRecovered Reward Function (5x5 Gridworld):")
        print(np.round(reward_fxn_grid, 3))
        
        print(f"\nFinal Expected State Visitation Frequencies:")
        print(np.round(svf[:25].reshape(5,5), 3))

        # Save 3D Reward Plot for this specific LR
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
        surf = ax.plot_surface(X, Y, reward_fxn_grid, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Reward')
        plt.title(f'Recovered Reward Function (LR = {lr})')
        plt.savefig(f'reward_function_lr_{lr}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # NEW: Save 2D Reward Grid Overlay Plot for this specific LR
        plt.figure(figsize=(6,6))
        cmap = plt.get_cmap('Pastel1') if hasattr(plt, 'get_cmap') else plt.cm.get_cmap('Pastel1', 4)
        plt.imshow(grid_feats, cmap=cmap)
        for i in range(5):
            for j in range(5):
                plt.text(j, i, f"{reward_fxn_grid[i,j]:.2f}", ha='center', va='center', fontweight='bold', color='black')
        plt.title(f'2D Reward Map (LR = {lr})')
        plt.axis('off')
        plt.savefig(f'reward_function_2d_lr_{lr}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"======================================================")
    print(f"\n=> Best Learning Rate chosen: {best_lr}")

    # Plot Convergence Ablation
    plt.figure(figsize=(10,6))
    for lr in lrs: 
        plt.plot(results[lr]['grads'], label=f'LR = {lr}', linewidth=2)
    plt.title('Ablation: Gradient Norm vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm of Gradient (Feature Error)')
    plt.ylim(0, 7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_ablation_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Extract Data for Best LR
    best_w = results[best_lr]['weights']
    best_svf = results[best_lr]['svf']
    
    full_rewards = np.dot(sf, best_w)
    full_rewards[25] = 0 # Terminal state has 0 reward

    # ==========================================
    # PATH EXTRACTION USING VALUE ITERATION
    # ==========================================
    optimal_policy = value_iteration(trans_mat, full_rewards)
    
    action_symbols = {0: 'v', 1: '^', 2: '<', 3: '>'}
    grid_policy = []
    for s in range(25):
        if s == 24:
            grid_policy.append('G')
        else:
            grid_policy.append(action_symbols[optimal_policy[s]])
    grid_policy = np.array(grid_policy).reshape(5, 5)
    
    print("\n======================================================")
    print("--- AGENT NAVIGATION USING LEARNED REWARDS ---")
    print("Optimal Policy Grid (v: Down, ^: Up, <: Left, >: Right, G: Goal)")
    for row in grid_policy:
        print("  " + "  ".join(row))
        
    print("\nOptimal Paths from EVERY starting state to the Goal:")
    agent_paths = []
    for start_s in range(24):
        path = [start_s]
        curr = start_s
        steps = 0
        while curr != 24 and steps < 30: # Max 30 steps to prevent loops
            a = optimal_policy[curr]
            next_s = np.argmax(trans_mat[curr, a])
            curr = next_s
            path.append(curr)
            steps += 1
        agent_paths.append(path)
        print(f"Start State {start_s:2d} -> Path: {path}")
    print("======================================================\n")

    # 3. 3D Reward Plot for Best LR
    reward_fxn_grid = np.reshape(full_rewards[:25], (5,5))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
    surf = ax.plot_surface(X, Y, reward_fxn_grid, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Reward')
    plt.title(f'Recovered Reward Function (Optimal LR = {best_lr})')
    plt.savefig('reward_function_best_lr.png', dpi=300, bbox_inches='tight')
    plt.close()

    # NEW: 2D Reward Grid Overlay Plot for Best LR
    grid_feats = np.argmax(sf[:25], axis=1).reshape(5,5)
    plt.figure(figsize=(6,6))
    cmap = plt.get_cmap('Pastel1') if hasattr(plt, 'get_cmap') else plt.cm.get_cmap('Pastel1', 4)
    plt.imshow(grid_feats, cmap=cmap)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{reward_fxn_grid[i,j]:.2f}", ha='center', va='center', fontweight='bold', color='black')
    plt.title(f'2D Reward Map (Optimal LR = {best_lr})')
    plt.axis('off')
    plt.savefig('reward_function_2d_best_lr.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Expert Trajectory Plot (Overlay)
    plt.figure(figsize=(6,6))
    plt.imshow(grid_feats, cmap=cmap)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, str(i*5+j), ha='center', va='center', fontweight='bold')
            
    colors = ['red', 'blue', 'green', 'purple']
    for idx, d in enumerate(demos):
        pts = [(s//5, s%5) for s in d[:-1]] 
        y, x = zip(*pts)
        plt.plot(x, y, color=colors[idx], marker='o', alpha=0.6, linewidth=3, label=f'Demo {idx+1}')
        
    plt.title('Expert Demonstrations Overlay on Feature Map')
    plt.legend(loc='lower left')
    plt.axis('off')
    plt.savefig('trajectories_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Agent Learned Trajectories Plot (Separated per state)
    if not os.path.exists('agent_individual_paths'):
        os.makedirs('agent_individual_paths')

    for start_s, path in enumerate(agent_paths):
        plt.figure(figsize=(6,6))
        plt.imshow(grid_feats, cmap=cmap)
        for i in range(5):
            for j in range(5):
                plt.text(j, i, str(i*5+j), ha='center', va='center', fontweight='bold')
                
        # Plot the individual path
        pts = [(s//5, s%5) for s in path]
        y, x = zip(*pts)
        
        plt.plot(x, y, color='dodgerblue', marker='o', alpha=0.8, linewidth=4, label=f'Path from State {start_s}')
        plt.plot(x[0], y[0], color='red', marker='*', markersize=18, label='Start')
        
        plt.title(f'Agent Optimal Path (Start = {start_s})')
        plt.legend(loc='lower left')
        plt.axis('off')
        
        filename = f'agent_individual_paths/agent_optimal_path_{start_s:02d}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print("\n[SUCCESS] Generated files:")
    print("- maxent_irl_log.txt (Contains all console outputs)")
    print("- lr_ablation_convergence.png")
    print("- reward_function_best_lr.png")
    print("- reward_function_2d_best_lr.png")
    print("- trajectories_ablation.png")
    print("- 24 individual path images saved in the 'agent_individual_paths/' directory.")
    print("- 2D and 3D reward plots saved for all individual learning rates.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import numpy as np
# import numpy.random as rand
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import math

        
# def build_trans_mat_gridworld():
#   # 5x5 gridworld laid out like:
#   # 0  1  2  3  4
#   # 5  6  7  8  9 
#   # 9  10 11 12 13
#   # 14 15 16 17 18
#   # 20 21 22 23 24
#   # where 24 is a goal state that always transitions to a 
#   # special zero-reward terminal state (25) with no available actions
#   trans_mat = np.zeros((26,4,26))
  
#   # NOTE: the following iterations only happen for states 0-23.
#   # This means terminal state 25 has zero probability to transition to any state, 
#   # even itself, making it terminal, and state 24 is handled specially below.
  
#   # Action 0 = down
#   for s in range(24):
#     if s < 20:
#       trans_mat[s,0,s+5] = 1
#     else:
#       trans_mat[s,0,s] = 1
      
#   # Action 1 = up
#   for s in range(24):
#     if s >= 5:
#       trans_mat[s,1,s-5] = 1
#     else:
#       trans_mat[s,1,s] = 1
      
#   # Action 2 = left
#   for s in range(24):
#     if s%5 > 0:
#       trans_mat[s,2,s-1] = 1
#     else:
#       trans_mat[s,2,s] = 1
      
#  # Action 3 = right
#   for s in range(24):
#     if s%5 < 4:
#       trans_mat[s,3,s+1] = 1
#     else:
#       trans_mat[s,3,s] = 1

#   # Finally, goal state always goes to zero reward terminal state
#   for a in range(4):
#     trans_mat[24,a,25] = 1  
      
#   return trans_mat


# def build_state_features_gridworld():
#   # There are 4 features and only one is active at any given state, represented 1-hot vector at each state, with the layout as follows:
#   # 0 0 0 0 0
#   # 0 1 1 1 1
#   # 0 0 2 0 0
#   # 0 0 0 0 0 
#   # 0 0 0 0 4
#   # And the special terminal state (25) has all zero state features.

#   sf = np.zeros((26,4))  
#   sf[0,0] = 1
#   sf[1,0] = 1
#   sf[2,0] = 1
#   sf[3,0] = 1
#   sf[4,0] = 1
#   sf[5,0] = 1
#   sf[6,1] = 1
#   sf[7,1] = 1
#   sf[8,1] = 1
#   sf[9,1] = 1
#   sf[10,0] = 1
#   sf[11,0] = 1
#   sf[12,2] = 1
#   sf[13,0] = 1
#   sf[14,0] = 1
#   sf[15,0] = 1
#   sf[16,0] = 1
#   sf[17,0] = 1
#   sf[18,0] = 1
#   sf[19,0] = 1
#   sf[20,0] = 1
#   sf[21,0] = 1
#   sf[22,0] = 1
#   sf[23,0] = 1
#   sf[24,3] = 1
#   return sf


           
# def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index):
#   """
#   Implement steps 1-3 of Algorithm 1 in Ziebart et al.
  
#   For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
  
#   trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
#   horizon: the finite time horizon (int) of the problem for calculating state frequencies
#   r_weights: a size F array of the weights of the current reward function to evaluate
#   state_features: an S x F array that lists F feature values for each state in S
#   term_index: the index of the special terminal state
  
#   return: an S x A policy in which each entry is the probability of taking action a in state s
#   """
#   n_states = np.shape(trans_mat)[0]
#   n_actions = np.shape(trans_mat)[1]
#   policy = np.zeros((n_states,n_actions))  
#   return policy


  
# def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
#   """
#   Implement steps 4-6 of Algorithm 1 in Ziebart et al.
  
#   Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
#   trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
#   horizon: the finite time horizon (int) of the problem for calculating state frequencies
#   start_dist: a size S array of starting start probabilities - must sum to 1
#   policy: an S x A array array of probabilities of taking action a when in state s
  
#   return: a size S array of expected state visitation frequencies
#   """
  
#   n_states = np.shape(trans_mat)[0]
#   n_actions = np.shape(trans_mat)[1]
#   state_freq = np.zeros(n_states)
#   return state_freq
  


# def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index):
#   """
#   Implement the outer loop of MaxEnt IRL that takes gradient steps in weight space
  
#   Compute a MaxEnt reward function from demonstration trajectories
  
#   trans_mat: an S x A x S' array that describes transition probabilities from state s to s' if action a is taken
#   state_features: an S x F array that lists F feature values for each state in S
#   demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
#   seed_weights: a size F array of starting reward weights
#   n_epochs: how many times (int) to perform gradient descent steps
#   horizon: the finite time horizon (int) of the problem for calculating state frequencies
#   learning_rate: a multiplicative factor (float) that determines gradient step size
#   term_index: the index of the special terminal state
  
#   return: a size F array of reward weights
#   """
  
#   n_features = np.shape(state_features)[1]
#   r_weights = np.zeros(n_features)
#   return r_weights
  
 
 
# if __name__ == '__main__':
  
#   # Build domain, features, and demos
#   trans_mat = build_trans_mat_gridworld()
#   state_features = build_state_features_gridworld() 
#   demos = [[4,9,14,19,24,25],[3,8,13,18,19,24,25],[2,1,0,5,10,15,20,21,22,23,24,25],[1,0,5,10,11,16,17,22,23,24,25]]
#   seed_weights = np.zeros(4)
#   term_index = 25
  
#   # Parameters
#   n_epochs = 25
#   horizon = 15
#   learning_rate = 0.1
  
#   # Main algorithm call
#   r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index)
  
#   # Construct reward function from weights and state features
#   reward_fxn = []
#   for s_i in range(25):
#     reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
#   reward_fxn = np.reshape(reward_fxn, (5,5))
  
#   # Plot reward function
#   fig = plt.figure()
#   ax = fig.add_subplot(111, projection='3d')
#   X = np.arange(0, 5, 1)
#   Y = np.arange(0, 5, 1)
#   X, Y = np.meshgrid(X, Y)
#   surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
# 			linewidth=0, antialiased=False)
#   plt.show()

