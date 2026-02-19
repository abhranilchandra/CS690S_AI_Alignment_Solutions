import gymnasium as gym
import numpy as np

def run_strategy(env, strategy_name, num_episodes=100):
    rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            position, velocity = obs
            
            # --- STRATEGY LOGIC ---
            if strategy_name == "1. Right -> Left -> Right":
                # Simulate human trying to power through, realizing it fails, 
                # backing up, and trying again.
                if step_count < 20:
                    action = 2 # Right
                elif step_count < 50:
                    action = 0 # Left
                else:
                    action = 2 # Right
                    
            elif strategy_name == "2. Left -> Right":
                # Simulate human deliberately backing up first, then gunning it.
                # 35 steps is roughly the sweet spot for the backswing.
                if step_count < 35:
                    action = 0 # Left
                else:
                    action = 2 # Right

            elif strategy_name == "3. Oscillate (L, R, L, R...)":
                # Simulate human rocking back and forth blindly
                # Switch every 15 steps
                if (step_count // 15) % 2 == 0:
                    action = 0 # Left
                else:
                    action = 2 # Right

            elif strategy_name == "4. Optimal Programmatic Expert":
                # The perfect feedback-based policy we used in BCO
                if velocity > 0:
                    action = 2 # Right
                elif velocity < 0:
                    action = 0 # Left
                else:
                    action = 0 # Kickstart
            
            # ----------------------

            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += rew
            step_count += 1
            
        rewards.append(total_reward)
        
    return rewards

def main():
    env = gym.make('MountainCar-v0')
    strategies = [
        "1. Right -> Left -> Right",
        "2. Left -> Right",
        "3. Oscillate (L, R, L, R...)",
        "4. Optimal Programmatic Expert"
    ]
    
    print(f"{'='*80}")
    print(f"{'STRATEGY PERFORMANCE ANALYSIS (100 Trials Each)':^80}")
    print(f"{'='*80}")
    print(f"{'Strategy Name':<35} | {'Avg Reward':<12} | {'Max Reward':<12} | {'Min Reward':<12}")
    print(f"{'-'*80}")
    
    for strat in strategies:
        rewards = run_strategy(env, strat)
        print(f"{strat:<35} | {np.mean(rewards):<12.2f} | {np.max(rewards):<12.1f} | {np.min(rewards):<12.1f}")
    
    print(f"{'='*80}")
    env.close()

if __name__ == "__main__":
    main()
