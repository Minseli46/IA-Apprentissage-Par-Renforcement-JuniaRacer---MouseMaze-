# MouseMazeTest.py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from MouseMazeEnv import MouseMazeEnv

# Register the environment
gym.register(id="MouseMaze-v1", entry_point="MouseMazeEnv:MouseMazeEnv")

# Create and wrap the environment
env = gym.make("MouseMaze-v1", size=8, is_slippery=True, render_mode="human")
env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv for SB3 compatibility

# Load the trained model
model = DQN.load("dqn_mouse_maze", env=env)

# Test with metrics
obs = env.reset()  # Returns the observation dict directly
total_reward = 0
episodes = 0
goals_reached = 0
holes_fallen = 0
rewards_collected = 0
max_steps = 1000

for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)  # action is a numpy array
    action = int(action)  # Convert to scalar integer
    obs, reward, done, info = env.step([action])  # Pass action as a list
    # Extract scalars from arrays (single env)
    reward = reward[0]
    done = done[0]
    total_reward += reward

    if reward == 0.5:
        rewards_collected += 1
    elif reward == 1:
        goals_reached += 1

    if done:
        episodes += 1
        if reward == 0:  # Assume hole if done with no reward
            holes_fallen += 1
        obs = env.reset()

env.close()

# Print results
print(f"Tested for {max_steps} steps across {episodes} episodes:")
print(f"Total reward: {total_reward}")
print(f"Goals reached: {goals_reached} ({goals_reached/episodes*100:.1f}%)")
print(f"Holes fallen into: {holes_fallen} ({holes_fallen/episodes*100:.1f}%)")
print(f"Rewards collected: {rewards_collected}")