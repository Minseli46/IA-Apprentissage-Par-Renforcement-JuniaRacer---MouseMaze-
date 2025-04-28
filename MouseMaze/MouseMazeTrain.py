# MouseMazeTrain.py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from MouseMazeEnv import MouseMazeEnv  # Import from separate file

# Register the environment
gym.register(id="MouseMaze-v1", entry_point="MouseMazeEnv:MouseMazeEnv")
env = gym.make("MouseMaze-v1", size=8, is_slippery=True, render_mode="human")

# Train with MultiInputPolicy
model = DQN(
    "MultiInputPolicy", 
    env,
    learning_rate=1e-4,         # Taux d'apprentissage
    buffer_size=100000,         # Taille du replay buffer
    batch_size=32,              # Taille des mini-batchs
    exploration_fraction=0.1,   # Fraction de timesteps pour la décroissance d'epsilon
    exploration_final_eps=0.05, # Valeur finale d'epsilon (exploration minimale)
    gamma=0.99,                 # Facteur de discount
    verbose=1
)

model.learn(total_timesteps=1000000, progress_bar=True)
model.save("dqn_mouse_maze")

# Load and evaluate
model = DQN.load("dqn_mouse_maze", env=env)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if done:
        obs = env.reset()
env.close()