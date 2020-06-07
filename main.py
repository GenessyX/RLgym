from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.deep_q_agent import DeepQAgent
from gym.wrappers import ResizeObservation
from max_frameskip_env import MaxFrameskipEnv
from reward_cache_env import RewardCacheEnv
from penalize_death_env import PenalizeDeathEnv


env = gym.make('Breakout-v0')

#env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v1')
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = RewardCacheEnv(env)
env = ResizeObservation(env, (64,64))
#env = PenalizeDeathEnv(env)
#env = MaxFrameskipEnv(env)


#env = gym.make('SpaceInvaders-v0')

agent = DeepQAgent(env=env,render_mode='human', dueling_network=True, prioritized_experience_replay=True)
print(agent)
#print(agent.model.summary())
#agent.train()
#"""
#agent.model.load_weights("model.h5")

"""
print(env.observation_space)

done = True
for step in range(50000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(agent.predict(state, 0.05))
    env.render()
#"""
