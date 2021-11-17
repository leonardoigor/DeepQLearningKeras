import gym
import numpy as np
from dddqn_keras import DQNAgent


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(
        alpha=0.0005,
        gamma=0.99,
        n_actions=env.action_space.n,
        epsilon=1.0,
        batch_size=64,
        input_dims=env.observation_space.shape[0],
        # input_dims=8,
    )
    # agent.load_models()
    agent.load_model()
    n_games = 501
    scores = []
    eps_history = []

    # env = gym.wrappers.Monitor(env, './videos/', force=True)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            if i > 100:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        if i % 10 == 0 and i > 0:
            agent.save_model()
        print('episode ', i, 'score %.1f' % score, 'average score %.1f' %
              avg_score, 'epsilon %.2f' % agent.epsilon)
