import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deep_q_model import build_deep_q_model
from models.dueling_deep_q_model import build_dueling_deep_q_model
from agents.agent import Agent
from typing import Callable
import gym
import numpy as np
from tqdm import tqdm
from keras.optimizers import Optimizer
from keras.optimizers import Adam
from models.losses import huber_loss
from base.annealing_variable import AnnealingVariable
from base.replay_queue import ReplayQueue
from base.prioritized_replay_queue import PrioritizedReplayQueue
from keras.models import load_model


class DeepQAgent(Agent):

    def __init__(self,
        env: gym.Env,
        render_mode: str=None,
        replay_memory_size: int=750000,
        prioritized_experience_replay: bool=False,
        discount_factor: float=0.8,
        update_frequency: int=4,
        optimizer: Optimizer=Adam(lr=2e-5),
        exploration_rate: AnnealingVariable=AnnealingVariable(1., .1, 500000),
        loss: Callable=huber_loss,
        target_update_freq: int=10000,
        dueling_network: bool=False,
    ) -> None:
        """
        Initialize a new Deep Q Agent

        Args:
            env: environment for the agent to experience
            render_mode:
                - None: don't render
                - 'human': render in a window
            replay_memory_size: the number of previous experiences to store
                                in the exp replay queue
            discount_factor: discount factor, y, for discounting future reward
            update_frequency: the number of actions between updates to the
                              deep Q network from replay memory
            optimizer: the optimization method to use on the CNN gradients
            exploration_rate: the exploration rate, e, expected as an decaying value
            loss: the loss method to use at the end of the CNN
            target_update_freq: frequencty to update the target network (steps)
            dueling_network: whether to use the dueling architecture

        Returns:
            None
         """
        super().__init__(env, render_mode)
        self.queue = ReplayQueue(replay_memory_size)
        self.prioritized_experience_replay = prioritized_experience_replay
        if prioritized_experience_replay:
            self.queue = PrioritizedReplayQueue(replay_memory_size)
        else:
            self.queue = ReplayQueue(replay_memory_size)
        self.update_frequency = update_frequency
        self.discount_factor = discount_factor
        self.optimizer = optimizer
        self.exploration_rate = exploration_rate
        self.loss = loss
        self.target_update_freq = target_update_freq
        self.dueling_network = dueling_network
        mask_shape = (1, env.action_space.n)
        self.mask = np.ones(mask_shape, dtype=np.float32)
        self.action_onehot = np.eye(env.action_space.n, dtype=np.float32)
        if dueling_network:
            build_model = build_dueling_deep_q_model
        else:
            build_model = build_deep_q_model

        self.model = build_model(
            image_size=env.observation_space.shape[:2],
            num_frames=env.observation_space.shape[-1],
            num_actions=env.action_space.n,
            loss=loss,
            optimizer=optimizer
        )

        self.target_model = build_model(
            image_size=env.observation_space.shape[:2],
            num_frames=env.observation_space.shape[-1],
            num_actions=env.action_space.n,
            loss=loss,
            optimizer=optimizer
        )

    def __repr__(self) -> str:
        _REPR_TEMPLATE = """
{}(
    env={},
    render_mode={}
    replay_memory_size={},
    prioritized_experience_replay={},
    discount_factor={},
    update_frequency={},
    optimizer={},
    exploration_rate={},
    loss={},
    target_update_freq={},
    dueling_network={}
)
""".lstrip()
        return _REPR_TEMPLATE.format(
            self.__class__.__name__,
            self.env,
            repr(self.render_mode),
            self.queue.size,
            self.prioritized_experience_replay,
            self.discount_factor,
            self.update_frequency,
            self.optimizer,
            self.exploration_rate,
            self.loss.__name__,
            self.target_update_freq,
            self.dueling_network
        )

    def _td_error(self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        d: np.ndarray,
        s2: np.ndarray
    ) -> float:
        """
        Calculate the TD-error for a single experience.

        Args:
            s: the current state
            a: the action to get from current state "state" to next state "state2"
            r: the reward resulting from taking action "action" in state "state"
            d: the flag denoting whether the episode ended after action "action"
            s2: the next state from taking action "action" in state "state"

        Returns:
            the TD-error as a result of the experience
        """
        if d:
            # terminal states have a Q value of zero by definition
            Q_t = 0.0
        else:
            # predict Q values for the next state and take the max value
            Q_t = self.target_model.predict([s2[None, :, :, :], self.mask])

        # calculate the predicted Q value from the current state and action
        Q = self.model.predict([s[None, :, :, :], self.mask])
        # calculate the TD error based on the reward, discounted future
        # reward, and the predicted future reward
        td_error = abs(r + self.discount_factor * np.max(Q_t) - np.max(Q))
        return td_error

    def _remember(self,
        s: np.ndarray,
        a: int,
        r: int,
        d: bool,
        s2: np.ndarray,
    ) -> None:
        """
        Push an experience onto the replay queue

        Args:
            s: the current state
            a: the action to get from current state "state" to next state "state2"
            r: the reward resulting from taking action "action" in state "state"
            d: the flag denoting whether the episode ended after action "action"
            s2: the next state from taking action "action" in state "state"

        Returns:
            None
        """
        if self.prioritized_experience_replay:
            # calculate the priority of the experience based on the TD error
            priority = self._td_error(s, a, r, d, s2)
            self.queue.push(s, a, r, d, s2, priority=priority)
        else:
            self.queue.push(s, a, r, d, s2)

    def _replay(self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        d: np.ndarray,
        s2: np.ndarray
    ) -> float:
        """
        Train the network on a mini-batch of replay data.

        Args:
            s: the current state
            a: the action to get from current state "state" to next state "state2"
            r: the reward resulting from taking action "action" in state "state"
            d: the flag denoting whether the episode ended after action "action"
            s2: the next state from taking action "action" in state "state"

        Returns:
            the loss as a result of training
        """
        # init target y values
        y = np.zeros((len(s), self.env.action_space.n), dtype=np.float32)

        # predict Q values for the next state of each memory in the batch and
        # take the max value. don't mask any outputs, i.e. use ones
        mask = np.repeat(self.mask, len(s), axis=0)
        Q = np.max(self.target_model.predict_on_batch([s2, mask]), axis=1)
        # terminal states have a Q value of zero by def.
        Q[d] = 0
        y[range(y.shape[0]), a] = r + self.discount_factor * Q

        return self.model.train_on_batch([s, self.action_onehot[a]], y)

    def observe(self, replay_start_size: int=50000) -> None:
        """
        Observe random moves to initialize the replay memory.

        Args:
            replay_start_size: the number of random observations to make

        Returns:
            None
        """
        progress = tqdm(total=replay_start_size, unit='frame')

        while replay_start_size > 0:
            # reset the game and get the init. state
            state = self._initial_state()
            # done flag indicating that an episode has ended
            done = False

            while not done:
                # sample a random action to perform
                action = self.env.action_space.sample()
                # perform action and observe the reward and next state
                next_state, reward, done = self._next_state(action)
                # push the memory onto the ReplayQueue
                self._remember(state, action, reward, done, next_state)
                # set the state to the new state
                state = next_state
                replay_start_size -= 1
                progress.update(1)
        progress.close()


    def predict(self, frames: np.ndarray, exploration_rate: float) -> int:
        """
        Predict an action from a stack of frames.

        Args:
            frames: the stack of frames to predict Q values from
            exploration_rate: the exploration rate for epsilon greedy selection

        Returns:
            the predicted optimal action based on the frames

        """
        if np.random.random() < exploration_rate:
            return self.env.action_space.sample()

        frames = frames[np.newaxis, :, :, :]
        actions = self.model.predict([frames, self.mask])
        return np.argmax(actions)

    def save_model(self, frames,):
        self.model.save("model.h5")
        self.target_model.save("target_model.h5")
        with open("state.txt", "w") as f:
            data = "{}\n{}\n{}".format(
                frames,
                self.exploration_rate.__repr__(),
                self.exploration_rate.value
            )
            f.write(data)

    def _load_model(self):
        del self.model
        del self.target_model
        self.model = load_model("model.h5")
        self.target_model = load_model("target_model.h5")
        with open("state.txt", "r") as f:
            data = f.read()
        data = data.split("\n")
        frames = data[0]
        exec("self.exploration_rate = " + data[1])
        self.exploration_rate.value = float(data[2])


    def train(self,
        frames_to_play: int=50000000,
        batch_size: int=32,
        callback: Callable=None,
    ) -> None:
        """
        Train the network for a number of episodes (games).
        Args:
            frames_to_play: the number of frames to play the game for
            batch_size: the size of the replay history batches
            callback: an optional callback to get updates about the score,
                      loss, discount factor, and exploration rate every
                      episode
        Returns:
            None
        """
        progress = tqdm(total=frames_to_play, unit="frame")
        progress.set_postfix(score="?", loss="?", rate="?")
        self._load_model()

        while frames_to_play > 0:
            done = False
            score = 0
            loss = 0
            frames = 0

            state = self._initial_state()

            while not done:
                action = self.predict(state, self.exploration_rate.value)
                self.exploration_rate.step()
                next_state, reward, done = self._next_state(action)
                score += reward
                self._remember(state, action, reward, done, next_state)
                state = next_state
                frames_to_play -= 1
                frames += 1

                if frames_to_play % self.update_frequency == 0:
                    loss += self._replay(*self.queue.sample(size=batch_size))
                if frames_to_play % self.target_update_freq == 0:
                    self.target_model.set_weights(self.model.get_weights())
            if callable(callback):
                callback(self, score, loss)
            self.save_model(frames)

            progress.set_postfix(score=score, loss=loss, rate=self.exploration_rate.value)
            progress.update(frames)

        progress.close()

    def play(self, games: int=100, exploration_rate: float=0.05) -> np.ndarray:
        """
        Run the agent without training for the given number of games.
        Args:
            games: the number of games to play
            exploration_rate: the epsilon for epsilon greedy exploration
        Returns:
            an array of scores, one for each game
        """
        progress = tqdm(range(games), unit='game')
        progress.set_postfix(score='?')

        scores = np.zeros(games)

        for game in progress:
            done = False
            score = 0
            state = self._initial_state()

            while not done:
                action = self.predict(state, exploration_rate)
                next_state, reward, done = self._next_state(action)
                score += reward
                state = next_state

            scores[games] = score
            progress.set_postfix(score=score)
            progress.update(1)
        progress.close()

        return scores
