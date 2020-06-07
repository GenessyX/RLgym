"""A queue for storing previous experiences to sample from."""


import numpy as np

class ReplayQueue(object):

    def __init__(self, size: int) -> None:
        """
        Initialize replay buffer with a given size

        Args:
            size: the size of the replay buffer

        Returns:
            None
        """
        self.queue = [None] * size
        self.index = 0
        self.top = 0

    def __repr__(self) -> str:
        return "{}(size={})".format(self.__class__.__name__, self.size)

    @property
    def size(self) -> int:
        return len(self.queue)

    def push(self,
        state: np.ndarray,
        action: int,
        reward: int,
        done: bool,
        state2: np.ndarray,
    ) -> None:
        """
        Push a new experience onto the ReplayQueue

        Args:
            state: the current state
            action: the action to get from current state "state" to next state "state2"
            reward: the reward resulting from taking action "action" in state "state"
            done: the flag denoting whether the episode ended after action "action"
            state2: the next state from taking action "action" in state "state"

        Returns: None


        """
        # push variables onto the queue
        self.queue[self.index] = state, action, reward, done, state2
        # increment the index
        self.index = (self.index + 1) % self.size
        # incerement the top pointer
        if self.top < self.size:
            self.top += 1

    def sample(self, size: int=32) -> tuple:
        """
        Return a random sample of items from the queue

        Args:
            size: the number of items to sample and return

        Returns:
            A random sample from the queue sampled uniformly
        """
        s = [None] * size
        a = [None] * size
        r = [None] * size
        d = [None] * size
        s2 = [None] * size
        for batch, sample in enumerate(np.random.randint(0, self.top, size)):
            _s, _a, _r, _d, _s2 = self.queue[sample]
            s[batch] = np.array(_s, copy=False)
            a[batch] = _a
            r[batch] = _r
            d[batch] = _d
            s2[batch] = np.array(_s2, copy=False)

        return (
            np.array(s),
            np.array(a, dtype=np.uint8),
            np.array(r, dtype=np.int8),
            np.array(d, dtype=np.bool),
            np.array(s2),
        )
