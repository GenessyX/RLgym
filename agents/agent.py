import numpy as np
import gym


class Agent(object):
    def __init__(self, env, render_mode: str=None):
        """
        Args:
            env: environment
            render_mode:
                None: don't render
                human: render in a window
        """
        self.env = env
        self.render_mode = render_mode

    def __repr__(self):
        return "{}(env={}, render_mode={})".format(
            self.__class__.__name__,
            self.env,
            self.render_mode
        )

    def _initial_state(self) -> np.ndarray:
        """
            Reset env and return initial state
        """
        state = self.env.reset()
        if self.render_mode is not None:
            self.env.render(mode=self.render_mode)

        return state

    def _next_state(self, action: int) -> tuple:
        """
            Return next state based on the given action
            Args:
                action: the action to perform for some frame

            Returns:
                tuple of:
                    - next state
                    - reward as a result of the action
                    - flag determining end of episode
                    - additional info
        """
        state, reward, done, info = self.env.step(action=action)
        if self.render_mode is not None:
            self.env.render(mode=self.render_mode)

        return state, reward, done

__all__ = [Agent.__name__]

if __name__ == "__main__":
    test = Agent("env", None)
    print(test)
