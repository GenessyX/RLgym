class AnnealingVariable(object):

    def __init__(self,
        initial_value: float,
        final_value: float,
        steps: int
    ) -> None:
        """
        Args:
            initial_value: starting value
            final value: stopping value
            steps: number of steps
        Returns:
            None
        """
        initial_value = float(initial_value)
        final_value = float(final_value)
        steps = int(steps)

        self.initial_value = initial_value
        self.final_value = final_value
        self.value = initial_value
        self.steps = steps
        self.rate = (final_value / initial_value)**(1.0 / steps)

        if abs(self.rate) > 1:
            self.bound = min
        else:
            self.bound = max

    def __repr__(self) -> str:
        return "{}(initial_value={}, final_value={}, steps={})".format(
            self.__class__.__name__,
            self.initial_value,
            self.final_value,
            self.steps
            )

    def step(self) -> None:
        self.value = self.bound(self.value*self.rate, self.final_value)

if __name__ == "__main__":
    test = AnnealingVariable(1,0.1,10000)
    print(test)
    var = exec("test1 = " + test.__repr__())
    print(var)
    print(test1)
