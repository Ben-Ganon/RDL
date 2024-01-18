class Params:
    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 0.01)
        self.discount = kwargs.get('discount', 0.99)
        self.greedy_epsilon = kwargs.get('greedy_epsilon', 0.8)
        self.greedy_decay = kwargs.get('greedy_decay', 0.99)
        self.epochs = kwargs.get('epochs', 1000)
        self.greedy_min = kwargs.get('greedy_min', 0.1)

    def decay_greedy(self):
        if self.greedy_epsilon > self.greedy_min:
            self.greedy_epsilon *= self.greedy_decay