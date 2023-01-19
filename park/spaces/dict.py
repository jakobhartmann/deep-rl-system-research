from park import core


class Dict(core.Space):
    """
    A dict of simpler spaces

    Example usage:
    self.observation_space = spaces.Dict({
        'space1': spaces.Discrete(2), 
        'space2': spaces.Discrete(3)
    })
    """
    def __init__(self, spaces):
        self.spaces = spaces
        core.Space.__init__(self, None, None)

    def __getitem__(self, key):
        return self.spaces[key]

    def sample(self):
        return dict([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, dict):
            x = dict(x)  # Promote list to tuple for contains check
        return isinstance(x, dict) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space,part) in zip(self.spaces,x))
