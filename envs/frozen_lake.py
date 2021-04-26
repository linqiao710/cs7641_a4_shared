import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFHHH",
        "FHFHFFFF",
        "FFFFFFHF",
        "FFFFFFHF",
        "FHFFFHFF",
        "FFHFFFFF",
        "FFFFHFFH",
        "FFFFFFFG"
    ],
    "10x10": [
        "SFFHFHFFFH",
        "FFFFFFFFHF",
        "FFFFFFFHFF",
        "HFFFFFFFFH",
        "FHFFFHFFHF",
        "FFFFFFFFFF",
        "FFFFFHFHHF",
        "FFHFFFFFFF",
        "FFFFFFFFHF",
        "FFFFFFFHFG"
    ],
    "16x16": [
        "SFFFFHFFFFFFHFFF",
        "FHHFHFFFFFFFFFFF",
        "FFFFHFFFFFFFFHFF",
        "FFFFFFFFFFFHFFFH",
        "FFFFHFFFFFFHFFFF",
        "FFFFFHFFFFHFFFFH",
        "HFFFFHHFFFFFHFHF",
        "FFHFFFFHFFFFFFFF",
        "HFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFHF",
        "FFHFFFFFFFHHFFFF",
        "FFFFFFFHFFFFHFFF",
        "FHHFFHFFFFFFFFFF",
        "FFFFHFHHFFFFFHHF",
        "HFHFFFFFFHFFFFFF",
        "HFFFFFFHFFFFFFFG"
    ],
    '20x20': [
        "SFFFFFFHHHFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFHHFF",
        "FFFHFFFFFFFHHFFFFFFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFHFFFFFFFHHFF",
        "FFFFFHFFFFHHFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFFFFHHHHHHHFF",
        "HHHHFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFHHHFFFHHFF",
        "FFFFFFFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFHFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFFFFHFFFFFFFF",
        "FHHFFFHFFFFHFFFFFHFF",
        "FHHFHFHFFFFFFFFFFFFF",
        "FFFHFFFFFHFFFFHHFHFG"
    ],
    '21x21':[
        "SFHFFFHFFFFFFFHFFFHFF",
        "FFFFFFFFFFHHFFFFHHHHH",
        "FHFFFFFFFFFHFFFFFFFFF",
        "FHHFFFFHFFHFFFHFHFFFF",
        "FFFFFHFHFFFFHHFFFFFFF",
        "FFFFFFFHFFFHFFFFFFFFH",
        "HFFFHFFFFFFFFFFHFFFFH",
        "HFHFFFFFFFFFFFFFFFFFF",
        "HFFFHFFHFFFFFFFFFFFFF",
        "FFFFFFFFFFFHFFFFFHFFF",
        "FFFFFFFHFFFFFHFFFHFFH",
        "FFHFFHFFFHFFFFHFFFFFF",
        "FFFFFFFFFFHFFFFHFFFHF",
        "HFFFHFFFHHHFFFHFHHFFF",
        "FFFFFFFFFFFFFHHFFFFFH",
        "HFHFFFFHFFHHFHFHHFFHH",
        "FFFFFFHFFFFFFFFHFFFFF",
        "FFFFFFHFHFHFFFFFFHFFH",
        "FFFFFFHFFFHFFFHFFFFFF",
        "FFHFFFFFFFFFFFFFFFFHF",
        "HFFFFFFFHFFFHFFFFFFFG"

    ],
    '25x25': [
        "SFFFFFFFFFFFFHFFFFFFFFHFF",
        "HFFHFHFHFFFHFFFFFHHHFFFFF",
        "FFFFFFHFFFFFHFHFHFHFHHFFF",
        "HFFFHFHFFFFFFFHHHHFFFFFHF",
        "FFHFFFHFFHFFFFFFHFFFFFFHF",
        "HFHFFFFFFFHFFFFHHFFFFFFFF",
        "FHHFFFFHHFFHHFHFFFFFFFFFF",
        "FFFHFHFFFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFHFFFFFHFFFFFH",
        "FFFFFFFFFFFFFFFFHFFFFFFFF",
        "FFFFFHFFHFFFFFFHFFFFFFFFF",
        "FFFFFFFFHFFFFFFHFFHFHFFHF",
        "FFFHHFFHFFFFFFFHFHFFFHFFH",
        "FFHHFFFFFHFFFHFFFFFFHFFHF",
        "FFHFFHFFFFFFFFHFHFFFFFFFF",
        "FFFFHHFFFFFFFFHFFHFFHFFHH",
        "HFHHFFFFHFFFFFFHHFFFFFFFF",
        "FFFFFFFFFFFFFFFFFHFFFFHFF",
        "FFHFHHFFFHFFHFHFHFFFFFFHF",
        "FFFFFFHFFHFFFHHFFFFFHFFFF",
        "FHFFFFFFFHHFFFFFHFFHFHFFF",
        "FHFFFFHFHHFFFFFFFFHFFFFFH",
        "FFFFFFFFFHFFHFFFFFFFFFHFF",
        "FFFFFFFHFFFFHFFFFFFFFHFFF",
        "FFFHHFFFFFFHFFFHHFFFHFFFG"
    ],
    '26x26': [
        "SFFFFFFHHFHHFFFFFFFFHFFHFF",
        "FFFFFHFFFFFFFHFFFFFFFFFFFH",
        "HFHFFHHFFHFHFFFFHHFFFFFHFF",
        "FFHFFFFFHFHFFFFFFFFFFFHFHF",
        "FFFFFFFFFFHFHFHFFFHFFFFFFF",
        "HFFFFHFFFHFHFFFFFFFHFFFFHF",
        "HFFFFHFFFHFFHFFFFFHFFFFHFF",
        "FFHFFHFHFFFHFHFHFFFFFFFFHH",
        "FFFFFFFFHFFFHFFFFFHHFFFHHF",
        "FHFFFFHFFFFHFFHHFFFFFFFFFF",
        "HFFFFFFFFFFFHHFFFHFFFFFFHH",
        "FFFFFFFHFFFFFFFFFFFFFHFFHF",
        "FFHFFFFFFFHFFFFFFFFFFFHFFF",
        "FFFFFFHFFFFFHHFHFFHHFFFFFF",
        "FFFFHFHHHFFFFFFFFFHHFFFFFF",
        "FFFFFFFFFFFHHFFHFFFHFFFFHF",
        "HHFFFHFFFHFFFFFHFFFFFHFFFF",
        "FFFFFFFFFFFFHHFFFFFHFFHHFF",
        "HFFFFHFFFHHHFFFHFFFFHFFHHF",
        "FFFFFFFHFFFFFFFHFFFFFFFHFF",
        "FFFFHFFHFFFFFFFFFFFFFFFFFF",
        "FFFFFHFHFFFFFHHHFFHHFFFHFF",
        "HFHFFFFFFFFFHFHFFHFFFHFFFF",
        "FFFHFFFFHHFFFHFFHHFFHFFFFF",
        "FHHHFFFFFHFHFHFFFHFFHHFFFF",
        "HFFFFFFFFFFFFFFFFFFHFHHHFG"
    ],
    '30x30': [
        "SFFFHFFFFFFFFFFHFFFHHFHHFFFFFF",
        "FFHFFFFFFFFFHFHFFHFFFHHFFFFHFH",
        "FHFFFFFFFFHFFFFFFFHFFFFFFFFFFF",
        "FFFFFFFFHFFFFFFFFFHFHFFFHFFFFH",
        "FFFFFFFFFFFFHFFFFFFFFHFFFFHFFF",
        "HHFFFFFFFHFHHFFHFFHFFFFFHFFHFF",
        "FHFFFHFFFFHFFFHFHFFFFHFFFHFFHH",
        "FFFFFFFFFFFFHFFFFFHHFFFHFFHFFF",
        "HHFHFHFFFFFFFFFHFFHFFFFFFHHHFF",
        "FFFFFFFFFHFFFFFFFHHFHFFFHFFFFF",
        "FFFFHFFFFFHHHFFFFFFFHFFFFFFFFF",
        "FHHHFFFFFFFFFFFFFHHFFFHFFFFFFH",
        "HFHFFFFFHFFFHFFFHHFFFFFFFFFFFF",
        "FFFFHFFFHHHFFFFFFHFFFFFFFFFFFF",
        "FFHFFHFFFFFFFHFHFFFFHFFFFHFHFF",
        "HHHFHHHFFHFFFFHFFHFHFFFHFFFFFF",
        "HFFFFFFFHHHFFFFFHFFFHFHFFFFFFF",
        "FFFHFFFFFFHFFFFFFFFFHFFFFHFFFF",
        "FFFHFFFHFFFFFFFHFFHFFFFFFFFHFF",
        "FFFFFFFFFFFFFHHHFFHFFFFFFFHHFF",
        "HFHFHFFFFFFFFFFFFFFFFFFFFFFFHF",
        "FHFFHFFFFHFFFFHFHFHHFHHFFFFFFF",
        "HFFHFFFHFFFFFFHFFHFFHFFFFFFFFF",
        "HHFFFFFFFFFFFHFFFFHHHFFFHFHFFF",
        "HFFFFHFFHFHFFFFFHFHFHFFFFFHFFF",
        "HFHFFFFHFFFHFFHFHFFFFFFFHFFFFF",
        "FFFHFFFFFFHFHHHFFFFFFFHFFFFFFF",
        "FFFFFFFHFHFFHFFFFHFFHFHFHHHFFH",
        "FHHFFHFFFFFFHFHFFFFFHHFFFFFFFF",
        "FHFFHFFFHFFFFFHFFFFFHFFFFFFFHG"
    ]
}


def generate_random_map(size=6, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1 - p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GH'
            reward = float(newletter == b'G')
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append((
                                    1. / 3.,
                                    *update_probability_matrix(row, col, b)
                                ))
                        else:
                            li.append((
                                1., *update_probability_matrix(row, col, a)
                            ))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][int(self.lastaction)]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def get_color(self, x, y):
        return self.colors()[self.desc[x, y]]

    def get_symbol(self, x, y, policy):
        if self.desc[x, y] == b'G':
            return '✯'
        elif self.desc[x, y] == b'H':
            return None
        else:
            return self.directions()[policy[x, y]]

    def get_value(self, x, y, value):
        if self.desc[x, y] == b'G':
            return '1.0'
        else:
            return np.round(value[x, y], 2)

    @staticmethod
    def colors():
        return {
            b'S': 'lightgrey',
            b'F': 'white',
            b'H': 'brown',
            b'G': 'mediumseagreen',
        }

    @staticmethod
    def directions():
        return {
            3: '⬆',
            2: '➡',
            1: '⬇',
            0: '⬅'
        }


if __name__ == "__main__":
    map = generate_random_map(size=21, p=0.8)
    for row in map:
        print('"{}",'.format(row))
