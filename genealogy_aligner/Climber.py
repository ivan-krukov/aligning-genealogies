from collections import defaultdict
from copy import copy

from .Genealogical import Genealogical


class Climber:

    def __init__(self, gen, source):
        self.gen = gen
        self.depth = gen.infer_depth(forward=False)
        self.depth_map = defaultdict(set)
        self.depth_map[0] = set(copy(source))
        self.generations = max(self.depth.values())

    def queue(self, node):
        node_depth = self.depth[node]
        self.depth_map[node_depth].add(node)

    def __iter__(self):
        for t in range(self.generations):
            for node in self.depth_map[t]:
                parents = list(self.gen.graph.predecessors(node))
                yield node, parents
