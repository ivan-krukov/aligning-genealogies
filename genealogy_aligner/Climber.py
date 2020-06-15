from collections import defaultdict

from .Genealogical import Genealogical


class Climber:

    def __init__(self, gen, source):
        self.gen = gen
        self.depth = gen.infer_depth()
        self.depth_map = defaultdict(list)
        self.depth_map[0] = source
        self.generations = max(self.depth.values())

    def queue(self, node):
        node_depth = self.depth[node]
        self.depth_map[node_depth].append(node)

    def __iter__(self):
        for t in range(self.generations):
            for node in self.depth_map[t]:
                parents = list(self.gen.graph.predecessors(node))
                yield node, parents
