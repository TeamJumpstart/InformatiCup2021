import numpy as np
import time
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed
from state_representation import occupancy_map
from heuristics import ConstantHeuristic


class DLASTree():
    """Depth-Limited Action Search Tree """
    def __init__(self, action, occ_prob=0.0, depth=0, max_depth=3, parent=None, seed=None):
        self.action = action
        self.occ_prob = occ_prob
        self.depth = depth
        self.max_depth = max_depth

        self.parent = parent
        self.children = None

        # do not need to evaluate score if player is dead or player will be certainly dead
        self.evaluated = occ_prob >= 1.0
        # evaluate only final nodes, TODO: find good threshold for occ_prob
        self.final = depth >= max_depth or occ_prob >= 0.99

        self.rng = np.random.default_rng(seed)

    def expandNode(self, env, occ_map):
        """ Compute the probability of collision with opponenents per action
            and create child nodes based on the results.
        """
        # early out for final nodes
        if self.final:
            return

        self.evaluated = True
        self.children = []
        # randomise action traversal to explore different nodes
        for action in self.rng.permutation(spe_ed.actions):
            occ_prob = 1.0
            new_env = env.step([action])
            if new_env.players[0].active:
                # probability that all newly entered cells are free
                free_prob_action = np.prod(1 - occ_map[np.logical_xor(new_env.cells, env.cells)])
                # update probability (all cells free) with all previously entered cells (all cells free)
                all_free_prob = free_prob_action * (1 - self.occ_prob)
                # probability for at least one entered cell is occupied
                occ_prob = 1 - all_free_prob
            # create new child node
            self.children += [DLASTree(action, occ_prob, self.depth + 1, self.max_depth, self)]


class DepthSearchPolicy(Policy):
    """Depth-Limited Action Search Policy.
    Evaluates heuristic after a certain depth in the search tree hast bee reached.
    Tries to evaluate a single guided depth first search on each valid action.
    Node expansion is determined by the occupancy map. Implements a watch dog for an early out.
    """
    def __init__(
        self,
        base_heuristic,
        depth_heuristic,
        weights=None,
        depth=6,
        occupancy_depth=3,
        priority_heuristic=ConstantHeuristic(),
        seed=None
    ):
        """Initialize DepthSearchPolicy.

        Args:
            heuristic: `Heuristic` that will be evaluated on final nodes of the search tree.
            depth: defines the depth of the search tree.
        """
        self.rng = np.random.default_rng(seed)
        self.base_heuristic = base_heuristic
        self.depth_heuristic = depth_heuristic

        if weights is not None and len(weights) != 2:
            raise ValueError(f"Number of weights {weights} shold be 2")
        if weights is None:
            self.weights = np.ones(2)
        else:
            self.weights = weights / np.sum(weights)

        self.search_tree_depth = depth
        self.occupancy_map_depth = occupancy_depth
        self.priority_heuristic = priority_heuristic

        self.interrupt = 0
        self.explored = 0
        self.evaluated = 0

    def evaluateOneNode(self, init_env, occ_map, priority_queue):
        """Evaluate one node, given a priority queue and an initial environment.

        First, compute the environment of the node with the highest priority (backtracking to root node).
        Expand the node further in a depth-first fashion until reaching a final node or evaluating all children.
        Update the priority queue accordingly and return a score based on the given heuristic.

        Return:
            Score value of one final node and an updated priority queue
        """
        score = 0
        # early out, if no elements to expand
        if len(priority_queue) <= 0:
            return (score, priority_queue)

        # get node with highest priority
        node = priority_queue.pop(0)[1]

        actions = []
        n = node
        # backtrack actions to root node
        while n.parent is not None:
            actions += [n.action]
            n = n.parent
        # compute the current environment
        if len(actions) > 0:
            actions.reverse()
            env = init_env.step(actions)
        player = env.players[0]
        opponents = [p for p in env.players if p.active and p.player_id != player.player_id]

        # expand nodes until reach a final node
        while not node.final:
            node.expandNode(env, occ_map)
            child_queue = []
            for child in node.children:
                if not child.evaluated:
                    env = env.step([child.action])

                    try:
                        priority_value = self.priority_heuristic.score(env.cells, env.players[0], opponents, env.rounds)
                        priority_value *= 1 - child.occ_prob
                    except:
                        print(child.action, child.occ_prob, child.depth)
                        print(env.players[0])
                    self.explored += 1
                    child_queue += [(priority_value, child)]
                    env = env.undo()
            # sort children based on occupancy
            child_queue.sort(key=lambda pair: pair[0], reverse=True)

            # continue with highest priority child
            if len(child_queue) > 0:
                node = child_queue.pop(0)[1]
                env = env.step([node.action])
                # add remaining nodes to priority queue for later processing
                priority_queue.extend(child_queue)
            # all children evaluated (e.g. occ >= 1.0)
            else:
                node.final = True

        # evaluate heuristic on the final node
        if not node.evaluated and env.players[0].active:
            if len(opponents) > 0:
                # evaluate the heuristic
                score = self.depth_heuristic.score(env.cells, env.players[0], opponents, env.rounds)
                node.evaluated = True
                self.evaluated += 1

        # sort the priority queue by occupancy value
        priority_queue.sort(key=lambda pair: pair[0], reverse=True)

        # weighed with probability that all enetered cells are free
        score *= (1 - node.occ_prob)

        # return score and priority queue
        return (score, priority_queue)

    def act(self, cells, player, opponents, rounds):
        """ Perform Depth-Limited Action Search based on an occupancy map
            to choose actions based on weighted heuristic scores with occupancy map.
        """
        self.evaluated = 0
        self.explored = 5

        # self.interrupt = time.time() + np.random.randint(2, 15)  # time in seconds
        self.interrupt = time.time() + 3  # time in seconds

        scores = dict(zip(spe_ed.actions, [0.0 for _ in spe_ed.actions]))
        depth_scores = dict(zip(spe_ed.actions, [0.0 for _ in spe_ed.actions]))

        # create initial environment
        init_env = Spe_edSimulator(cells, [player] + opponents, rounds)
        # compute occupancy map
        occ_map = occupancy_map(cells, opponents, rounds, self.occupancy_map_depth)

        # create root node and expand it once
        root_node = DLASTree(None, 0.0, 0, self.search_tree_depth)
        root_node.expandNode(init_env, occ_map)

        # evaluate base heuristic on first-level nodes
        for node in root_node.children:
            # print(node.action, node.evaluated)
            if not node.evaluated:
                env = init_env.step([node.action])
                scores[node.action] = self.base_heuristic.score(env.cells, env.players[0], opponents, env.rounds)
                scores[node.action] *= np.prod(1 - occ_map[np.logical_xor(env.cells, init_env.cells)])

        # initialize priority queues
        priority_queues = {}
        for node in root_node.children:
            if not node.evaluated:
                priority_queues[node.action] = [(0.0, node)]
        # perform limited-depth search
        # iterate until time runs out or no more items in priority queues
        while self.interrupt > time.time() and len(priority_queues) > 0:
            # evaluate evenly one node for each valid action
            for action in priority_queues:
                # early out if time runs out
                if self.interrupt <= time.time():
                    break
                # evaluate one node
                score, new_queue = self.evaluateOneNode(init_env, occ_map, priority_queues[action])
                # update priority queue and scores
                priority_queues[action] = new_queue
                depth_scores[action] = max(depth_scores[action], float(score))
            # remove empty queues
            priority_queues = {action: queue for action, queue in priority_queues.items() if queue}

        print(f"Nodes: {str(self.evaluated)}/{self.explored}")
        print(f"Base Scores: {str(scores)}")
        print(f"Depth Scores: {str(depth_scores)}")
        # print(f"Action: {str(max(scores, key=scores.get))}")
        # print("Scores:", scores)
        # for action in priority_queues:
        #    print(str(action), np.unique([k for k, v in priority_queues[action]], return_counts=True))

        # weight base and depth scores accordingly
        for action in scores:
            scores[action] = self.weights[0] * scores[action] + self.weights[1] * depth_scores[action]

        # select action with the highest score
        return max(scores, key=scores.get)

    def __str__(self):
        """Get readable representation."""
        return f"DepthSearchPolicy[\
            search_tree_depth={str(self.search_tree_depth)}, \
            occupancy_map_depth={str(self.occupancy_map_depth)}, \
            heuristic={str(self.heuristic)}, \
            priority_heuristic={str(self.priority_heuristic)}]"

    def interrupt(self):
        self.interrupt = 0
