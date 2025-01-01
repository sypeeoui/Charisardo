from copy import deepcopy

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
from multiprocessing import Pool

from policies.WeightedGreedy import get_greedy, WeightedGreedy
from utils import MyPkmEnv

class Node:

    def __init__(self):
        self.move: int = None
        self.env: MyPkmEnv = None
        self.parent: Node = None
        self.depth = 0
        self.eval = 0.0

class PrunedTreeSearch(BattlePolicy):
    """
    Agent that uses Pruned Tree Search to select actions.
    """

    def __init__(self, weights: list[float] = [1.0, 1.0], max_depth: int = 2, instances: int = 50, parallel: bool = True, debug: bool = True):
        self.weights = weights
        # self.min_depth = min_depth
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.max_depth = max_depth
        self.instances = instances
        self.parallel = parallel
        self.debug = debug

    def game_state_eval(self, g: PkmBattleEnv):
        """
        Evaluates the current game state of a Pokémon battle.
        This function calculates the health points (HP) of both the player's team and the opponent's team,
        and returns a score based on the ratio of current HP to maximum HP for both teams. The score is 
        weighted by the `self.weights` attribute.
        Args:
            g (PkmBattleEnv): The current game environment, which includes the teams and their statuses.
        Returns:
            float: A score representing the evaluation of the game state. Returns -1 if the player's team 
                   has no remaining HP, 1 if the opponent's team has no remaining HP, or a weighted score 
                   based on the HP ratios otherwise.
        """

        my_team = g.teams[0]
        opp_team = g.teams[1]
        my_sum_hp = my_team.active.hp + sum([p.hp for p in my_team.party])
        opp_sum_hp = opp_team.active.hp + sum([p.hp for p in opp_team.party])
        my_sum_max_hp = my_team.active.max_hp + sum([p.max_hp for p in my_team.party])
        opp_sum_max_hp = opp_team.active.max_hp + sum([p.max_hp for p in opp_team.party])
        
        if my_sum_hp == 0.0:
            return float('-1')
        elif opp_sum_hp == 0.0:
            return float('1')
        
        # print(f"HP: {my_sum_hp}, {opp_sum_hp}")
        # print(f"Max HP: {my_sum_max_hp}, {opp_sum_max_hp}")
        return my_sum_hp / my_sum_max_hp * self.weights[0] - opp_sum_hp / opp_sum_max_hp * self.weights[1]

    def get_action(self, env: MyPkmEnv) -> int:
        """
        Determines the best action to take in the given environment using a pruned tree search.
        Args:
            env (PkmBattleEnv): The Pokémon battle environment.
            depth (int, optional): The depth of the search tree. Defaults to 2.
            instances (int, optional): The number of instances to run in parallel. Defaults to 8.
        Returns:
            int: The index of the best action to take.
        """
        depth = self.max_depth
        instances = self.instances
    
        moves = [0 for _ in range(DEFAULT_N_ACTIONS)]
        
        if self.parallel:
            with Pool() as pool:
                results = pool.map(self.estimate_move, [(env, depth) for i in range(instances)], chunksize=1)
        else:
            results = map(self.estimate_move, [(env, depth) for i in range(instances)])

        for result in results:
                moves[result] += 1     

        if self.debug:
            print(f"confidence: {max(moves) / instances}")
            print(f"moves: {moves}")
        return moves.index(max(moves))

    def estimate_move(self, params) -> int: 
        """
        Determines the best action to take based on the given parameters using a pruned tree search algorithm.
        Args:
            params (list): A list containing the environment and depth multiplier. 
                           params[0] is the environment object which will be deep-copied.
                           params[1] is an integer representing the depth multiplier for the search tree.
        Returns:
            int: The move determined to be the best action after performing the alpha-beta pruning search.
        """
        env = deepcopy(params[0])
        root = Node()
        root.env = env
        root.depth = params[1] * 2

        for team in root.env.teams:
            for move in team.active.moves:
                move.real_acc = move.acc
                move.acc = 1.0
    
            for party in team.party:
                for move in party.moves:
                    move.real_acc = move.acc
                    move.acc = 1.0

        outcome = self.alpha_beta(root, float('-inf'), float('inf'), True)
        return outcome.move

    def alpha_beta(self, node: Node, alpha: float, beta: float, maximizing_player: bool) -> Node:
        """
        Perform the alpha-beta pruning algorithm to find the best move for the current player.
        Args:
            node (Node): The current node in the game tree.
            alpha (float): The best value that the maximizer currently can guarantee at that level or above.
            beta (float): The best value that the minimizer currently can guarantee at that level or above.
            maximizing_player (bool): True if the current player is the maximizer, False if the current player is the minimizer.
        Returns:
            Node: The node with the best evaluated move for the current player.
        """

        # print(node.env.teams[0].active.moves[0].acc) 
        if node.depth == 0:
            eval_allhit = self.game_state_eval(node.env)
            node.eval = eval_allhit
            return node
        
        if maximizing_player:
            best_node = Node()
            best_node.eval = float('-inf')
            for i in range(DEFAULT_N_ACTIONS):
                max_player_node = Node()
                max_player_node.move = i
                max_player_node.env = deepcopy(node.env)
                max_player_node.parent = node
                max_player_node.depth = node.depth - 1
                min_player_node1 = self.alpha_beta(max_player_node, alpha, beta, False)

                if i < 4:
                    max_player_node = Node()
                    max_player_node.move = 99
                    max_player_node.env = deepcopy(node.env)
                    max_player_node.parent = node
                    max_player_node.depth = node.depth - 1
                    min_player_node2 = self.alpha_beta(max_player_node, alpha, beta, False)

                    eval1 = min_player_node1.eval * node.env.teams[0].active.moves[i].real_acc
                    eval2 = min_player_node2.eval * (1 - node.env.teams[0].active.moves[i].real_acc)
                    avg = eval1 + eval2
                else:
                    avg = min_player_node1.eval

                if avg > best_node.eval:
                    best_node = min_player_node1
                    best_node.eval = avg

                if beta <= best_node.eval:
                    break
                alpha = max(alpha, best_node.eval)
            return best_node
        
        else:
            best_node = Node()
            best_node.eval = float('inf')
            for i in range(DEFAULT_N_ACTIONS):
                min_player_node = Node()
                min_player_node.move = i
                min_player_node.env = deepcopy(node.env)
                min_player_node.parent = node
                min_player_node.depth = node.depth - 1
                min_player_node.env.step([node.move, i])
                max_player_node1 = self.alpha_beta(min_player_node, alpha, beta, True)

                if i < 4:
                    min_player_node = Node()
                    min_player_node.move = 99
                    min_player_node.env = deepcopy(node.env)
                    min_player_node.parent = node
                    min_player_node.depth = node.depth - 1
                    min_player_node.env.step([node.move, 99])
                    max_player_node2 = self.alpha_beta(min_player_node, alpha, beta, True)

                    eval1 = max_player_node1.eval * node.env.teams[1].active.moves[i].real_acc
                    eval2 = max_player_node2.eval * (1 - node.env.teams[1].active.moves[i].real_acc)
                    avg = eval1 + eval2
                else:
                    avg = max_player_node1.eval
                
                if avg < best_node.eval:
                    best_node = min_player_node
                    best_node.eval = avg
                    best_node.move = i

                if beta <= alpha:
                    break
                beta = min(beta, best_node.eval)
            return best_node
        
def n_fainted(t: PkmTeam):
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted

class PrunedTreeSearch2(PrunedTreeSearch):
    """
    Agent that uses Pruned Tree Search to select actions.
    """

    def __init__(self, weights: list[float] = [1.0, 1.0], max_depth: int = 2, instances: int = 50, parallel: bool = True, debug: bool = True):
        super().__init__(weights, max_depth, instances, parallel, debug)

    def alpha_beta(self, node: Node, alpha: float, beta: float, maximizing_player: bool) -> Node:
        """
        Perform the alpha-beta pruning algorithm to find the best move for the current player.
        Args:
            node (Node): The current node in the game tree.
            alpha (float): The best value that the maximizer currently can guarantee at that level or above.
            beta (float): The best value that the minimizer currently can guarantee at that level or above.
            maximizing_player (bool): True if the current player is the maximizer, False if the current player is the minimizer.
        Returns:
            Node: The node with the best evaluated move for the current player.
        """

        # print(node.env.teams[0].active.moves[0].acc) 
        if node.depth == 0:
            eval_allhit = self.game_state_eval(node.env)
            node.eval = eval_allhit
            return node
        
        if maximizing_player:
            best_node = Node()
            best_node.eval = float('-inf')
            for i in range(3, 6):
                if i < 4:
                    move = get_greedy(node.env)[1]
                else:
                    move = i
                max_player_node = Node()
                max_player_node.move = move
                max_player_node.env = deepcopy(node.env)
                max_player_node.parent = node
                max_player_node.depth = node.depth - 1
                min_player_node1 = self.alpha_beta(max_player_node, alpha, beta, False)
               
                if i < 4 and node.env.teams[0].active.moves[move].real_acc < 0.9:
                    max_player_node = Node()
                    max_player_node.move = 99
                    max_player_node.env = deepcopy(node.env)
                    max_player_node.parent = node
                    max_player_node.depth = node.depth - 1
                    min_player_node2 = self.alpha_beta(max_player_node, alpha, beta, False)

                    eval1 = min_player_node1.eval * node.env.teams[0].active.moves[i].real_acc
                    eval2 = min_player_node2.eval * (1 - node.env.teams[0].active.moves[i].real_acc)
                    avg = eval1 + eval2
                else:
                    avg = min_player_node1.eval

                if avg > best_node.eval:
                    best_node = max_player_node
                    best_node.eval = avg
                    best_node.move = move
                if beta <= best_node.eval:
                    break
                alpha = max(alpha, best_node.eval)

            return best_node
        
        else:
            best_node = Node()
            best_node.eval = float('inf')
            for i in range(3, 6):
                if i < 4:
                    move = get_greedy(node.env.get_states()[1])[1]
                else:
                    move = i
                
                min_player_node = Node()
                min_player_node.move = move
                min_player_node.env = deepcopy(node.env)
                min_player_node.parent = node
                min_player_node.depth = node.depth - 1
                min_player_node.env.step([node.move, i])
                max_player_node1 = self.alpha_beta(min_player_node, alpha, beta, True)
                
                if i < 4 and node.env.teams[1].active.moves[move].real_acc < 0.9:
                    min_player_node = Node()
                    min_player_node.move = 99
                    min_player_node.env = deepcopy(node.env)
                    min_player_node.parent = node
                    min_player_node.depth = node.depth - 1
                    min_player_node.env.step([node.move, 99])
                    max_player_node2 = self.alpha_beta(min_player_node, alpha, beta, True)

                    eval1 = max_player_node1.eval * node.env.teams[1].active.moves[i].real_acc
                    eval2 = max_player_node2.eval * (1 - node.env.teams[1].active.moves[i].real_acc)
                    avg = eval1 + eval2
                else:
                    avg = max_player_node1.eval

                if avg < best_node.eval:
                    best_node = min_player_node
                    best_node.eval = avg
                    best_node.move = move

                if beta <= alpha:
                    break

                beta = min(beta, best_node.eval)
            return best_node
        
