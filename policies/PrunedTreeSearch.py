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
        

    # def get_action(self, g: PkmBattleEnv):
    #     g = deepcopy(g)
    #     curr_eval = self.game_state_eval(g)
    #     print(f"Current eval: {curr_eval}")

    #     # Set the accuracy of all moves to 1
    #     for team in g.teams:
    #         for move in team.active.moves:
    #             move.real_acc = move.acc
    #             move.acc = 1.0
            
    #         for party in g.teams[0].party:
    #             for move in party.moves:
    #                 move.real_acc = move.acc
    #                 move.acc = 1.0

    #     # Root initialization
    #     root = Node()
    #     root.g = g
    #     root.eval = curr_eval
    #     root.depth = 0
    #     root.alpha = self.alpha
    #     root.beta = self.beta

    #     stack = [root]
    #     best_node = None
    #     global_best_value = float('-inf')
        
    #     my_accs = [move.real_acc for move in g.teams[0].active.moves]
    #     opp_accs = [move.real_acc for move in g.teams[1].active.moves]

    #     while stack:
    #         curr_node = stack.pop()
    #         depth = curr_node.depth
    #         g = curr_node.g

    #         print(f"#####\nDepth: {depth}\n#####")
    #         if depth >= self.max_depth:
    #             continue

    #         alpha = curr_node.alpha
    #         beta = curr_node.beta
    #         print(f"Alpha: {alpha}, Beta: {beta}")
            
    #         my_best_value = float('-inf')

    #         for my_move in range(DEFAULT_N_ACTIONS):
    #             opp_best_value = float('inf')

    #             my_childs = []

    #             for opp_move in range(DEFAULT_N_ACTIONS):
    #                 if my_move >= 4:
    #                     my_acc = 1.0
    #                 else:
    #                     my_acc = my_accs[my_move]

    #                 if opp_move >= 4:
    #                     opp_acc = 1.0
    #                 else:
    #                     opp_acc = opp_accs[opp_move]


    #                 # Generate the next game state in all possible ways
    #                 g_copy = deepcopy(g)
    #                 s, _, _, _, _ = g_copy.step([99, opp_move]) # I skip
    #                 eval_iskip = self.game_state_eval(g_copy)

    #                 g_copy = deepcopy(g)
    #                 s, _, _, _, _ = g_copy.step([my_move, 99]) # Opponent skips
    #                 eval_oppskip = self.game_state_eval(g_copy)
                    
    #                 g_copy = deepcopy(g)
    #                 s, _, _, _, _ = g_copy.step([my_move, opp_move])
    #                 eval_allhit = self.game_state_eval(g_copy)

    #                 new_eval = my_acc * opp_acc * eval_allhit + (1 - my_acc) * opp_acc * eval_iskip + \
    #                     my_acc * (1 - opp_acc) * eval_oppskip + (1 - my_acc) * (1 - opp_acc) * curr_node.eval
                    
    #                 print(f"######\nNew eval: {new_eval}, depth: {depth}, my_move: {my_move}, opp_move: {opp_move}")
    #                 print(f"Eval: {eval_iskip}, {eval_oppskip}, {eval_allhit}, {curr_node.eval}")
    #                 print(f"Acc: {my_acc}, {opp_acc}")

    #                 # minimizing player's turn
    #                 opp_best_value = min(opp_best_value, new_eval)
    #                 beta = min(beta, opp_best_value)

    #                 if beta < alpha:
    #                     print(f"Pruning alpha, {opp_best_value} < {alpha}")
    #                     continue
                    
                    
    #                 print(f"New Beta: {beta}, New Opp Best Value: {opp_best_value}")


    #                 # Create and add the new node to the stack
    #                 child_node = Node()
    #                 child_node.g = g_copy
    #                 child_node.eval = new_eval
    #                 child_node.depth = depth + 1
    #                 child_node.alpha = alpha
    #                 child_node.beta = beta
    #                 child_node.parent = curr_node
    #                 child_node.a = my_move

    #                 my_childs.append(child_node)

    #             # Maximizing player's turn (my_move updates alpha)
    #             my_best_value = max(my_best_value, opp_best_value)
    #             alpha = max(alpha, my_best_value)

    #             if my_best_value > global_best_value:
    #                 global_best_value = my_best_value
    #                 best_node = curr_node

    #             if alpha > beta:
    #                 print(f"Pruning alpha, {my_best_value} > {beta}")
    #                 continue
                    

    #             stack.extend(my_childs)

    #     # get the best action
    #     while best_node.depth > 1:
    #         print(best_node.depth)
    #         best_node = best_node.parent
        
    #     return best_node.a

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
        
