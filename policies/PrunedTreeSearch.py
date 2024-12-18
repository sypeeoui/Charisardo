from copy import deepcopy

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition

class Node:

    def __init__(self):
        self.a = None
        self.g = None
        self.parent = None
        self.depth = 0
        self.eval = 0.0

class PrunedTreeSearch(BattlePolicy):
    """
    Agent that uses Pruned Tree Search to select actions.
    """

    def __init__(self, weights: list[float] = [1.0, 1.0], max_depth: int = 3):
        self.weights = weights
        # self.min_depth = min_depth
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.max_depth = max_depth

    def game_state_eval(self, g: PkmBattleEnv):
        my_team = g.teams[0]
        opp_team = g.teams[1]
        my_sum_hp = my_team.active.hp + sum([p.hp for p in my_team.party])
        opp_sum_hp = opp_team.active.hp + sum([p.hp for p in opp_team.party])
        my_sum_max_hp = my_team.active.max_hp + sum([p.max_hp for p in my_team.party])
        opp_sum_max_hp = opp_team.active.max_hp + sum([p.max_hp for p in opp_team.party])
        
        if my_sum_hp == 0.0:
            return float('-inf')
        elif opp_sum_hp == 0.0:
            return float('inf')
        
        # print(f"HP: {my_sum_hp}, {opp_sum_hp}")
        # print(f"Max HP: {my_sum_max_hp}, {opp_sum_max_hp}")
        return my_sum_hp / my_sum_max_hp * self.weights[0] - opp_sum_hp / opp_sum_max_hp * self.weights[1]

    def get_action(self, g: PkmBattleEnv):
        g = deepcopy(g)
        curr_eval = self.game_state_eval(g)
        print(f"Current eval: {curr_eval}")

        # Set the accuracy of all moves to 1
        for team in g.teams:
            for move in team.active.moves:
                move.real_acc = move.acc
                move.acc = 1.0
            
            for party in g.teams[0].party:
                for move in party.moves:
                    move.real_acc = move.acc
                    move.acc = 1.0

        # Root initialization
        root = Node()
        root.g = g
        root.eval = curr_eval
        root.depth = 0
        root.alpha = self.alpha
        root.beta = self.beta

        stack = [root]
        best_node = None
        global_best_value = float('-inf')
        
        my_accs = [move.real_acc for move in g.teams[0].active.moves]
        opp_accs = [move.real_acc for move in g.teams[1].active.moves]

        while stack:
            curr_node = stack.pop()
            depth = curr_node.depth
            g = curr_node.g

            print(f"#####\nDepth: {depth}\n#####")
            if depth >= self.max_depth:
                continue

            alpha = curr_node.alpha
            beta = curr_node.beta
            print(f"Alpha: {alpha}, Beta: {beta}")
            
            my_best_value = float('-inf')

            for my_move in range(DEFAULT_N_ACTIONS):
                opp_best_value = float('inf')

                my_childs = []

                for opp_move in range(DEFAULT_N_ACTIONS):
                    if my_move >= 4:
                        my_acc = 1.0
                    else:
                        my_acc = my_accs[my_move]

                    if opp_move >= 4:
                        opp_acc = 1.0
                    else:
                        opp_acc = opp_accs[opp_move]


                    # Generate the next game state in all possible ways
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, opp_move]) # I skip
                    eval_iskip = self.game_state_eval(g_copy)

                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([my_move, 99]) # Opponent skips
                    eval_oppskip = self.game_state_eval(g_copy)
                    
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([my_move, opp_move])
                    eval_allhit = self.game_state_eval(g_copy)

                    new_eval = my_acc * opp_acc * eval_allhit + (1 - my_acc) * opp_acc * eval_iskip + \
                        my_acc * (1 - opp_acc) * eval_oppskip + (1 - my_acc) * (1 - opp_acc) * curr_node.eval
                    
                    print(f"######\nNew eval: {new_eval}, depth: {depth}, my_move: {my_move}, opp_move: {opp_move}")
                    print(f"Eval: {eval_iskip}, {eval_oppskip}, {eval_allhit}, {curr_node.eval}")
                    print(f"Acc: {my_acc}, {opp_acc}")

                    # minimizing player's turn
                    opp_best_value = min(opp_best_value, new_eval)
                    beta = min(beta, opp_best_value)

                    if beta < alpha:
                        print(f"Pruning alpha, {opp_best_value} < {alpha}")
                        continue
                    
                    
                    print(f"New Beta: {beta}, New Opp Best Value: {opp_best_value}")


                    # Create and add the new node to the stack
                    child_node = Node()
                    child_node.g = g_copy
                    child_node.eval = new_eval
                    child_node.depth = depth + 1
                    child_node.alpha = alpha
                    child_node.beta = beta
                    child_node.parent = curr_node
                    child_node.a = my_move

                    my_childs.append(child_node)

                # Maximizing player's turn (my_move updates alpha)
                my_best_value = max(my_best_value, opp_best_value)
                alpha = max(alpha, my_best_value)

                if my_best_value > global_best_value:
                    global_best_value = my_best_value
                    best_node = curr_node

                if alpha > beta:
                    print(f"Pruning alpha, {my_best_value} > {beta}")
                    continue
                    

                stack.extend(my_childs)

        # get the best action
        while best_node.depth > 1:
            print(best_node.depth)
            best_node = best_node.parent
        
        return best_node.a
