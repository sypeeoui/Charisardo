
from copy import deepcopy

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, Weather, PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
import numpy as np

SKIP = 99

def get_greedy(env, opp_move=SKIP) -> tuple[int, float]:
    opp_hp_initial = env.teams[1].active.hp
    fainted = n_fainted(env.teams[1])
    damages = []
    for move in range(4):
        g = deepcopy(env)
        g.step((move, opp_move))
        if n_fainted(g.teams[1]) > fainted:
            damages.append(opp_hp_initial * env.teams[0].active.moves[move].real_acc)
        else:
            opp_hp_final = g.teams[1].active.hp
            damages.append((opp_hp_initial - opp_hp_final) * env.teams[0].active.moves[move].real_acc)
    best = np.argmax(damages)
    # print(damages)
    return damages[best], best

def n_fainted(t: PkmTeam):
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted

class WeightedGreedy(BattlePolicy):
    def __init__(self, weight: list[float] = [1.85]):
        self.weight = weight
        pass
                
    def get_action(self, env: PkmBattleEnv) -> int:
        for team in env.teams:
            for move in team.active.moves:
                move.real_acc = move.acc
                move.acc = 1.0
    
            for party in team.party:
                for move in party.moves:
                    move.real_acc = move.acc
                    move.acc = 1.0
        best_choices = []
        best_choices.append(get_greedy(env))
        for move in range(4, 6):
            g = deepcopy(env)
            g.step((move, SKIP))
            damage, _ = get_greedy(g)
            best_choices.append((damage / self.weight[0], move))
        best = max(best_choices)
        # print(best)
        return best[1]
    
class WeightedGreedy2(BattlePolicy):
    def __init__(self, weight: list[float] = [1.1, 1.3, 1.7]):
        self.weight = weight
        pass
      
    def get_action(self, env: PkmBattleEnv) -> int:
        # print(env)
        for team in env.teams:
            for move in team.active.moves:
                move.real_acc = move.acc
                move.acc = 1.0
    
            for party in team.party:
                for move in party.moves:
                    move.real_acc = move.acc
                    move.acc = 1.0
        best_choices = []
        s, _, _, _, _ = env.step((SKIP, SKIP))
        my_best = get_greedy(env)
        his_best_nochange = get_greedy(s[1])
        best_choices.append((my_best[0] - his_best_nochange[0], my_best[1]))
        # print("###### \n\n\n\n\nI'm here \n\n\n\n##########")

        for move in range(4, 6):
            g = deepcopy(env)
            g.debug = True
            # print(g.teams[0].__str__())
            # print(g.teams[1].__str__())
            s, _, _, _, _ = g.step((move, SKIP))
            # a = deepcopy(g.teams[0])
            # print(a.__str__())
            # print("#######COMPACT#####")
            # print(g.teams[0].__str__())
            # print(g.teams[1].__str__())
            # print(s[0].teams[0].__str__())
            # print(s[0].teams[1].__str__())
            # print(s[1].teams[0].__str__())
            # print(s[1].teams[1].__str__())
            damage, _ = get_greedy(g)

            # print(new_env.teams[0].__str__())
            # print(new_env.teams[1].__str__())
            his_best_withchange = get_greedy(s[1])
            # print(f"#####\n{g.log}\n####")
            val = (damage - his_best_nochange[0] / self.weight[0] - his_best_withchange[0] / self.weight[1]) / self.weight[2]
            if val < 0:
                val *= 2
            best_choices.append((val, move))
        
        best = max(best_choices)
        # print(best)
        return best[1]

