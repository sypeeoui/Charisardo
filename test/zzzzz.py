# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from vgc.behaviour.BattlePolicies import *
from vgc.datatypes.Objects import PkmTeam
# from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.balance.meta import StandardMetaData
from vgc.behaviour.TeamBuildPolicies import RandomTeamBuilder
from vgc.behaviour.TeamSelectionPolicies import RandomTeamSelectionPolicy
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
from multiprocessing.connection import Client

import sys, os
# add relative parent directory to path
sys.path.append(os.path.dirname(os.getcwd()))

from utils import OwnRandomTeamGenerator, MyPkmEnv
from policies.PrunedTreeSearch import PrunedTreeSearch


# %%
gen = OwnRandomTeamGenerator()
full_team0 = gen.get_team()
full_team1 = gen.get_team()
team0 = full_team0.get_battle_team([0, 1, 2])
team1 = full_team1.get_battle_team([0, 1, 2])
agent0, agent1 = PrunedTreeSearch(parallel = False, instances = 3), TypeSelector()

env = MyPkmEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True)  # set new environment with teams


# %%
# team0.active.moves[0].acc = 0
# team0.active.moves[1].acc = 0
# team0.active.moves[2].acc = 0
# team0.active.moves[3].acc = 0

for move in team0.active.moves:
    print(move.__str__())

print()
for move in team1.active.moves:
    print(move.__str__())
print()

print(team0.__str__())
print(team1.__str__())


# team1.active.moves[0].pp = 1

# %%
print(team0.active)

# %%
# # print(agent0.game_state_eval(env))
s, a, b, c, d = env.step([5, 66])
print(env.log)
# print(agent0.get_action(env))

# %%
print(s[0].teams[0].__str__())
print(s[0].teams[1].__str__())
print(s[1].teams[0].__str__())
print(s[1].teams[1].__str__())

# %%
print(a)


# %%
print(env.log)

# %%
env.reset()

# %%
# from multiprocessing import Pool

# def get_action_wrapper(_):
#     return agent0.get_action(env, depth=4)

# moves = [0, 0, 0, 0, 0, 0]
# with Pool() as pool:
#     results = pool.map(get_action_wrapper, range(15), chunksize=1)

# # print(agent0.get_action(env, depth=4))

# for result in results:
#     moves[result] += 1

# print(moves)

# %%
get_action_wrapper(0)

# %%
class Padre:
    def _a(self):
        return "Metodo del padre"

    def usa_a(self):
        return self._a()

class Figlio(Padre):
    def _a(self):
        return "Metodo del figlio"

# Test
figlio = Figlio()
print(figlio.usa_a())  # Output: "Metodo del figlio"


# %%
env.reset()

# %%
g = env
for m in g.teams[0].active.moves:
    print(m.acc)
    m.real_acc = float(m.acc)
    m.acc = 1.0
for m in g.teams[0].active.moves:
    print(m.acc)
    print(m.real_acc)

# %%



