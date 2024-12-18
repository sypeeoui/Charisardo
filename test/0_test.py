import numpy as np
import matplotlib.pyplot as plt
import time
from vgc.behaviour.BattlePolicies import *
from vgc.datatypes.Objects import PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.balance.meta import StandardMetaData
from vgc.behaviour.TeamBuildPolicies import RandomTeamBuilder
from vgc.behaviour.TeamSelectionPolicies import RandomTeamSelectionPolicy
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
from multiprocessing.connection import Client

import sys, os
# add relative parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import OwnRandomTeamGenerator


gen = OwnRandomTeamGenerator()
full_team0 = gen.get_team()
full_team1 = gen.get_team()
team0 = full_team0.get_battle_team([0, 1, 2])
team1 = full_team1.get_battle_team([0, 1, 2])
agent0, agent1 = TypeSelector(), RandomPlayer()

address = ('localhost', 6000)
conn = Client(address, authkey='VGC AI'.encode('utf-8'))

env = PkmBattleEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True, conn=conn)  # set new environment with teams
n_battles = 3  # total number of battles
env.reset()
t = False
battle = 0
while battle < n_battles:
    s, _ = env.reset()
    while not t:  # True when all pkms of one of the two PkmTeam faint
        a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
        s, _, t, _, _ = env.step(a)  # for inference, we don't need reward
        env.render(mode='ux')
        print(env.log)
    t = False
    battle += 1
print(env.winner)  # winner id number

