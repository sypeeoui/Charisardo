import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from vgc.datatypes.Constants import MAX_HIT_POINTS, MOVE_POWER_MAX, MOVE_POWER_MIN, BASE_HIT_POINTS, \
    DEFAULT_PKM_N_MOVES, MAX_TEAM_SIZE, DEFAULT_TEAM_SIZE, DEFAULT_N_MOVES_PKM
from vgc.datatypes.Objects import Pkm, PkmMove, PkmFullTeam, PkmRoster, PkmTemplate, PkmTeam
from vgc.datatypes.Types import PkmType
from vgc.util import softmax
from vgc.util.generator.PkmTeamGenerators import *


class OwnRandomTeamGenerator(PkmTeamGenerator):

    def __init__(self, party_size: int = MAX_TEAM_SIZE - 1):
        self.party_size = party_size

    def get_team(self) -> PkmFullTeam:
        team: List[Pkm] = []
        for i in range(self.party_size + 1):
            p_type: PkmType = random.choice(LIST_OF_TYPES)
            # generate random hp with normal distribution but not less than 100
            max_hp: float = max(100, np.random.normal(300, 100))
            moves: List[PkmMove] = []
            for i in range(DEFAULT_PKM_N_MOVES):
                # generate random type for move
                m_type: PkmType = random.choice(LIST_OF_TYPES)

                # generate random power for move with normal distribution, and balance it using max_hp. Power must be in range [1, +inf]
                m_power: float = np.random.normal(80, 50)
                m_power = max(1, m_power - (max_hp-300)/5)

                # generate random accuracy for move and balance it using max_hp and m_power. Accuracy must be in range [0.1, 1]
                m_accuracy: float = np.random.normal(0.5, 0.2)
                m_accuracy += max_hp/m_power*0.1
                m_accuracy = max(0.1, min(1, m_accuracy))
                
                # generate PP for move based on power*accuracy
                # PP must be in the set {3, 5, 10, 15, 20, 25, 30}
                m_pp: int = 30
                if m_power*m_accuracy > 20:
                    m_pp = 25
                if m_power*m_accuracy > 30:
                    m_pp = 20
                if m_power*m_accuracy > 50:
                    m_pp = 15
                if m_power*m_accuracy > 70:
                    m_pp = 10
                if m_power*m_accuracy > 90:
                    m_pp = 5
                if m_power*m_accuracy > 130:
                    m_pp = 3
                

                moves.append(PkmMove(m_power, move_type=m_type, acc=m_accuracy, max_pp=m_pp))
            moves[0].type = p_type
            # random.shuffle(moves)
            team.append(Pkm(p_type, max_hp, move0=moves[0], move1=moves[1], move2=moves[2], move3=moves[3]))
        return PkmFullTeam(team)
