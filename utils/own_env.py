import random
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from typing import Tuple
from multiprocessing.connection import Client
from copy import deepcopy

from vgc.competition.StandardPkmMoves import Struggle
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS, STATE_DAMAGE, SPIKES_2, SPIKES_3, \
    TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import PkmTeam, Pkm, GameState, Weather
from vgc.datatypes.Types import WeatherCondition, PkmEntryHazard, PkmType, PkmStatus, PkmStat, N_HAZARD_STAGES, \
    MIN_STAGE, MAX_STAGE
from vgc.engine.HiddenInformation import set_pkm
from vgc.util.Encoding import GAME_STATE_ENCODE_LEN, partial_encode_game_state, encode_game_state

class MyPkmEnv(PkmBattleEnv):
    def __init__(self, teams: Tuple[PkmTeam, PkmTeam], weather=Weather(), debug: bool = False, conn: Client = None,
                 encode: Tuple[bool, bool] = (True, True)):
        super().__init__(teams, weather, debug, conn, encode)
        
    def _PkmBattleEnv__get_attack_order(self, actions) -> Tuple[int, int]:
        """
        Get attack order for this turn. 
        Priority is given to the pkm with highest speed_stage. Otherwise random.
        Fixed with random.
        :return: tuple with first and second trainer to perform attack
        """
        # print("ordering with my function")
        # return 0
        action0 = actions[0]
        action1 = actions[1]
        speed0 = self.teams[0].stage[PkmStat.SPEED] + (
            self.teams[0].active.moves[action0].priority if action0 < DEFAULT_PKM_N_MOVES else 0)
        speed1 = self.teams[1].stage[PkmStat.SPEED] + (
            self.teams[1].active.moves[action1].priority if action1 < DEFAULT_PKM_N_MOVES else 0)
        if speed0 > speed1:
            order = [0, 1]
        elif speed1 > speed0:
            order = [1, 0]
        else:
            # random attack order
            order = [0, 1]
            random.shuffle(order)
        return order[0], order[1]
    
    def _PkmBattleEnv__get_forward_env(self, player: int):
        """
        Get the next environment state after the current action.
        :return: next environment state
        """
        # print("getting forward env with my function")
        # return self
        # print("sono dentro")
        
        env = MyPkmEnv((deepcopy(self.teams[player]), deepcopy(self.teams[not player])), deepcopy(self.weather),
                           encode=self.requires_encode)
        env.n_turns_no_clear = self.n_turns_no_clear
        env.turn = self.turn
        env.winner = self.winner
        # print(f"BBBBBB\n{env.teams[0].__str__()}\nBBBBBB")
        # print(f"BBBBBB\n{env.teams[1].__str__()}\nBBBBBB")
        # print(env.teams[1].__str__())
        # print("fuori")
        return env

        
        

