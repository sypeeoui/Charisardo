from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition

class PrunedTreeSearch(BattlePolicy):
    # TODO: Implement this policy
    def get_action(self, g: GameState):
        return 0 