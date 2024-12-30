from vgc.behaviour import BattlePolicy, TeamBuildPolicy
from vgc.competition.Competitor import Competitor
from .PunisherTeamBuildPolicy import DamageTypeTeamBuilder
from vgc.datatypes.Types import PkmType, PkmStat
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
import numpy as np

# derived from damage_estimation but ignores weather
def simplified_damage_estimator(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, move_pp: int) -> float:
    stab = 1.5 if move_type == pkm_type else 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * stage * move_power
    if move_pp == 0:
        return 0.
    return damage

class Punisher(BattlePolicy):
    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    """
        Agent that always selects the move with the highest damage output based on
        the simplified damage estimator that looks for type advantages and stab.
    """
    def get_action(self, state) -> int:
        my_team = state.teams[0]
        enemy_team = state.teams[1]
        enemy_active = enemy_team.active
        my_pokemon = my_team.active

        max_damage = 0
        best_action = 0

        for j, move in enumerate(my_pokemon.moves):
            damage = simplified_damage_estimator(move.type, my_pokemon.type, move.power, enemy_active.type,
                                          my_team.stage[PkmStat.ATTACK], enemy_team.stage[PkmStat.DEFENSE], move.pp)
            # find the move with the highest damage
            if damage > max_damage:
                max_damage = damage
                best_action = j

        return best_action


class PunisherCompetitor(Competitor):

    def __init__(self, name: str = "Punisher"):
        self._name = name
        self._battle_policy = Punisher()
        self._team_build_policy = DamageTypeTeamBuilder()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
    
    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self._team_build_policy

