'''
hayo4.py에서 should_switch_based_on_opponent 및 find_best_switch 메서드를 확장
'''

from typing import List
import numpy as np

from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmEntryHazard

class hayo5_BattlePolicy(BattlePolicy):
    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES, n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        # 날씨 효과가 게임 중 사용되었는지 
        self.hail_used = False 
        self.sandstorm_used = False
        # 상대방이 사용한 마지막 몇 개의 움직임을 기록 
        self.opp_move_history = []

    @staticmethod # 공격의 예상 데미지 계산 
    def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType, attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
        stab = 1.5 if move_type == pkm_type else 1.
        weather_bonus = 1.5 if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY) else 0.5 if (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (move_type == PkmType.FIRE and weather == WeatherCondition.RAIN) else 1
        stage = (attack_stage - defense_stage + 2) / 2 if attack_stage - defense_stage >= 0 else 2 / (-attack_stage + defense_stage + 2)
        return TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather_bonus * stage * move_power

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    # GameState를 기반으로 가장 적합한 행동을 결정한다. 
    def get_action(self, g: GameState) -> int:
        weather = g.weather.condition
        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        
        opp_team = g.teams[1]
        opp_active = opp_team.active
        
        last_move=getattr(opp_active, 'last_move', None)
        if last_move:
            self.update_opponent_move_history(last_move)

        # Analyze patterns and decide on strategy
        # 상대방 최근 공격 이력을 기반으로 포켓몬을 교체할지 결정한다. 
        if self.should_switch_based_on_opponent(opp_active):
            switch_to = self.find_best_switch(my_party, opp_active)
            if switch_to is not None:
                return switch_to + 4  # Adjust for switching command index

        # Continue with other decision-making
        # 상성에 기반한 포켓몬 교체 
        move_id = self.find_best_move(my_active, opp_active, weather)
        if move_id is not None:
            return move_id
        
        return 0  # Default to first move if no better option is found

    # 상대방의 이동 기록을 업데이트한다. 
    def update_opponent_move_history(self, move):
        if move:
            self.opp_move_history.append(move)
            if len(self.opp_move_history) > 10:
                self.opp_move_history.pop(0)

    # 상대방의 최근 행동을 바탕으로 포켓몬을 교체할지 결정한다.
    def should_switch_based_on_opponent(self, opp_active):
        if not self.opp_move_history:
            return False
        
        last_move_type = self.opp_move_history[-1].type

        # 공격 타입에 따른 교체 필요 여부 결정
        if last_move_type in [PkmType.FIRE, PkmType.ELECTRIC, PkmType.GRASS, PkmType.ICE]:
            return True
        return False

    # 가장 유리한 교체 대상을 찾는다.
    def find_best_switch(self, party, opp_active):
        last_move_type = self.opp_move_history[-1].type if self.opp_move_history else None

        type_advantage = {
            PkmType.FIRE: PkmType.WATER,
            PkmType.ELECTRIC: PkmType.GROUND,
            PkmType.GRASS: PkmType.FIRE,
            PkmType.ICE: PkmType.FIRE,
            # 추가적인 타입 상성 정보를 여기에 추가
        }

        best_switch_index = None
        for i, pkm in enumerate(party):
            if last_move_type and pkm.type == type_advantage.get(last_move_type):
                best_switch_index = i
                break

        return best_switch_index
    
    # 사용할 최적의 공격을 결정한다. 
    def find_best_move(self, my_active, opp_active, weather):
        default_stage = {PkmStat.ATTACK: 0, PkmStat.DEFENSE: 0}
        
        # my_active와 opp_active의 stage 정보 접근 시 기본값 사용
        my_attack_stage = getattr(my_active, 'stage', default_stage).get(PkmStat.ATTACK, 0)
        my_defense_stage = getattr(my_active, 'stage', default_stage).get(PkmStat.DEFENSE, 0)
        opp_attack_stage = getattr(opp_active, 'stage', default_stage).get(PkmStat.ATTACK, 0)
        opp_defense_stage = getattr(opp_active, 'stage', default_stage).get(PkmStat.DEFENSE, 0)

        # 데미지 계산
        return np.argmax([self.estimate_damage(move.type, my_active.type, move.power, opp_active.type, 
                                               my_attack_stage, opp_defense_stage, weather) 
                          for move in my_active.moves])
    
    # 현재 날씨 상태를 체크하고 전략적으로 날씨를 변경할 기회를 판단한다. 
    def check_weather_condition(self, active_pkm, party):
        for i, move in enumerate(active_pkm.moves):
            if move.weather in [WeatherCondition.SANDSTORM, WeatherCondition.HAIL]:
                for pkm in party:
                    if pkm.type in [PkmType.ICE] and move.weather == WeatherCondition.HAIL:
                        self.hail_used = True
                        return i
                    elif pkm.type in [PkmType.ROCK, PkmType.STEEL, PkmType.GROUND] and move.weather == WeatherCondition.SANDSTORM:
                        self.sandstorm_used = True
                        return i
        return None

    # 방어적 위치를 평가하여 교체가 필요한지 결정한다. 
    def evaluate_defensive_position(self, party, opp_active, current_active):
        min_damage = float('inf')
        switch_index = None
        for i, pkm in enumerate(party):
            if not pkm.fainted() and pkm is not current_active:  # 기절하지 않았고 현재 전투 중인 포켓몬이 아닐 때
                pkm_damage = evaluate_matchup(pkm.type, opp_active.type, [m.type for m in opp_active.moves])
                if pkm_damage < min_damage:
                    min_damage = pkm_damage
                    switch_index = i
        return switch_index
    
# 상태 이상을 유발하는 공격이 가능한지 확인하고 적용한다. 
def apply_status_effects(self, my_active, opp_active):
    for move in my_active.moves:
        if move.effect == 'poison' and not opp_active.status['poisoned']: 
            return my_active.moves.index(move)
    return None

def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: List[PkmType]) -> float:
    damage = [TYPE_CHART_MULTIPLIER[mtype][pkm_type] for mtype in moves_type]
    return min(damage) if damage else 1.0


class hayo_competitor5(Competitor):
    def __init__(self, name: str = "hayo5"):
        self._name = name
        self._battle_policy = hayo5_BattlePolicy()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
