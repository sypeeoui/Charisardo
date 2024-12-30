from typing import List
from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.behaviour.BattlePolicies import RandomPlayer
from vgc.behaviour.TeamSelectionPolicies import FirstEditionTeamSelectionPolicy
from vgc.competition.Competitor import Competitor
from vgc.datatypes.Objects import PkmFullTeam, PkmRoster, Pkm
from vgc.datatypes.Types import PkmType
import random
from vgc.balance.meta import MetaData, StandardMetaData


# 포켓몬 타입 간의 상성을 정의하는 딕셔너리
type_advantage = {
    PkmType.FIRE: [PkmType.GRASS, PkmType.ICE, PkmType.BUG, PkmType.STEEL],
    PkmType.WATER: [PkmType.FIRE, PkmType.GROUND, PkmType.ROCK],
    PkmType.GRASS: [PkmType.WATER, PkmType.GROUND, PkmType.ROCK],
    PkmType.ELECTRIC: [PkmType.WATER, PkmType.FLYING],
    PkmType.ROCK: [PkmType.FIRE, PkmType.ICE, PkmType.FLYING, PkmType.BUG],
    PkmType.ICE: [PkmType.GRASS, PkmType.GROUND, PkmType.FLYING, PkmType.DRAGON],
    PkmType.FIGHT: [PkmType.NORMAL, PkmType.ICE, PkmType.ROCK, PkmType.DARK, PkmType.STEEL],
    PkmType.POISON: [PkmType.GRASS, PkmType.FAIRY],
    PkmType.GROUND: [PkmType.FIRE, PkmType.ELECTRIC, PkmType.POISON, PkmType.ROCK, PkmType.STEEL],
    PkmType.FLYING: [PkmType.GRASS, PkmType.FIGHT, PkmType.BUG],
    PkmType.PSYCHIC: [PkmType.FIGHT, PkmType.POISON],
    PkmType.BUG: [PkmType.GRASS, PkmType.PSYCHIC, PkmType.DARK],
    PkmType.GHOST: [PkmType.PSYCHIC, PkmType.GHOST],
    PkmType.DRAGON: [PkmType.DRAGON],
    PkmType.DARK: [PkmType.PSYCHIC, PkmType.GHOST],
    PkmType.STEEL: [PkmType.ICE, PkmType.ROCK, PkmType.FAIRY],
    PkmType.FAIRY: [PkmType.FIGHT, PkmType.DRAGON, PkmType.DARK]
}

# 예측된 상대방 팀을 기반으로 팀을 구성하는 클래스
class StrategicTeamBuildPolicy(TeamBuildPolicy):
    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    # 메타 데이터를 기반으로 상대방의 가장 일반적인 포켓몬 타입을 예측한다. 
    # 상대방 팀의 포켓몬 타입을 기반으로 최적의 상성을 가진 포켓몬을 선택하여 팀을 구성한다. 
    def get_action(self, meta: MetaData) -> PkmFullTeam:
        team: List[Pkm] = []
        predicted_opponent_types = self.predict_opponent_types(meta)
        selected_types = []

        for target_type in predicted_opponent_types:
            best_counter_types = [type for type, counters in type_advantage.items() if target_type in counters]
            for pkm in self.roster:
                if pkm.type in best_counter_types and pkm.type not in selected_types:
                    moves = [0, 1, 2, 3]
                    team.append(pkm.gen_pkm(moves))
                    selected_types.append(pkm.type)
                    break

        # 팀이 충분히 채워지지 않았다면 나머지는 랜덤 선택
        while len(team) < 3:
            pkm = random.choice(self.roster)
            if pkm.type not in selected_types:
                moves = [0, 1, 2, 3]
                team.append(pkm.gen_pkm(moves))
                selected_types.append(pkm.type)

        return PkmFullTeam(team)
    
    def predict_opponent_types(self, meta: StandardMetaData) -> List[PkmType]:
        type_count = {ptype: 0 for ptype in PkmType}
        n_teams = meta.get_n_teams()

        for i in range(n_teams):
            team = meta.get_team(i)
            for pkm in team.pkm_list:
                type_count[pkm.type] += 1

        predicted_types = sorted(type_count, key=type_count.get, reverse=True)[:2]
        return predicted_types
 
    def predict_opponent_strategies(self, meta: StandardMetaData) -> List[PkmType]:
        strategy_weight = {}
        n_teams = meta.get_n_teams()

        for i in range(n_teams):
            team = meta.get_team(i)
            team_strategy = self.analyze_team_strategy(team)
            for strategy, weight in team_strategy.items():
                if strategy in strategy_weight:
                    strategy_weight[strategy] += weight
                else:
                    strategy_weight[strategy] = weight

        selected_strategies = sorted(strategy_weight, key=strategy_weight.get, reverse=True)[:2]
        predicted_types = [strategy for strategy in selected_strategies]
        return predicted_types

    def analyze_team_strategy(self, team) -> dict:
        strategy_count = {ptype: 0 for ptype in PkmType}
        for pkm in team.pkm_list:
            strategy_count[pkm.type] += 1
        return strategy_count

class hayoteam5_Competitor(Competitor):
    def __init__(self, name: str = "hayo5"):
        self._name = name
        self._battle_policy = RandomPlayer()
        self._team_selection_policy = FirstEditionTeamSelectionPolicy()
        self._team_build_policy = StrategicTeamBuildPolicy()

    @property
    def name(self):
        return self._name

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self._team_build_policy

    @property
    def team_selection_policy(self) -> TeamSelectionPolicy:
        return self._team_selection_policy

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
