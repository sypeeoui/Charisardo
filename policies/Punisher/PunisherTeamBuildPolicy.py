from typing import List, Set
from vgc.behaviour import TeamBuildPolicy
from vgc.datatypes.Types import PkmType
from vgc.datatypes.Objects import PkmRoster, PkmFullTeam, Pkm
from vgc.balance.meta import MetaData

def calculate_damage_output(pkm: Pkm) -> float:
    return sum(move.power * move.acc for move in pkm.moves)

class DamageTypeTeamBuilder(TeamBuildPolicy):
    """
    Agent that selects teams by filtering Pokémon with HP greater than 150 and sorting by highest damage output.
    Only Unique Types are getting into the team.
    """

    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        filtered_roster = [pkm for pkm in self.roster if pkm.max_hp > 150]
        sorted_roster = sorted(filtered_roster, key=calculate_damage_output, reverse=True)
        team_finalized: List[Pkm] = []
        added_types: Set[PkmType] = set()
        for pt in sorted_roster:
            if len(team_finalized) < 3 and pt.type not in added_types:
                team_finalized.append(pt.gen_pkm([0, 1, 2, 3]))
                added_types.add(pt.type)
        return PkmFullTeam(team_finalized)
