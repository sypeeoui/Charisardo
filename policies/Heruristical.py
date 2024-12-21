from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam, Pkm
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
from vgc.engine.PkmBattleEnv import PkmBattleEnv
import numpy as np
from copy import deepcopy

class Heuristical(BattlePolicy):
    """
    Heuristical policy implementing a one-turn lookahead strategy based on the described heuristic algorithm.

    The core idea:
    - For each possible action (either using a move or switching to another Pokemon), 
      we estimate a heuristic "probability of eventually winning."
    - We do this by comparing how many moves each side needs to KO the other, 
      considering move accuracies and speed (from team stages).
    - We repeat the calculation not only for the currently active Pokemon but also 
      for the other party members. For non-active Pokemon, we consider the penalty of switching in 
      (they effectively lose a turn due to a dummy, zero-damage move).
    - We then combine these probabilities from all non-fainted allies using the geometric mean 
      (nth-root of the product).
    - We pick the action that maximizes this final heuristic value.

    Details:
    - The speed comparison is based on team stages, not on individual Pokemon attributes. The Pkm object 
      does not have a `stats` attribute for speed. Instead, the effective speed factor is represented by 
      the team's speed stage `team.stage[PkmStat.SPEED]`.
    - The accuracy field of a move is called `acc`, not `accuracy`.
    - We treat future hypothetical scenarios consistently by applying switch penalties (extra dummy move + 
      hazard damage) to non-active Pokemon.

    This heuristic is not a full tree search; it's a one-turn lookahead approach.
    """

    def estimate_damage(self, move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                        attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
        """Estimate the damage a single move would do, considering STAB, weather, and stages."""
        stab = 1.5 if move_type == pkm_type else 1.0
        if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or \
           (move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
            w = 1.5
        elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or \
             (move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
            w = 0.5
        else:
            w = 1.0

        stage_level = attack_stage - defense_stage
        if stage_level >= 0:
            stage = (stage_level + 2.) / 2
        else:
            stage = 2. / (abs(stage_level) + 2.)

        damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * w * stage * move_power
        return damage

    def best_move_info(self, attacker: Pkm, defender: Pkm, atk_stage: int, def_stage: int, weather: WeatherCondition):
        """
        Find the best damaging move (max damage) and return its damage and accuracy.
        If no damaging move is found, returns (0.0, 1.0).
        """
        best_dmg = 0.0
        best_acc = 1.0
        for m in attacker.moves:
            dmg = self.estimate_damage(m.type, attacker.type, m.power, defender.type, atk_stage, def_stage, weather)
            if dmg > best_dmg:
                best_dmg = dmg
                best_acc = m.acc
        if best_dmg == 0.0:
            # no damage means can't KO
            return 0.0, 1.0
        return best_dmg, best_acc

    def moves_to_ko(self, hp: float, dmg: float) -> int:
        """Compute how many hits are needed to KO a Pokemon with given hp using moves of given damage."""
        if dmg <= 0:
            return 9999  # effectively cannot KO
        return int(np.ceil(hp / dmg))

    def winning_probability(self, m_o: int, m_t: int, p_ours: float, p_theirs: float, we_are_faster: bool) -> float:
        """
        Compute the heuristic winning probability given:
        - m_o: moves we need to KO the opponent
        - m_t: moves they need to KO us
        - p_ours: probability that all our required hits land
        - p_theirs: probability that all their required hits land
        - we_are_faster: boolean indicating if we strike first
        """
        # Heuristic rules:
        if m_o < m_t:
            if we_are_faster:
                return p_ours
            else:
                return p_ours * (1 - (p_theirs ** m_o))
        elif m_o == m_t:
            if we_are_faster:
                return p_ours
            else:
                return p_ours * (1 - (p_theirs ** m_o))
        else:  # m_o > m_t
            return p_ours * (1 - (p_theirs ** m_t))

    def is_faster(self, my_speed_stage: float, opp_speed_stage: float):
        """Check if we are faster or equal in speed compared to the opponent based on team speed stages."""
        return my_speed_stage >= opp_speed_stage

    def evaluate_team_probability(self, chosen_pkm: Pkm, opponent_pkm: Pkm,
                                  switch_penalty: bool, 
                                  my_attack_stage: int, opp_defense_stage: int,
                                  opp_attack_stage: int, my_defense_stage: int,
                                  weather: WeatherCondition, hazard_damage: float,
                                  my_speed_stage: float, opp_speed_stage: float) -> float:
        """
        Evaluate the probability of eventually winning if chosen_pkm faces opponent_pkm.
        switch_penalty: True if we must consider losing a turn (dummy move) and hazard damage.
        We consider speed advantage using team speed stages.
        """
        # Work on copies to avoid mutating originals
        us = deepcopy(chosen_pkm)
        them = deepcopy(opponent_pkm)

        # Apply hazard damage if switching in
        if switch_penalty:
            us.hp = max(0.0, us.hp - hazard_damage)

        if us.hp <= 0:
            # We faint immediately due to hazards
            return 0.0

        # Get best moves info
        my_best_dmg, my_best_acc = self.best_move_info(us, them, my_attack_stage, opp_defense_stage, weather)
        opp_best_dmg, opp_best_acc = self.best_move_info(them, us, opp_attack_stage, my_defense_stage, weather)

        if my_best_dmg == 0.0:
            # We can't kill them
            return 0.0

        # If opponent can't kill us at all:
        if opp_best_dmg == 0.0:
            m_o = self.moves_to_ko(them.hp, my_best_dmg)
            if switch_penalty:
                m_o += 1  # One wasted turn
            p_ours = (my_best_acc ** m_o)
            return p_ours

        # Moves needed for each side
        m_o = self.moves_to_ko(them.hp, my_best_dmg)
        m_t = self.moves_to_ko(us.hp, opp_best_dmg)
        if switch_penalty:
            # One extra turn wasted
            m_o += 1

        p_ours = (my_best_acc ** m_o)
        p_theirs = (opp_best_acc ** m_t)

        speed_advantage = self.is_faster(my_speed_stage, opp_speed_stage)
        p_win = self.winning_probability(m_o, m_t, p_ours, p_theirs, speed_advantage)
        return p_win

    def get_action(self, g: PkmBattleEnv):
        # We consider all actions (4 moves + up to 2 switches)
        actions = list(range(DEFAULT_N_ACTIONS))

        my_team = g.teams[0]
        opp_team = g.teams[1]
        weather = g.weather.condition

        # Assume no hazard damage given by instructions (not specified).
        hazard_damage = 0.0

        my_active = my_team.active
        opponent_active = opp_team.active

        # Prepare stage variables
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]
        my_speed_stage = my_team.stage[PkmStat.SPEED]

        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]
        opp_speed_stage = opp_team.stage[PkmStat.SPEED]

        # Gather our pokemons (active + party)
        all_my_pokemons = [my_active] + list(my_team.party)

        best_action = 0
        best_value = -1.0

        # Iterate over possible actions
        for action in actions:
            if action < DEFAULT_PKM_N_MOVES:
                # Using a move with current active Pokemon
                probabilities = []
                for pkm_idx, pkm in enumerate(all_my_pokemons):
                    if pkm.hp > 0:
                        # Active pokemon: no penalty
                        # Non-active pokemon: switching in would cost a turn
                        switch_pen = (pkm_idx > 0)
                        p_win = self.evaluate_team_probability(pkm, opponent_active, switch_pen,
                                                                my_attack_stage, opp_defense_stage,
                                                                opp_attack_stage, my_defense_stage,
                                                                weather, hazard_damage,
                                                                my_speed_stage, opp_speed_stage)
                        probabilities.append(p_win)

                # Compute geometric mean of probabilities
                probs = [p for p in probabilities if p > 0]
                if len(probs) == 0:
                    value = 0.0
                else:
                    product = np.prod(probs)
                    value = product ** (1.0 / len(probs))

                if value > best_value:
                    best_value = value
                    best_action = action

            else:
                # Switching to a party Pokemon
                switch_idx = action - DEFAULT_PKM_N_MOVES
                if switch_idx >= len(my_team.party) or my_team.party[switch_idx].hp <= 0:
                    # Invalid switch or fainted
                    continue

                chosen_pkm = my_team.party[switch_idx]

                probabilities = []
                for pkm in all_my_pokemons:
                    if pkm.hp > 0:
                        # The chosen one to switch in gets penalty now,
                        # others also considered as needing penalty if brought in later.
                        switch_pen = True
                        p_win = self.evaluate_team_probability(pkm, opponent_active, switch_pen,
                                                                my_attack_stage, opp_defense_stage,
                                                                opp_attack_stage, my_defense_stage,
                                                                weather, hazard_damage,
                                                                my_speed_stage, opp_speed_stage)
                        probabilities.append(p_win)

                probs = [p for p in probabilities if p > 0]
                if len(probs) == 0:
                    value = 0.0
                else:
                    product = np.prod(probs)
                    value = product ** (1.0 / len(probs))

                if value > best_value:
                    best_value = value
                    best_action = action

        return best_action
