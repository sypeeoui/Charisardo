import numpy as np
import random
from copy import deepcopy

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import (
    DEFAULT_PKM_N_MOVES,
    DEFAULT_N_ACTIONS,
    TYPE_CHART_MULTIPLIER
)
from vgc.datatypes.Objects import Pkm
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
from vgc.engine.PkmBattleEnv import PkmBattleEnv


class Heuristical(BattlePolicy):
    """
    A one-turn lookahead heuristic that selects among up to four moves (0..3) or switches (4..5).

    Overview:
    1. If we choose an action < 4, we use exactly that move index with our active Pokémon.
       Non-active Pokémon (party members) are assumed to switch in (with a penalty) if they
       eventually fight, and will use their *best* damaging move when they do.
    2. If we choose a switch action (>=4), we bring in another Pokémon (with a penalty).
       After switching, that newly active Pokémon and any other non-active teammates
       are assumed to use their own best move in future.
    3. For each of our non-fainted Pokémon, we compute a "winning probability" against
       the opponent's active Pokémon via a simple formula:
       - If a Pokémon needs m_o moves to KO and the opponent needs m_t moves to KO in return,
         we consider move accuracies and whether we're faster.
         Then we produce a probability of eventually winning that head-to-head scenario.
       - If the Pokémon isn't active yet, it pays a one-turn "switch penalty" that effectively
         increases m_o by 1 and applies any hazard damage (set to 0.0 here, but you can adjust).
    4. We combine each non-fainted Pokémon's probability with the geometric mean:
          final_value = ( Π probabilities )^(1 / N )
       Then pick the action that yields the highest final_value.
    5. If all actions yield 0.0 (meaning our simple model says we have no chance),
       we fallback to picking the single most damaging move for our active Pokémon
       instead of a random 0-probability choice.

    This is *not* a full tree search. It's a greedy, one-turn-ahead calculation, but
    includes a fallback if we appear to have no winning chance. 
    """

    EPSILON = 1e-9  # Float comparison tolerance for tie-breaking

    def estimate_damage(
        self,
        move_type: PkmType,
        user_type: PkmType,
        move_power: float,
        opp_type: PkmType,
        attack_stage: float,
        defense_stage: float,
        weather: WeatherCondition
    ) -> float:
        """
        Estimate the per-hit damage a move does, factoring in:
        - STAB (same-type attack bonus)
        - Weather influence
        - Stage differences (attack - defense)
        - Type chart multipliers
        """
        # STAB
        stab = 1.5 if move_type == user_type else 1.0

        # Weather-based multiplier
        if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or \
           (move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
            w_mult = 1.5
        elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or \
             (move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
            w_mult = 0.5
        else:
            w_mult = 1.0

        # Attack vs. Defense stage
        stage_diff = attack_stage - defense_stage
        stage_multiplier = (
            (stage_diff + 2.0) / 2.0 if stage_diff >= 0
            else 2.0 / (abs(stage_diff) + 2.0)
        )

        # Type chart multiplier
        type_mult = TYPE_CHART_MULTIPLIER[move_type][opp_type]

        return type_mult * stab * w_mult * stage_multiplier * move_power

    def single_move_damage_acc(
        self,
        attacker: Pkm,
        defender: Pkm,
        move_idx: int,
        atk_stage: float,
        def_stage: float,
        weather: WeatherCondition
    ) -> (float, float):
        """
        Calculate (damage, accuracy) if 'attacker' uses a *specific* move index (move_idx).
        If the move is non-damaging (power=0), returns (0.0, 1.0).
        """
        move = attacker.moves[move_idx]
        dmg = self.estimate_damage(
            move.type, attacker.type, move.power,
            defender.type, atk_stage, def_stage, weather
        )
        return dmg, move.acc

    def best_move_damage_acc(
        self,
        attacker: Pkm,
        defender: Pkm,
        atk_stage: float,
        def_stage: float,
        weather: WeatherCondition
    ) -> (float, float):
        """
        Calculate (damage, accuracy) for the single best (max-damage) move
        among 'attacker' moves.
        """
        best_dmg = 0.0
        best_acc = 1.0
        for m in attacker.moves:
            dmg = self.estimate_damage(
                m.type, attacker.type, m.power,
                defender.type, atk_stage, def_stage, weather
            )
            if dmg > best_dmg:
                best_dmg = dmg
                best_acc = m.acc
        return (best_dmg, best_acc) if best_dmg > 0 else (0.0, 1.0)

    def moves_to_ko(self, hp: float, dmg: float) -> int:
        """Returns how many hits of 'dmg' are needed to reduce 'hp' to 0 or below."""
        return 9999 if dmg <= 0 else int(np.ceil(hp / dmg))

    def winning_probability(
        self,
        m_o: int,    # Moves we need to KO
        m_t: int,    # Moves opponent needs to KO
        p_ours: float,
        p_theirs: float,
        we_are_faster: bool
    ) -> float:
        """
        Simple heuristic probability of winning a head-to-head:
        - If we need fewer moves (m_o < m_t) and are faster, 
          we often get a straightforward advantage.
        - Otherwise, we rely on the chance the opponent misses.
        """
        if m_o < m_t:
            return p_ours if we_are_faster else p_ours * (1.0 - p_theirs**m_o)
        elif m_o == m_t:
            return p_ours if we_are_faster else p_ours * (1.0 - p_theirs**m_o)
        else:  # m_o > m_t
            return p_ours * (1.0 - p_theirs**m_t)

    def is_faster(self, my_speed_stage: float, opp_speed_stage: float) -> bool:
        """Return True if our speed stage is >= opponent's (we act first on ties)."""
        return my_speed_stage >= opp_speed_stage

    def evaluate_team_probability(
        self,
        chosen_pkm: Pkm,
        opponent_pkm: Pkm,
        use_specific_move: bool,
        move_idx: int,
        switch_penalty: bool,
        my_atk_stage: float,
        opp_def_stage: float,
        opp_atk_stage: float,
        my_def_stage: float,
        weather: WeatherCondition,
        hazard_damage: float,
        my_speed_stage: float,
        opp_speed_stage: float
    ) -> float:
        """
        Compute how likely 'chosen_pkm' is to eventually beat 'opponent_pkm'
        under our heuristic. Key points:
          - If use_specific_move=True, the pkm must use 'move_idx'.
          - Otherwise, it picks its best move.
          - If switch_penalty=True, it takes hazard_damage and effectively wastes +1 move (m_o += 1).
          - The opponent picks its best move.
          - We check how many moves each side needs, factor in speed to see who hits first,
            and combine accuracies accordingly.
        """
        us = deepcopy(chosen_pkm)
        them = deepcopy(opponent_pkm)

        # Apply hazard damage if switching
        if switch_penalty:
            us.hp = max(0.0, us.hp - hazard_damage)
        if us.hp <= 0:
            return 0.0

        # Our moves
        if use_specific_move:
            my_dmg, my_acc = self.single_move_damage_acc(
                us, them, move_idx, my_atk_stage, opp_def_stage, weather
            )
        else:
            my_dmg, my_acc = self.best_move_damage_acc(
                us, them, my_atk_stage, opp_def_stage, weather
            )

        # Opponent's best move
        opp_dmg, opp_acc = self.best_move_damage_acc(
            them, us, opp_atk_stage, my_def_stage, weather
        )
        if my_dmg == 0.0:  # no way to KO
            return 0.0
        if opp_dmg == 0.0:  # opponent can't damage us at all
            needed = self.moves_to_ko(them.hp, my_dmg)
            return (my_acc ** (needed + (1 if switch_penalty else 0)))

        # Moves needed by each side
        m_o = self.moves_to_ko(them.hp, my_dmg)
        m_t = self.moves_to_ko(us.hp, opp_dmg)
        if switch_penalty:
            m_o += 1

        p_ours = (my_acc ** m_o)
        p_theirs = (opp_acc ** m_t)
        faster = self.is_faster(my_speed_stage, opp_speed_stage)
        return self.winning_probability(m_o, m_t, p_ours, p_theirs, faster)

    def get_action(self, g: PkmBattleEnv) -> int:
        """
        Main decision method. 
        1. We evaluate each action (0..3 => moves, 4..5 => switches) by computing
           the geometric mean of "team-winning-probabilities" (for each not-fainted Pokémon).
        2. We pick the action(s) with the highest final value; tie-break randomly.
        3. If all values remain 0.0, we pick the single highest-damage move 
           on our current active Pokémon as a fallback.
        """
        # Basic references
        my_team, opp_team = g.teams[0], g.teams[1]
        weather = g.weather.condition
        hazard_damage = 0.0  # adjust if you have real hazards

        # Our stages
        my_atk, my_def, my_spd = (my_team.stage[s] for s in (PkmStat.ATTACK, PkmStat.DEFENSE, PkmStat.SPEED))
        # Opponent stages
        opp_atk, opp_def, opp_spd = (opp_team.stage[s] for s in (PkmStat.ATTACK, PkmStat.DEFENSE, PkmStat.SPEED))

        # Active Pokémon
        my_active = my_team.active
        opp_active = opp_team.active
        # Full team = [active] + party
        all_my_pokemons = [my_active] + list(my_team.party)

        best_value = -1.0
        best_actions = []

        # 1) Evaluate each possible action
        for action in range(DEFAULT_N_ACTIONS):
            if action < DEFAULT_PKM_N_MOVES:
                # Using move #action with active Pokémon
                probabilities = []
                for idx, pkm in enumerate(all_my_pokemons):
                    if pkm.hp > 0.0:
                        # If idx=0 => active => forced to use 'action'
                        # If idx>0 => not active => must switch in => best move
                        use_specific = (idx == 0)
                        switch_penalty = (idx > 0)
                        val = self.evaluate_team_probability(
                            chosen_pkm=pkm,
                            opponent_pkm=opp_active,
                            use_specific_move=use_specific,
                            move_idx=action,
                            switch_penalty=switch_penalty,
                            my_atk_stage=my_atk,
                            opp_def_stage=opp_def,
                            opp_atk_stage=opp_atk,
                            my_def_stage=my_def,
                            weather=weather,
                            hazard_damage=hazard_damage,
                            my_speed_stage=my_spd,
                            opp_speed_stage=opp_spd
                        )
                        probabilities.append(val)
                value = (np.prod(probabilities) ** (1/len(probabilities))) if probabilities else 0.0

            else:
                # Switch to party Pokémon => action - DEFAULT_PKM_N_MOVES
                switch_idx = action - DEFAULT_PKM_N_MOVES
                if switch_idx >= len(my_team.party):
                    continue  # invalid
                chosen_pkm = my_team.party[switch_idx]
                if chosen_pkm.hp <= 0:
                    continue  # fainted

                # If we switch, *all* our Pokémons pay penalty if they come in
                probabilities = []
                for pkm in all_my_pokemons:
                    if pkm.hp > 0.0:
                        val = self.evaluate_team_probability(
                            chosen_pkm=pkm,
                            opponent_pkm=opp_active,
                            use_specific_move=False,
                            move_idx=0,
                            switch_penalty=True,
                            my_atk_stage=my_atk,
                            opp_def_stage=opp_def,
                            opp_atk_stage=opp_atk,
                            my_def_stage=my_def,
                            weather=weather,
                            hazard_damage=hazard_damage,
                            my_speed_stage=my_spd,
                            opp_speed_stage=opp_spd
                        )
                        probabilities.append(val)
                value = (np.prod(probabilities) ** (1/len(probabilities))) if probabilities else 0.0

            # 2) Tie-breaking with EPSILON
            if value > best_value + self.EPSILON:
                best_value = value
                best_actions = [action]
            elif abs(value - best_value) <= self.EPSILON:
                best_actions.append(action)

        # 3) Fallback if everything is 0.0
        if abs(best_value) <= self.EPSILON:
            # Pick the single most damaging move from the active Pokémon
            best_move_idx, best_dmg = 0, -1.0
            for move_i in range(DEFAULT_PKM_N_MOVES):
                dmg, _ = self.single_move_damage_acc(
                    my_active, opp_active, move_i,
                    my_atk, opp_def, weather
                )
                if dmg > best_dmg:
                    best_dmg = dmg
                    best_move_idx = move_i
            return best_move_idx

        # Otherwise, pick randomly among best actions
        return random.choice(best_actions)
