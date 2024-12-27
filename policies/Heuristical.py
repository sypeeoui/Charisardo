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
    Heuristic agent that does a one-turn lookahead. 
    
    Features:
    - If the chosen action is a move (0..3), we actually use that move index for 
      the active Pokémon's immediate attack rather than always using the "best move".
    - If the chosen action is a switch (>=4), we switch in that Pokémon (if valid) 
      and assume it and any other non-active teammates will use their own best move 
      when they appear in battle.
    - We compute a final "winning probability" for each possible action by:
      1. Calculating the "winning probability" of each of our non-fainted Pokémon 
         (including the currently active one and those in the party).
      2. Combining these probabilities via their geometric mean 
         (the nth root of their product, where n = number of non-fainted Pokémon).
      3. Selecting the action that yields the highest final value.
    - **Fallback**: if all actions end up with a computed 0.0, 
      we choose the single most damaging move from our active Pokémon 
      rather than picking among all actions that tie at 0.
    
    Explanation of Probability Model:
    - For each Pokémon, we compare how many moves we need to KO the opponent 
      vs. how many moves the opponent needs to KO us.
    - We factor in move accuracies (e.g., 0.8 => 80%).
    - If a Pokémon is non-active, we apply a "switch penalty" (i.e., hazard damage 
      plus effectively wasting one extra turn before it can start attacking).
    - We then apply a simple heuristic formula: 
        P_win = P_ours * (1 - P_theirs^m_t) 
      adjusted for differences in the number of moves (m_o vs m_t) and whether 
      we are faster or not.
    - This can lead to a computed 0.0 if, for example, the opponent never misses 
      and needs fewer (or equal) moves than we do. Hence the new fallback logic.
    """

    EPSILON = 1e-9  # tie-breaking tolerance

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
        Estimate the damage for a single move, factoring in STAB, weather, and stage differences.
        """
        # Same-type attack bonus
        stab = 1.5 if move_type == user_type else 1.0

        # Weather
        if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or \
           (move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
            w = 1.5
        elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or \
             (move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
            w = 0.5
        else:
            w = 1.0

        # Stage multiplier
        stage_diff = attack_stage - defense_stage
        if stage_diff >= 0:
            stage_multiplier = (stage_diff + 2.0) / 2.0
        else:
            stage_multiplier = 2.0 / (abs(stage_diff) + 2.0)

        # Type chart multiplier
        type_mult = TYPE_CHART_MULTIPLIER[move_type][opp_type]

        return type_mult * stab * w * stage_multiplier * move_power

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
        Returns (damage, accuracy) for a *specific* move index.
        If the move is non-damaging (power=0), returns (0.0, 1.0).
        """
        move = attacker.moves[move_idx]
        dmg = self.estimate_damage(
            move.type, attacker.type, move.power,
            defender.type, atk_stage, def_stage, weather
        )
        # If your engine provides move.acc in [0,1], just use move.acc. 
        # If in [0..100], you'd do (move.acc / 100.0). 
        # Example: move.acc=0.8 => 80% accuracy
        acc = move.acc
        return dmg, acc

    def best_move_damage_acc(
        self, 
        attacker: Pkm, 
        defender: Pkm,
        atk_stage: float, 
        def_stage: float,
        weather: WeatherCondition
    ) -> (float, float):
        """
        Returns (damage, accuracy) for the single best (max-damage) move 
        among the attacker's move set.
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
        if best_dmg == 0.0:
            return 0.0, 1.0
        return best_dmg, best_acc

    def moves_to_ko(self, hp: float, dmg: float) -> int:
        """
        Number of hits needed to reduce hp to 0 or less, given 'dmg' per hit.
        """
        if dmg <= 0:
            return 9999
        return int(np.ceil(hp / dmg))

    def winning_probability(
        self, 
        m_o: int,    # moves we need to KO
        m_t: int,    # moves opponent needs to KO
        p_ours: float, 
        p_theirs: float,
        we_are_faster: bool
    ) -> float:
        """
        Simple heuristic for winning probability:
        If we need fewer moves (m_o < m_t), we have a better chance,
        else if we need more (m_o > m_t), we rely on them missing, etc.
        """
        if m_o < m_t:
            if we_are_faster:
                return p_ours
            else:
                return p_ours * (1.0 - p_theirs ** m_o)
        elif m_o == m_t:
            if we_are_faster:
                return p_ours
            else:
                return p_ours * (1.0 - p_theirs ** m_o)
        else:  # m_o > m_t
            return p_ours * (1.0 - p_theirs ** m_t)

    def is_faster(self, my_speed_stage: float, opp_speed_stage: float) -> bool:
        """
        Return True if we are faster or equal in speed.
        """
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
        Evaluate the probability that 'chosen_pkm' eventually beats 'opponent_pkm'.

        - If use_specific_move=True, we forcibly use 'move_idx' for the chosen_pkm.
        - If use_specific_move=False, we pick the best move for chosen_pkm.
        - If switch_penalty=True, the chosen pkm takes hazard damage and 
          effectively wastes one extra turn (m_o += 1).
        - We factor in the opponent's best move as well.
        """
        us = deepcopy(chosen_pkm)
        them = deepcopy(opponent_pkm)

        # Apply hazard damage if switching
        if switch_penalty:
            us.hp = max(0.0, us.hp - hazard_damage)
        if us.hp <= 0:
            return 0.0

        # 1) Our damage & accuracy
        if use_specific_move:
            # forced to use a specific move index
            my_dmg, my_acc = self.single_move_damage_acc(
                us, them, move_idx,
                my_atk_stage, opp_def_stage,
                weather
            )
        else:
            # pick best move
            my_dmg, my_acc = self.best_move_damage_acc(
                us, them, my_atk_stage, opp_def_stage,
                weather
            )

        # 2) Opponent's best damage & accuracy
        opp_dmg, opp_acc = self.best_move_damage_acc(
            them, us, opp_atk_stage, my_def_stage, weather
        )

        # If we do no damage, 0 chance to KO
        if my_dmg == 0.0:
            return 0.0

        # If opponent does no damage, we only need to land enough hits
        if opp_dmg == 0.0:
            moves_needed = self.moves_to_ko(them.hp, my_dmg)
            if switch_penalty:
                moves_needed += 1
            return (my_acc ** moves_needed)

        # Moves needed for each side
        m_o = self.moves_to_ko(them.hp, my_dmg)  # ours
        m_t = self.moves_to_ko(us.hp, opp_dmg)   # theirs
        if switch_penalty:
            m_o += 1

        p_ours = (my_acc ** m_o)
        p_theirs = (opp_acc ** m_t)
        faster = self.is_faster(my_speed_stage, opp_speed_stage)
        return self.winning_probability(m_o, m_t, p_ours, p_theirs, faster)

    def get_action(self, g: PkmBattleEnv) -> int:
        """
        Decide the best action among 0..3 moves or 4..5 switches
        by comparing the final geometric mean of "win probabilities"
        for each action. If all result in 0.0, fallback to the
        single most damaging move from the active Pokémon.
        """
        actions = range(DEFAULT_N_ACTIONS)

        # Team references
        my_team = g.teams[0]
        opp_team = g.teams[1]
        weather = g.weather.condition

        # Hazard damage (if you have real hazards, override here)
        hazard_damage = 0.0

        # Our team stage
        my_atk_stage = my_team.stage[PkmStat.ATTACK]
        my_def_stage = my_team.stage[PkmStat.DEFENSE]
        my_spd_stage = my_team.stage[PkmStat.SPEED]

        # Opponent stage
        opp_atk_stage = opp_team.stage[PkmStat.ATTACK]
        opp_def_stage = opp_team.stage[PkmStat.DEFENSE]
        opp_spd_stage = opp_team.stage[PkmStat.SPEED]

        # Active
        my_active = my_team.active
        opp_active = opp_team.active

        # My full team
        all_my_pokemons = [my_active] + list(my_team.party)

        best_value = -1.0
        best_actions = []

        # 1) Evaluate each possible action
        for action in actions:
            if action < DEFAULT_PKM_N_MOVES:
                # "Use move #action" with the active Pokemon
                probabilities = []
                for idx, pkm in enumerate(all_my_pokemons):
                    if pkm.hp > 0.0:
                        if idx == 0:
                            # The currently active pkm is forced to use move #action
                            switch_pen = False
                            use_specific = True
                            move_idx = action
                        else:
                            # Non-active => would have to switch in + best move
                            switch_pen = True
                            use_specific = False
                            move_idx = 0  # dummy
                        
                        val = self.evaluate_team_probability(
                            chosen_pkm=pkm,
                            opponent_pkm=opp_active,
                            use_specific_move=use_specific,
                            move_idx=move_idx,
                            switch_penalty=switch_pen,
                            my_atk_stage=my_atk_stage,
                            opp_def_stage=opp_def_stage,
                            opp_atk_stage=opp_atk_stage,
                            my_def_stage=my_def_stage,
                            weather=weather,
                            hazard_damage=hazard_damage,
                            my_speed_stage=my_spd_stage,
                            opp_speed_stage=opp_spd_stage
                        )
                        probabilities.append(val)

                if probabilities:
                    product = np.prod(probabilities)
                    value = product ** (1.0 / len(probabilities))
                else:
                    value = 0.0

            else:
                # "Switch" => action - DEFAULT_PKM_N_MOVES in the party
                switch_idx = action - DEFAULT_PKM_N_MOVES
                if switch_idx >= len(my_team.party):
                    continue
                chosen_pkm = my_team.party[switch_idx]
                if chosen_pkm.hp <= 0:
                    continue

                probabilities = []
                # All my pokemons are considered with a penalty if they come in
                for pkm in all_my_pokemons:
                    if pkm.hp > 0.0:
                        val = self.evaluate_team_probability(
                            chosen_pkm=pkm,
                            opponent_pkm=opp_active,
                            use_specific_move=False,
                            move_idx=0,
                            switch_penalty=True,
                            my_atk_stage=my_atk_stage,
                            opp_def_stage=opp_def_stage,
                            opp_atk_stage=opp_atk_stage,
                            my_def_stage=my_def_stage,
                            weather=weather,
                            hazard_damage=hazard_damage,
                            my_speed_stage=my_spd_stage,
                            opp_speed_stage=opp_spd_stage
                        )
                        probabilities.append(val)

                if probabilities:
                    product = np.prod(probabilities)
                    value = product ** (1.0 / len(probabilities))
                else:
                    value = 0.0

            # 2) Tie-breaking
            if value > best_value + self.EPSILON:
                best_value = value
                best_actions = [action]
            elif abs(value - best_value) <= self.EPSILON:
                best_actions.append(action)

        #print(f"best_actions = {best_actions}, best_value={best_value}")
        # 3) If everything is 0.0, fallback to "max immediate damage" for the active pkm
        if abs(best_value) <= self.EPSILON:
            # We never found anything better than 0.0, so pick the single best immediate damage move
            # from our active Pokemon (move index 0..3).
            best_move_idx = 0
            best_damage = -1.0
            for move_idx in range(DEFAULT_PKM_N_MOVES):
                dmg, _ = self.single_move_damage_acc(
                    my_active, opp_active, move_idx,
                    my_atk_stage, opp_def_stage,
                    weather
                )
                if dmg > best_damage:
                    best_damage = dmg
                    best_move_idx = move_idx
            return best_move_idx

        # Otherwise pick randomly among top actions
        chosen_action = random.choice(best_actions)
        return chosen_action
