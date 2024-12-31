import sys
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
        stab = 1.5 if move_type == user_type else 1.0
        if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or \
           (move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
            w_mult = 1.5
        elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or \
             (move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
            w_mult = 0.5
        else:
            w_mult = 1.0

        stage_diff = attack_stage - defense_stage
        stage_multiplier = (
            (stage_diff + 2.0) / 2.0 if stage_diff >= 0
            else 2.0 / (abs(stage_diff) + 2.0)
        )

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
        Calculate (damage, accuracy) if 'attacker' uses move index = move_idx.
        If move is non-damaging (power=0), returns (0.0, 1.0).
        """
        move = attacker.moves[move_idx]
        dmg = self.estimate_damage(
            move.type, attacker.type, move.power,
            defender.type, atk_stage, def_stage, weather
        )
        return dmg, move.acc

    def duel_probability(self, H1, p1, d1, H2, p2, d2, skip_first=False):
        """
        Returns the probability that 'Pkm1' (with HP=H1, accuracy=p1, damage=d1)
        eventually kill Pkm2 (HP=H2, accuracy=p2, damage=d2).

        If skip_first=True, then on *Pkm1's* very first attack, it deals 0 damage (only once).
        The rest of the duel proceeds as normal.

        Uses top-down memoized recursion (dynamic programming) plus the closed-form
        solution for P(x,y) in terms of sub-states, so we do NOT get infinite recursion.
        """

        # --- Quick logic checks ---
        if H1 <= 0:
            return 0.0
        if H2 <= 0:
            return 1.0
        if d1 <= 0:
            return 0.0  # You can never deal damage => 0% chance to kill enemy
        if d2 <= 0:
            return 1.0  # Enemy can never deal damage => you eventually kill them w.p. 1

        # -------------------------------------------------------------------------
        # 1) Define a memo array for the recursion:
        #    We'll store P(x, y) in memo[x, y].
        #    A value >= 0.0 means "already computed."
        # -------------------------------------------------------------------------
        memo = np.full((H1 + 1, H2 + 1), -1.0, dtype=np.float64)

        def getP(x, y):
            """
            Recursive function returning the probability Pkm1 eventually wins
            starting with 'x' HP for Pkm1, 'y' HP for Pkm2,
            *after* any special first-turn logic has been handled.

            Implements the 4-case closed-form from Sympy:

              Sub-case A: (x <= d2, y <= d1):
                 P(x,y) = p1*(p2 - 2) / [ 2 * (p1*p2 - p1 - p2) ]

              Sub-case B: (x > d2, y <= d1):
                 P(x,y) = [ -P(x-d2, y)*p1*p2 + P(x-d2, y)*p2 + p1 ]
                          / [ -p1*p2 + p1 + p2 ]

              Sub-case C: (x <= d2, y > d1):
                 P(x,y) = P(x, y-d1)*p1*(p2 - 1) / [ p1*p2 - p1 - p2 ]

              Sub-case D: (x > d2, y > d1):
                 P(x,y) = [ P(x-d2, y-d1)*p1*p2
                            - P(x, y-d1)*p1*p2 + P(x, y-d1)*p1
                            - P(x-d2, y)*p1*p2 + P(x-d2, y)*p2 ]
                          / [ -p1*p2 + p1 + p2 ]
            """

            # --- Boundary checks ---
            if x <= 0:
                return 0.0
            if y <= 0:
                return 1.0
            if memo[x, y] >= 0.0:
                return memo[x, y]

            # --- Sub-case logic ---
            if x <= d2 and y <= d1:
                # Sub-case A: No sub-states needed; direct formula
                #    P(x,y) = p1*(p2 - 2) / [ 2*(p1*p2 - p1 - p2) ]
                numerator   = p1*(p2 - 2)
                denominator = 2.0*(p1*p2 - p1 - p2)
                val = numerator / denominator

            elif x > d2 and y <= d1:
                # Sub-case B: We need P(x-d2, y) = Pxd2_y
                Pxd2_y = getP(x - d2, y)  # boundary logic handles x-d2 <= 0
                #    P(x,y) = [ -Pxd2_y * p1*p2 + Pxd2_y*p2 + p1 ]
                #              / [ -p1*p2 + p1 + p2 ]
                numerator = -Pxd2_y*p1*p2 + Pxd2_y*p2 + p1
                denominator = -p1*p2 + p1 + p2
                val = numerator / denominator

            elif x <= d2 and y > d1:
                # Sub-case C: We need P(x, y-d1) = Px_yd1
                Px_yd1 = getP(x, y - d1)
                #    P(x,y) = Px_yd1 * p1*(p2 - 1) / [ p1*p2 - p1 - p2 ]
                numerator = Px_yd1 * p1 * (p2 - 1)
                denominator = (p1*p2 - p1 - p2)
                val = numerator / denominator

            else:  # x > d2 and y > d1
                # Sub-case D: We need all three sub-states:
                Pxd2_yd1  = getP(x - d2, y - d1)
                Px_yd1    = getP(x,      y - d1)
                Pxd2_y    = getP(x - d2, y)
                # Formula:
                #   [ Pxd2_yd1*p1*p2
                #     - Px_yd1*p1*p2 + Px_yd1*p1
                #     - Pxd2_y*p1*p2 + Pxd2_y*p2
                #   ] / [ -p1*p2 + p1 + p2 ]
                numerator = (
                    Pxd2_yd1*p1*p2
                    - Px_yd1*p1*p2  + Px_yd1*p1
                    - Pxd2_y*p1*p2  + Pxd2_y*p2
                )
                denominator = -p1*p2 + p1 + p2
                val = numerator / denominator

            memo[x, y] = val
            return val

        # -------------------------------------------------------------------------
        # 2) If skip_first=False, we just compute getP(H1, H2) directly.
        #    If skip_first=True, do one "special" turn where Pkm1's first attack
        #    deals 0 damage, then Pkm2 responds once, and only then do we
        #    proceed to the normal recursion.
        # -------------------------------------------------------------------------
        if not skip_first:
            return getP(H1, H2)
        else:
            # "First move" = Pkm1 attacks for 0 damage, then Pkm2 attacks if it's alive.
            # If Pkm1 is still alive, then we revert to normal recursion from new HPs.

            # 1) Pkm1 does 0 damage -> Pkm2 HP stays = H2
            # 2) Pkm2 attacks with probability p2
            #     - If (H1 <= d2) and it hits => Pkm1 dies => 0% chance
            #     - If (H1 >  d2) and it hits => new state is (H1 - d2, H2)
            #     - If Pkm2 misses => new state is (H1, H2)
            prob_if_enemy_hits = getP(H1 - d2, H2) if (H1 > d2) else 0.0
            prob_if_enemy_miss = getP(H1, H2)

            # Weighted by the chance the enemy hits or misses
            return p2 * prob_if_enemy_hits + (1.0 - p2) * prob_if_enemy_miss

    def get_action(self, g: PkmBattleEnv) -> int:
        """
        1) We assume the opponent does NOT switch. He only picks among his 4 moves.
           Similarly, we do NOT switch *for the sake of the enemy's best-move calculation*.
        2) We find the 'enemy best response' for each of our 4 possible moves:
           For each (our_move, enemy_move), we compute the probability that we eventually win
           using the DP-based 'win_probability'. The enemy picks the move that minimizes
           our probability of winning.
           => That gives us 4 results (one for each of our moves).
        3) Next, for possible switches among our *party* Pokémon, we compute the probability
           of winning if we do switch. In that scenario, the *first turn we effectively do 0 dmg
           with 100% accuracy* (the "switch penalty"): we pass `skip_first=True` so that our
           first attack deals 0. The enemy again picks among 4 moves to minimize our chance.
        4) We pick the best among these (4 normal moves + possible switches).
        5) If all options yield extremely low probability (< EPSILON), we fallback to
           the single most damaging move from our active Pokémon.
        """

        # Basic references
        my_team, opp_team = g.teams[0], g.teams[1]
        weather = g.weather.condition

        # Our stage
        my_atk = my_team.stage[PkmStat.ATTACK]
        my_def = my_team.stage[PkmStat.DEFENSE]
        my_spd = my_team.stage[PkmStat.SPEED]
        # Opponent stage
        opp_atk = opp_team.stage[PkmStat.ATTACK]
        opp_def = opp_team.stage[PkmStat.DEFENSE]
        opp_spd = opp_team.stage[PkmStat.SPEED]

        # Active Pokémon
        my_active = my_team.active
        opp_active = opp_team.active

        # ------------------------------------------------------------
        # 1) Gather stats: current HP, accuracy, damage for *my_active* 
        #    for each of the 4 moves, and for the opponent as well.
        # ------------------------------------------------------------
        # For each of my 4 moves:
        my_move_stats = []
        for move_i in range(DEFAULT_PKM_N_MOVES):
            dmg_i, acc_i = self.single_move_damage_acc(
                my_active, opp_active, move_i, my_atk, opp_def, weather
            )
            # Round damage down to integer for DP (or keep float & ceil in HP). 
            # To keep it simple, let's cast HP as int(ceil()), damage as int(ceil())) 
            # in the win_probability calls. Accuracy stays as float.
            my_move_stats.append((dmg_i, acc_i))

        # Opponent's 4 moves:
        opp_move_stats = []
        for move_j in range(DEFAULT_PKM_N_MOVES):
            dmg_j, acc_j = self.single_move_damage_acc(
                opp_active, my_active, move_j, opp_atk, my_def, weather
            )
            opp_move_stats.append((dmg_j, acc_j))

        my_hp_int = int(np.ceil(my_active.hp))
        opp_hp_int = int(np.ceil(opp_active.hp))

        #print(f"DEBUG: My Active HP={my_hp_int}, Opponent Active HP={opp_hp_int}")

        # -----------------------------------------------------------------
        # 2) For each of my moves (0..3), compute the "worst-case" probability
        #    given the enemy picks its best counter-move. We do NOT consider 
        #    switching for either side in this step, just a pure 4x4 scenario.
        # -----------------------------------------------------------------
        move_win_probs = [0.0] * DEFAULT_PKM_N_MOVES
        for i in range(DEFAULT_PKM_N_MOVES):
            (my_dmg, my_acc) = my_move_stats[i]
            # We'll see how the enemy can respond:
            # They pick the move that yields the *lowest* probability for me.
            #print(f"DEBUG: Considering Move {i} with damage={my_dmg}, accuracy={my_acc}")
            worst_for_me = 1.0
            for j in range(DEFAULT_PKM_N_MOVES):
                (opp_dmg, opp_acc) = opp_move_stats[j]
                # Probability I eventually win if I do (my_dmg, my_acc) vs. (opp_dmg, opp_acc).
                p_win = self.duel_probability(
                    my_hp_int, my_acc, int(np.ceil(my_dmg)),
                    opp_hp_int, opp_acc, int(np.ceil(opp_dmg)),
                    skip_first=False
                )
                # DEBUG PRINT
                #print(f"DEBUG: Move {i} vs Enemy Move {j} => P(win)={p_win}")
                if p_win < worst_for_me:
                    worst_for_me = p_win
            move_win_probs[i] = worst_for_me
            # DEBUG PRINT
            #print(f"DEBUG: Worst-case P(win) for our Move {i} = {worst_for_me}")

        # -----------------------------------------------------------------
        # 3) Consider switching to each Pokemon in my party (if non-fainted).
        #    Now my first attack from that new Pokemon is 0 dmg, 100% acc.
        #    The enemy again picks among 4 moves to minimize my chance.
        # -----------------------------------------------------------------
        switch_actions = {}
        for idx, pkm in enumerate(my_team.party):
            if pkm.hp <= 0.0:
                continue  # fainted
            # Just index the "action" as 4 + idx in standard code
            action_id = DEFAULT_PKM_N_MOVES + idx

            # We'll compute worst-case across enemy's 4 moves for the new pkm:
            new_my_hp = int(np.ceil(pkm.hp))
            # We find that pkm's best damaging move (or we can test all moves?). 
            # But the user specifically wants "for every possible move of every party pokemon."
            # So let's do 4x4 again. BUT skip_first = True for my side.
            best_move_probability_for_this_switch = 0.0

            # For each move_i in [0..3] for my new pkm
            local_best_for_switch = 0.0
            for move_i in range(DEFAULT_PKM_N_MOVES):
                dmg_i, acc_i = self.single_move_damage_acc(
                    pkm, opp_active, move_i, my_atk, opp_def, weather
                )
                # Now find worst-case among enemy's 4 moves
                local_worst = 1.0
                for move_j in range(DEFAULT_PKM_N_MOVES):
                    (opp_dmg, opp_acc) = opp_move_stats[move_j]
                    p_win = self.duel_probability(
                        new_my_hp, acc_i, int(np.ceil(dmg_i)),
                        opp_hp_int, opp_acc, int(np.ceil(opp_dmg)),
                        skip_first=True  # Because we switched in
                    )
                    # DEBUG PRINT
                    #print(f"DEBUG: Switch to Pokemon {idx}, Move {move_i} vs Enemy Move {move_j} => P(win)={p_win}")
                    if p_win < local_worst:
                        local_worst = p_win
                    #if local_worst < local_best_for_switch:
                    #    print(f"DEBUG: Skip checking other moves for this switch, as it's already worse.")
                    #    break
                    #if local_worst <= self.EPSILON:
                    #    print(f"DEBUG: Skip checking other moves for this switch, as it's already very low.")
                    #    break

                # After considering all enemy moves for this switch and move_i
                if local_worst > local_best_for_switch:
                    local_best_for_switch = local_worst

                #print(f"DEBUG: Worst-case P(win) for switching to Pokemon {idx}, Move {move_i} = {local_worst}")
            # 'local_best_for_switch' is the best worst-case probability after switching and enemy's best response
            switch_actions[action_id] = local_best_for_switch
            # DEBUG PRINT
            #print(f"DEBUG: Best worst-case P(win) for switching to Pokemon {idx} = {local_best_for_switch}")

        # -----------------------------------------------------------------
        # 4) Combine the results: pick the highest among (move_win_probs + switch_actions).
        # -----------------------------------------------------------------
        best_move_value = max(move_win_probs) if move_win_probs else 0.0
        best_move_idx = int(np.argmax(move_win_probs)) if move_win_probs else 0

        best_switch_value = -1.0
        best_switch_id = None
        if switch_actions:
            # pick the max
            for k, val in switch_actions.items():
                if val > best_switch_value:
                    best_switch_value = val
                    best_switch_id = k
                # DEBUG PRINT
                #print(f"DEBUG: Comparing switch action {k} with P(win)={val} against current best_switch_value={best_switch_value}")

        # We compare best_move_value vs best_switch_value
        final_best_value = best_move_value
        final_best_action = best_move_idx
        if best_switch_id is not None and best_switch_value > final_best_value:
            final_best_value = best_switch_value
            final_best_action = best_switch_id
            # DEBUG PRINT
            #print(f"DEBUG: Selected to switch to action {best_switch_id} with P(win)={best_switch_value}")
        #else:
            # DEBUG PRINT
            #print(f"DEBUG: Selected to use Move {best_move_idx} with P(win)={best_move_value}")

        # 5) Fallback if final_best_value ~ 0
        if final_best_value <= self.EPSILON:
            # pick the single most damaging move from active pkm
            best_exp_dmg = -1.0
            best_exp_dmg_idx = 0
            for m_i in range(DEFAULT_PKM_N_MOVES):
                dmg_i, acc_i = self.single_move_damage_acc(
                    my_active, opp_active, m_i, my_atk, opp_def, weather
                )
                exp_dmg = dmg_i * acc_i
                if exp_dmg > best_exp_dmg:
                    best_exp_dmg = exp_dmg
                    best_exp_dmg_idx = m_i
            # DEBUG PRINT
            #print(f"DEBUG: Fallback to most (expected) damaging Move {best_exp_dmg_idx} with expected damage={best_exp_dmg}")
            return best_exp_dmg_idx

        return final_best_action
