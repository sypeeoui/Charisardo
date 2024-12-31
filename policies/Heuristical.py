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
    """
    A one-turn lookahead heuristic. 
    NOTE: We have **modified** the 'best_action' to remove the geometric-mean logic
    and instead use a turn-by-turn DP (win_probability) approach with a special
    "skip-first-attack" case for newly switched-in Pokémon, as described.
    All other methods and code remain the same.
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

    def win_probability(self, H1, p1, d1, H2, p2, d2, skip_first=False):
        """
        Computes the probability that a 'player' with (HP=H1, accuracy=p1, damage=d1)
        eventually wins a duel vs. an 'opponent' with (HP=H2, accuracy=p2, damage=d2),
        under the rules:
          - Each turn, exactly two attacks happen (one by you, one by the opponent).
          - The order of attacks each turn is random (0.5 probability each).
          - If skip_first=True, then on your *very first* attack, you deal 0 damage with 100% accuracy.
            (After that one 'null hit', your normal p1, d1 apply.)
        Returns a float in [0, 1].
        """

        # We store P[x,y] = probability that you eventually win if it is
        # the start of a turn and your HP is x, opponent's HP is y.
        # 0 <= x <= H1, 0 <= y <= H2
        P = np.zeros((H1 + 1, H2 + 1), dtype=np.float64)

        # Boundary conditions
        # If x=0 but y>0 => lose => P[0,y] = 0
        # If y=0 but x>0 => win => P[x,0] = 1
        P[0, :] = 0.0
        P[:, 0] = 1.0
        # by convention: P[0,0] = 0
        P[0, 0] = 0.0

        # Turn-based sub-functions: S1 => we attack first, S2 => opponent first
        # We modify S1 slightly if skip_first is still "active." We'll do this by a
        # single boolean that flips off after the first time we try to attack.

        # We'll store a special boolean array that indicates whether
        # "our first attack" is still in effect for each (x, y). To do that,
        # we add an extra dimension for the state of "skip_first" (0 or 1).
        # Let 0 => skip_first is *no longer* in effect, 1 => skip_first is still in effect.
        # So we store P[x][y][sf], where sf in {0,1}.
        P_3d = np.zeros((H1 + 1, H2 + 1, 2), dtype=np.float64)

        # Initialize boundary for both sf=0 and sf=1
        for x in range(H1 + 1):
            P_3d[x, 0, 0] = 1.0
            P_3d[x, 0, 1] = 1.0
        for y in range(H2 + 1):
            P_3d[0, y, 0] = 0.0
            P_3d[0, y, 1] = 0.0

        def S1(x, y, sf):
            """
            Probability we eventually win if WE attack first this turn,
            given state (x HP, y HP, skip_first_flag=sf).
            """
            # If skip_first_flag is 1, then we do 0 damage at 100% accuracy for *this* attack only.
            # Then we turn off skip_first_flag => 0 for subsequent states.
            if sf == 1:
                # We do a 'null' hit (0 dmg, 100% acc).
                # Then the opponent attacks if they're still alive.
                # Opponent hits us with probability p2:
                #   if x <= d2 => we die => prob=0
                #   else => new state => P_3d[x-d2, y, 0]
                # Opponent misses with probability (1-p2) => new state => P_3d[x, y, 0]
                return (
                    p2 * (0.0 if x <= d2 else P_3d[x - d2, y, 0]) +
                    (1 - p2) * P_3d[x, y, 0]
                )
            else:
                # normal attack (p1, d1)
                # with probability p1: we hit => if y <= d1 => we win => prob=1
                #                                 else => enemy attacks
                # with probability (1-p1): we miss => enemy attacks
                hit_part = p1 * (
                    1.0 if y <= d1 else (
                        p2 * (0.0 if x <= d2 else P_3d[x - d2, y - d1, 0]) +
                        (1 - p2) * P_3d[x, y - d1, 0]
                    )
                )
                miss_part = (1 - p1) * (
                    p2 * (0.0 if x <= d2 else P_3d[x - d2, y, 0]) +
                    (1 - p2) * P_3d[x, y, 0]
                )
                return hit_part + miss_part

        def S2(x, y, sf):
            """
            Probability we eventually win if the OPPONENT attacks first this turn,
            given state (x HP, y HP, skip_first_flag=sf).
            """
            # If the opponent attacks first, skip_first_flag doesn't affect them; it only affects us.
            # Opponent hits with probability p2 => if x <= d2 => we die => prob=0
            #                                  => else => we get to attack
            # Opponent misses with probability (1-p2) => we get to attack

            opp_hit_part = p2 * (
                0.0 if x <= d2 else S1(x - d2, y, sf)  # now it's our turn, same skip_first_flag
            )
            opp_miss_part = (1 - p2) * S1(x, y, sf)
            return opp_hit_part + opp_miss_part

        # Fill P_3d for x in [0..H1], y in [0..H2], sf in [0,1]
        for x in range(H1 + 1):
            for y in range(H2 + 1):
                for sf in [0, 1]:
                    if x == 0 and y > 0:
                        P_3d[x, y, sf] = 0.0
                    elif y == 0 and x > 0:
                        P_3d[x, y, sf] = 1.0
                    elif x == 0 and y == 0:
                        P_3d[x, y, sf] = 0.0
                    else:
                        # Mix of S1,S2 half the time each
                        P_3d[x, y, sf] = 0.5 * S1(x, y, sf) + 0.5 * S2(x, y, sf)

        # Final answer is P_3d[H1, H2, 1] if skip_first=True, else P_3d[H1, H2, 0]
        final_prob = float(P_3d[H1, H2, 1 if skip_first else 0])

        return final_prob

    def win_probability_fast(self, H1, p1, d1, H2, p2, d2, skip_first=False):
        # Create a 3D array: P_3d[x, y, sf], where sf ∈ {0,1}
        # sf=0 => we do normal attacks,
        # sf=1 => we skip the first attack (0 dmg, 100% acc), then revert to sf=0.
        P_3d = np.zeros((H1 + 1, H2 + 1, 2), dtype=np.float64)

        # Boundary conditions (vectorized):
        # - If y=0 but x>0 => P=1
        # - If x=0 but y>0 => P=0
        # - If x=0, y=0   => P=0
        P_3d[:, 0, :] = 1.0
        P_3d[0, :, :] = 0.0
        P_3d[0, 0, :] = 0.0

        # Fill in the table
        # Because any reference like x-d2 < x or y-d1 < y, we can do a simple pass from
        # x=0..H1 and y=0..H2. The required sub-states are guaranteed to be computed.
        for x in range(1, H1 + 1):
            for y in range(1, H2 + 1):
                # sf can be 0 or 1
                for sf in (0, 1):

                    # ----------------------------
                    # Compute S1 = we attack first
                    # ----------------------------
                    if sf == 1:
                        # skip-first case: 0 dmg, 100% acc
                        # Opponent hits (prob p2):
                        #   if x <= d2 => 0
                        #   else => P_3d[x-d2, y, 0]
                        # Opponent misses (prob 1-p2):
                        #   => P_3d[x, y, 0]
                        s1_val = p2 * (0.0 if x <= d2 else P_3d[x - d2, y, 0]) \
                                 + (1.0 - p2) * P_3d[x, y, 0]
                    else:
                        # normal attack
                        # hit (prob p1):
                        #   if y <= d1 => 1.0
                        #   else => enemy attacks
                        # miss (prob 1-p1):
                        #   => enemy attacks
                        if y <= d1:
                            hit_part = p1 * 1.0
                        else:
                            # If we hit, next the enemy attacks:
                            #   enemy hits (prob p2):
                            #     if x <= d2 => 0
                            #     else => P_3d[x - d2, y - d1, 0]
                            #   enemy misses (prob 1-p2):
                            #     => P_3d[x, y - d1, 0]
                            hit_part = p1 * (
                                p2 * (0.0 if x <= d2 else P_3d[x - d2, y - d1, 0])
                                + (1.0 - p2) * P_3d[x, y - d1, 0]
                            )
                        # If we miss:
                        #   enemy hits (prob p2):
                        #     if x <= d2 => 0
                        #     else => P_3d[x - d2, y, 0]
                        #   enemy misses (prob 1-p2):
                        #     => P_3d[x, y, 0]
                        miss_part = (1.0 - p1) * (
                            p2 * (0.0 if x <= d2 else P_3d[x - d2, y, 0])
                            + (1.0 - p2) * P_3d[x, y, 0]
                        )
                        s1_val = hit_part + miss_part

                    # ----------------------------
                    # Compute S2 = enemy attacks first
                    # ----------------------------
                    # enemy hits (prob p2):
                    #   if x <= d2 => 0
                    #   else => we get to attack (S1 with same sf)
                    # enemy misses (prob 1-p2):
                    #   => we get to attack (S1 with same sf)
                    hit_part_enemy = p2 * (0.0 if x <= d2 else s1_val if (x - d2 == x and y == y) else P_3d[x - d2, y, sf])
                    # Actually, note we must call "S1(x-d2, y, sf)" if x>d2. We have that in s1_val, but s1_val is for (x,y,sf).
                    # So let's just do it inline again (the same formula). But we want no function calls, so do direct:
                    # Actually, we can reuse s1_val if we shift x->(x-d2). Let's do it carefully:

                    # We'll directly compute the same "S1" logic for (x-d2,y,sf). It's simpler to do the same approach as above:
                    if x > d2:
                        if sf == 1:
                            s1_val_enemy = p2 * (0.0)  # dummy, we won't actually do that
                            # We'll inline skip-first for (x-d2,y,0)
                            skip_val = p2 * (0.0)  # not used
                            # Actually let's do it properly:
                            # The correct S1 for (x-d2, y, sf=1):
                            s1_enemy = p2 * (0.0)  # no, let's do the entire expression
                            # This is complicated. We'll do it more systematically:
                            # But we want to avoid function calls. We'll just define a small helper inline:
                            x2 = x - d2
                            if x2 <= 0:
                                s1_enemy = 0.0
                            else:
                                # skip-first for (x2,y)
                                s1_enemy = p2 * (0.0)  # This doesn't even make sense. Let's do it properly:

                                # Actually, let's just do it a second pass. This is messy. 
                                # Instead, let's fill the S2 in a separate pass to avoid big inline expansions. 
                                # But that was the original approach. 
                                # We'll do a short inline: S1(x2, y, sf=1).
                                # skip-first => 0 dmg => same logic as above:
                                s1_enemy = p2 * (0.0 if x2 <= d2 else P_3d[x2 - d2, y, 0]) + (1-p2)*P_3d[x2, y, 0]
                            s1_val_enemy = s1_enemy
                        else:
                            # normal attack for (x-d2,y, sf=0)
                            x2 = x - d2
                            if y <= d1:
                                hit_part2 = p1 * 1.0
                            else:
                                hit_part2 = p1 * (
                                    p2 * (0.0 if x2 <= d2 else P_3d[x2 - d2, y - d1, 0])
                                    + (1.0 - p2) * P_3d[x2, y - d1, 0]
                                )
                            miss_part2 = (1.0 - p1) * (
                                p2 * (0.0 if x2 <= d2 else P_3d[x2 - d2, y, 0])
                                + (1.0 - p2) * P_3d[x2, y, 0]
                            )
                            s1_val_enemy = hit_part2 + miss_part2
                    else:
                        s1_val_enemy = 0.0

                    # So the enemy's "hit" part is p2 * s1_val_enemy
                    # The enemy's "miss" part is (1-p2)*S1(x,y,sf)
                    # But we have S1(x,y,sf) in s1_val already
                    s2_val = p2 * (0.0 if x <= d2 else s1_val_enemy) + (1.0 - p2) * s1_val

                    # Finally P_3d[x, y, sf] = 0.5*S1 + 0.5*S2
                    P_3d[x, y, sf] = 0.5 * s1_val + 0.5 * s2_val

        # The final result
        val_fast = float(P_3d[H1, H2, 1 if skip_first else 0])

        # Compare with the original method for correctness
        #val_old = self.win_probability(H1, p1, d1, H2, p2, d2, skip_first)
        #if abs(val_fast - val_old) > 1e-12:
        #    print(
        #        "BIG ERROR: fast method diverges from the original!\n"
        #        f"H1={H1} p1={p1} d1={d1} H2={H2} p2={p2} d2={d2} skip_first={skip_first}\n"
        #        f"fast={val_fast}, old={val_old}"
        #    )
        #    sys.exit(1)

        return val_fast

    def duel_probability(self, H1, p1, d1, H2, p2, d2, skip_first=False):
        """
        Returns the probability that 'you' (with HP=H1, accuracy=p1, damage=d1)
        eventually kill the enemy (HP=H2, accuracy=p2, damage=d2).

        If skip_first=True, then on *your* very first attack, you deal 0 damage (only once).
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
            Recursive function returning the probability you eventually win
            starting with 'x' HP for you, 'y' HP for the enemy,
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
        #    If skip_first=True, do one "special" turn where your first attack
        #    deals 0 damage, then the enemy responds once, and only then do we
        #    proceed to the normal recursion.
        # -------------------------------------------------------------------------
        if not skip_first:
            return getP(H1, H2)
        else:
            # "First move" = you attack for 0 damage, then enemy attacks if he's alive.
            # If you're still alive, then we revert to normal recursion from new HPs.

            # 1) You do 0 damage -> enemy HP stays = H2
            # 2) Enemy attacks with probability p2
            #     - If (H1 <= d2) and he hits => you die => 0% chance
            #     - If (H1 >  d2) and he hits => new state is (H1 - d2, H2)
            #     - If he misses => new state is (H1, H2)
            prob_if_enemy_hits = getP(H1 - d2, H2) if (H1 > d2) else 0.0
            prob_if_enemy_miss = getP(H1, H2)

            # Weighted by the chance the enemy hits or misses
            return p2 * prob_if_enemy_hits + (1.0 - p2) * prob_if_enemy_miss

    ############################################################################
    # NEW BEST_ACTION (REPLACING THE OLD GEOMETRIC-MEAN LOGIC).
    # Only this method is replaced. Everything else remains the same.
    ############################################################################
    def get_action(self, g: PkmBattleEnv) -> int:
        """
        NEW best_action logic:
        
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
