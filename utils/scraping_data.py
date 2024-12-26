import json
import signal
import time
import sys
import os
from utils import OwnRandomTeamGenerator, MyPkmEnv, OwnRandomTeamGenerator2
import multiprocessing
import multiprocessing.pool

from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator

TeamGenerator = OwnRandomTeamGenerator
# JSON FORMAT FOR DATA OF POLICY vs POLICY
# {
#     "policies_names": ["policy1_name", "policy2_name"],
#     "n_battles_emulated": 1,
#     "battles": [{
#         "result": 1, // 0 if policy1 wins, 1 if policy2 wins, 2 if took too long
#         "turns": 10,
#         "total_time": 0.9,
#         "turns_time_p1": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
#         "turns_time_p2": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
#     }]
# }
def game_state_eval(g: MyPkmEnv):
    """
    Evaluates the current game state of a Pokémon battle.
    This function calculates the health points (HP) of both the player's team and the opponent's team,
    and returns a score based on the ratio of current HP to maximum HP for both teams. The score is 
    weighted by the `self.weights` attribute.
    Args:
        g (PkmBattleEnv): The current game environment, which includes the teams and their statuses.
    Returns:
        float: A score representing the evaluation of the game state. Returns -1 if the player's team 
                has no remaining HP, 1 if the opponent's team has no remaining HP, or a weighted score 
                based on the HP ratios otherwise.
    """

    my_team = g.teams[0]
    opp_team = g.teams[1]
    my_sum_hp = my_team.active.hp + sum([p.hp for p in my_team.party])
    opp_sum_hp = opp_team.active.hp + sum([p.hp for p in opp_team.party])
    my_sum_max_hp = my_team.active.max_hp + sum([p.max_hp for p in my_team.party])
    opp_sum_max_hp = opp_team.active.max_hp + sum([p.max_hp for p in opp_team.party])
    
    if my_sum_hp == 0.0:
        return float('-1')
    elif opp_sum_hp == 0.0:
        return float('1')
    
    # print(f"HP: {my_sum_hp}, {opp_sum_hp}")
    # print(f"Max HP: {my_sum_max_hp}, {opp_sum_max_hp}")
    return my_sum_hp / my_sum_max_hp - opp_sum_hp / opp_sum_max_hp

# create a function that runs a battle between two policies and returns the battle data
def run_battle(team1, team2, policy1, policy2, turns_limit=100, time_limit=1000, verbose=False):
    # create the battle data
    battle_data = {
        "result": 2,
        "turns": 0,
        "total_time": 0.0,
        "turns_time_p1": [],
        "turns_time_p2": [],
        "log": [],
        "eval": [],
        "team1": [],
        "team2": [],
    }

    for pkm in [team1.active, team1.party[0], team1.party[1]]:
        pkm_dict = {}
        pkm_dict["max_hp"] = pkm.max_hp
        pkm_dict["moves"] = []
        for move in pkm.moves:
            move_dict = {}
            move_dict["power"] = move.power
            move_dict["accuracy"] = move.acc
            move_dict["type"] = move.type
            move_dict["max_pp"] = move.max_pp
            pkm_dict["moves"].append(move_dict)
        battle_data["team1"].append(pkm_dict)
    for pkm in [team2.active, team2.party[0], team2.party[1]]:
        pkm_dict = {}
        pkm_dict["max_hp"] = pkm.max_hp
        pkm_dict["moves"] = []
        for move in pkm.moves:
            move_dict = {}
            move_dict["power"] = move.power
            move_dict["accuracy"] = move.acc
            move_dict["type"] = move.type
            move_dict["max_pp"] = move.max_pp
            pkm_dict["moves"].append(move_dict)
        battle_data["team2"].append(pkm_dict)    
    
    agent1, agent2 = policy1, policy2
    env = MyPkmEnv((team1, team2),
                    encode=(agent1.requires_encode(), agent2.requires_encode()),
                    debug=True)  # set new environment with teams
    s, _ = env.reset()

    battle_start_time = time.time()
    terminated = False
    while not terminated:
        if time.time() - battle_start_time > time_limit:
            break
        if battle_data["turns"] >= turns_limit:
            break

        choice_time_1_start = time.time()
        action1 = agent1.get_action(s[0])
        choice_time_1 = time.time() - choice_time_1_start
        battle_data["turns_time_p1"].append(choice_time_1)
        if verbose: print(f"Turn {battle_data['turns']+1} - Player 1 took choice in {choice_time_1:.4f}s")

        choice_time_2_start = time.time()
        action2 = agent2.get_action(s[1])
        choice_time_2 = time.time() - choice_time_2_start
        battle_data["turns_time_p2"].append(choice_time_2)
        if verbose: print(f"Turn {battle_data['turns']+1} - Player 2 took choice in {choice_time_2:.4f}s")
        
        actions = [action1, action2]
        s, _, terminated, _, _ = env.step(actions)
        # print(env.log)
        # print(f"Current eval: {game_state_eval(env)}")
        print("actions: ", actions)
        battle_data["log"].append(env.log)
        battle_data["eval"].append(game_state_eval(env))
        battle_data["turns"] += 1

    battle_data["total_time"] = time.time() - battle_start_time
    if env.winner == 0:
        battle_data["result"] = 0
        if verbose: print("Player 1 wins!")
    elif env.winner == 1:
        battle_data["result"] = 1
        if verbose: print("Player 2 wins!")
    else:
        battle_data["result"] = 2
        if verbose: print("Battle took too long!")

    battle_data["commands"] = env.commands
    
    return battle_data

# function that, given two policies (using the name of the policy class), creates a json file with the battle data if it doesn't exist
# else it reads the file and returns the data
def get_battle_data(policy1, policy2, folder="data"):
    policy1_name = policy1.__class__.__name__
    policy2_name = policy2.__class__.__name__

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = f"{folder}/{policy1_name}_vs_{policy2_name}.json"

    if not os.path.exists(file_path):
        # create file
        with open(file_path, "w") as file:
            data = {
                "policies_names": [policy1_name, policy2_name],
                "n_battles_emulated": 0,
                "battles": []
            }
            json.dump(data, file, indent=4)
        return data
    else:
        with open(file_path, "r") as file:
            return json.load(file)

# function that, given two policies (using the name of the policy class), runs battles and updates the json file with the battle data
def run_and_update_battle(policy1, policy2, folder="data", n_to_emulate=100, max_battles_in_file=100, turns_limit=100, time_limit=1000, verbose=False):
    '''Run n_to_emulate battles between policy1 and policy2 and update the json file with the battle data
    
    Args:
        policy1 (Policy): the first policy to battle
        policy2 (Policy): the second policy to battle
        n_to_emulate (int): the number of battles to emulate
        max_battles (int): the maximum number of battles to store in the json file. If the file already contains max_battles, the function does nothing
        turns_limit (int): the maximum number of turns for a battle
        time_limit (int): the maximum time for a battle
        verbose (bool): whether to print information about the battle. Propagates to run_battle
    '''
    policy1_name = policy1.__class__.__name__
    policy2_name = policy2.__class__.__name__
    file_path = f"{folder}/{policy1_name}_vs_{policy2_name}.json"

    # check if the file already exists with the opposite order of policies to avoid duplicates
    alternative_file_path = f"{folder}/{policy2_name}_vs_{policy1_name}.json"
    if os.path.exists(alternative_file_path):
        file_path = alternative_file_path
        policy1, policy2 = policy2, policy1
        policy1_name, policy2_name = policy2_name, policy1_name

    data = get_battle_data(policy1, policy2, folder=folder)

    n_emulated = 0

    while n_emulated < n_to_emulate and data["n_battles_emulated"] < max_battles_in_file:
        if verbose: print(f"Emulating battle {n_emulated + 1}/{n_to_emulate} (total {data['n_battles_emulated']+1}/{max_battles_in_file}) between {policy1_name} and {policy2_name}")
        random_team_generator = TeamGenerator()
        team1 = random_team_generator.get_team().get_battle_team([0, 1, 2])
        team2 = random_team_generator.get_team().get_battle_team([0, 1, 2])

        battle_data = run_battle(team1, team2, policy1, policy2, turns_limit, time_limit, verbose)

        data["battles"].append(battle_data)
        data["n_battles_emulated"] += 1
        n_emulated += 1

        battle_data = run_battle(team2, team1, policy1, policy2, turns_limit, time_limit, verbose)

        data["battles"].append(battle_data)
        data["n_battles_emulated"] += 1
        n_emulated += 1

        # make sure that data is safe in case of keyboard interrupt
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)


def run_and_update_battle_parallel_single_instance(task_number, policy1, policy2, lock, folder="data", n_to_emulate=100, max_battles_in_file=100, turns_limit=100, time_limit=1000, verbose=False):
    '''Run n_to_emulate battles between policy1 and policy2 and update the json file with the battle data
    
    Args:
        policy1 (Policy): the first policy to battle
        policy2 (Policy): the second policy to battle
        n_to_emulate (int): the number of battles to emulate
        max_battles (int): the maximum number of battles to store in the json file. If the file already contains max_battles, the function does nothing
        turns_limit (int): the maximum number of turns for a battle
        time_limit (int): the maximum time for a battle
        verbose (bool): whether to print information about the battle. Doesn't propagate to run_battle to avoid confusion between processes
    '''
    if verbose: print(f"Task {task_number:03}: Starting task")
    policy1_name = policy1.__class__.__name__
    policy2_name = policy2.__class__.__name__
    file_path = f"{folder}/{policy1_name}_vs_{policy2_name}.json"

    # check if the file already exists with the opposite order of policies to avoid duplicates
    with lock:
        alternative_file_path = f"{folder}/{policy2_name}_vs_{policy1_name}.json"
        if os.path.exists(alternative_file_path):
            file_path = alternative_file_path
            policy1, policy2 = policy2, policy1
            policy1_name, policy2_name = policy2_name, policy1_name

    n_emulated = 0

    # check if the file already contains the required number of battles to avoid unnecessary computation
    with lock:
        data = get_battle_data(policy1, policy2, folder=folder)
        if data["n_battles_emulated"] >= max_battles_in_file:
            if verbose: print(f"Task {task_number:03}: Maximum number of battles ({data['n_battles_emulated']}) already reached")
            return

    while n_emulated < n_to_emulate:
        if verbose: print(f"Task {task_number:03}: Emulating battle {n_emulated + 1}/{n_to_emulate} between {policy1_name} and {policy2_name}")
        random_team_generator = TeamGenerator()
        team1 = random_team_generator.get_team().get_battle_team([0, 1, 2])
        team2 = random_team_generator.get_team().get_battle_team([0, 1, 2])

        battle_data = run_battle(team1, team2, policy1, policy2, turns_limit, time_limit, verbose)

        battle_data2 = run_battle(team2, team1, policy1, policy2, turns_limit, time_limit, verbose)

        with lock:
            data = get_battle_data(policy1, policy2, folder=folder)
            data["battles"].append(battle_data)
            data["battles"].append(battle_data2)

            data["n_battles_emulated"] += 2
            n_emulated += 2
            if data["n_battles_emulated"] > max_battles_in_file:
                if verbose: print(f"Task {task_number:03}: Maximum number of battles ({max_battles_in_file}) reached")
                break

            # make sure that data is safe in case of keyboard interrupt
            with open(file_path, "w") as file:
                if verbose: print(f"Task {task_number:03}: Writing to file {file_path}")
                json.dump(data, file, indent=4)
            
            if verbose: print(f"Task {task_number:03}: Added battle {n_emulated}/{n_to_emulate} (total {data['n_battles_emulated']}) to the json file")

            if n_emulated == n_to_emulate or data["n_battles_emulated"] == max_battles_in_file:
                break

def run_and_update_battle_pool(policy1, policy2, folder="data", n_to_emulate_per_process=10, n_processes=10, max_battles_in_file=100, turns_limit=100, time_limit=1000, verbose=False):
    '''Run n_to_emulate battles between policy1 and policy2 and update the json file with the battle data
    
    Args:
        policy1 (Policy): the first policy to battle
        policy2 (Policy): the second policy to battle
        n_to_emulate_per_process (int): the number of battles to emulate per process
        n_processes (int): the number of processes to use
        max_battles (int): the maximum number of battles to store in the json file. If the file already contains max_battles, the function does nothing
        turns_limit (int): the maximum number of turns for a battle
        time_limit (int): the maximum time for a battle
        verbose (bool): whether to print information about the battle
    '''
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    tasks = [(  i,
                policy1,
                policy2,
                lock,
                folder,
                n_to_emulate_per_process,
                max_battles_in_file,
                turns_limit,
                time_limit,
                verbose) for i in range(n_processes)]
    with multiprocessing.Pool(n_processes) as pool:
        pool.starmap_async(run_and_update_battle_parallel_single_instance, tasks).get(999999)

# UTILITIES TO GET STATISTICS FROM BATTLE DATA
def get_avg_turn_time_per_turn(battle_data, max_turn=None, threshold=0):
    '''Get the average turn time per turn from battle data
    
    Args:
        battle_data (dict): the battle data
        max_turn (int): the maximum number of turns to consider
        threshold (int): the minimum number of battles that had to reach a turn to consider it
    
    Returns:
        list: the average turn time per turn
    '''
    avg_turn_time_per_turn = [[],[]]
    count = []
    for battle in battle_data["battles"]:
        for i, (time1, time2) in enumerate(zip(battle["turns_time_p1"], battle["turns_time_p2"])):
            if max_turn is not None and i >= max_turn:
                break
            if i >= len(avg_turn_time_per_turn[0]):
                avg_turn_time_per_turn[0].append(0)
                avg_turn_time_per_turn[1].append(0)
                count.append(0)
            avg_turn_time_per_turn[0][i] += time1
            avg_turn_time_per_turn[1][i] += time2
            count[i] += 1
    for i in range(len(avg_turn_time_per_turn[0])):
        avg_turn_time_per_turn[0][i] /= count[i]
        avg_turn_time_per_turn[1][i] /= count[i]
    # remove turns after first outlier
    for i in range(len(avg_turn_time_per_turn[0])):
        if count[i] < threshold:
            avg_turn_time_per_turn[0] = avg_turn_time_per_turn[0][:i]
            avg_turn_time_per_turn[1] = avg_turn_time_per_turn[1][:i]
    return avg_turn_time_per_turn

def get_games_per_turn_count(battle_data):
    '''Get the number of games that reached each turn from battle data
    
    Args:
        battle_data (dict): the battle data
    
    Returns:
        list: the number of games that reached each turn
    '''
    games_per_turn_count = []
    for battle in battle_data["battles"]:
        if len(games_per_turn_count) < battle["turns"]:
            games_per_turn_count += [0] * (battle["turns"] - len(games_per_turn_count))
        games_per_turn_count[battle["turns"]-1] += 1
    return games_per_turn_count

def get_all_game_state_evals(battle_data):
    '''Get all game state evaluations from battle data as list of lists

    Args:
        battle_data (dict): the battle data

    Returns:
        list: the game state evaluations
    '''
    all_game_state_evals = []
    for battle in battle_data["battles"]:
        all_game_state_evals.append(battle["eval"])
    return all_game_state_evals