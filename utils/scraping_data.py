import json
import signal
import time
import sys
import os
from utils import OwnRandomTeamGenerator, MyPkmEnv


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

# create a function that runs a battle between two policies and returns the battle data
def run_battle(policy1, policy2, turns_limit=100, time_limit=1000, verbose=False):
    # create the battle data
    battle_data = {
        "result": 2,
        "turns": 0,
        "total_time": 0.0,
        "turns_time_p1": [],
        "turns_time_p2": []
    }
    
    random_team_generator = OwnRandomTeamGenerator()
    team1 = random_team_generator.get_team().get_battle_team([0, 1, 2])
    team2 = random_team_generator.get_team().get_battle_team([0, 1, 2])
    agent1, agent2 = policy1, policy2

    env = MyPkmEnv((team1, team2),
                    encode=(agent1.requires_encode(), agent2.requires_encode()),
                    debug=False)  # set new environment with teams
    
    battle_start_time = time.time()
    terminated = False
    while not terminated:
        if time.time() - battle_start_time > time_limit:
            break
        if battle_data["turns"] >= turns_limit:
            break

        choice_time_1_start = time.time()
        action1 = agent1.get_action(env)
        choice_time_1 = time.time() - choice_time_1_start
        battle_data["turns_time_p1"].append(choice_time_1)
        if verbose: print(f"Turn {battle_data['turns']+1} - Player 1 took choice in {choice_time_1:.4f}s")

        choice_time_2_start = time.time()
        action2 = agent2.get_action(env)
        choice_time_2 = time.time() - choice_time_2_start
        battle_data["turns_time_p2"].append(choice_time_2)
        if verbose: print(f"Turn {battle_data['turns']+1} - Player 2 took choice in {choice_time_2:.4f}s")
        
        actions = [action1, action2]
        _, _, terminated, _, _ = env.step(actions)
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
    
    return battle_data

# function that, given two policies (using the name of the policy class), creates a json file with the battle data if it doesn't exist
# else it reads the file and returns the data
def get_battle_data(policy1, policy2):
    policy1_name = policy1.__class__.__name__
    policy2_name = policy2.__class__.__name__

    if not os.path.exists("data"):
        os.mkdir("data")
    
    file_path = f"data/{policy1_name}_vs_{policy2_name}.json"

    if not os.path.exists(file_path):
        # create file


        with open(file_path, "w") as file:
            data = {
                "policies_names": [policy1_name, policy2_name],
                "n_battles_emulated": 0,
                "battles": []
            }
            json.dump(data, file)
        return data
    else:
        with open(file_path, "r") as file:
            return json.load(file)

# function that, given two policies (using the name of the policy class), runs battles and updates the json file with the battle data
def run_and_update_battle(policy1, policy2, n_to_emulate=100, max_battles=100, turns_limit=100, time_limit=1000, verbose=False):
    '''Run n_to_emulate battles between policy1 and policy2 and update the json file with the battle data
    
    Args:
        policy1 (Policy): the first policy to battle
        policy2 (Policy): the second policy to battle
        n_to_emulate (int): the number of battles to emulate
        max_battles (int): the maximum number of battles to store in the json file. If the file already contains max_battles, the function does nothing
        turns_limit (int): the maximum number of turns for a battle
        time_limit (int): the maximum time for a battle
        verbose (bool): whether to print information about the battle
    
    
    
    '''
    policy1_name = policy1.__class__.__name__
    policy2_name = policy2.__class__.__name__
    file_path = f"data/{policy1_name}_vs_{policy2_name}.json"

    # check if the file already exists with the opposite order of policies to avoid duplicates
    alternative_file_path = f"data/{policy2_name}_vs_{policy1_name}.json"
    if os.path.exists(alternative_file_path):
        file_path = alternative_file_path
        policy1, policy2 = policy2, policy1
        policy1_name, policy2_name = policy2_name, policy1_name

    data = get_battle_data(policy1, policy2)

    n_emulated = 0

    while n_emulated < n_to_emulate and data["n_battles_emulated"] < max_battles:
        if verbose: print(f"Emulating battle {n_emulated + 1}/{n_to_emulate} (total {data['n_battles_emulated']+1}/{max_battles}) between {policy1_name} and {policy2_name}")
        battle_data = run_battle(policy1, policy2, turns_limit, time_limit, verbose)
        data["battles"].append(battle_data)
        data["n_battles_emulated"] += 1
        n_emulated += 1

        # make sure that data is safe in case of keyboard interrupt
        with open(file_path, "w") as file:
            json.dump(data, file)

