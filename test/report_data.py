import sys, os
# add relative parent directory to path
sys.path.append(os.path.dirname(os.getcwd()))


from vgc.behaviour.BattlePolicies import *
import utils.scraping_data as scraping_data
from policies.PrunedTreeSearch import PrunedTreeSearch
from policies.Heruristical import Heuristical

# Define the policies to compare
policies = [RandomPlayer(), OneTurnLookahead(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), PrunedTreeSearch(), Heuristical()]
policies_names = [policy.__class__.__name__ for policy in policies]
n = len(policies)

mode = "parallel"

# Running in parallel, pruned tree search must be run sequentially to avoid conflicts
if mode == "parallel":
    policies = [RandomPlayer(), OneTurnLookahead(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), PrunedTreeSearch(parallel=False), Heuristical()] # PrunedTreeSearch() must be run sequentially
    policies_names = [policy.__class__.__name__ for policy in policies]
    n = len(policies)
    n_processes = 10
    n_battles_per_process = 10
    n_battles_per_file = n_processes * n_battles_per_process
    if __name__ == '__main__': # to avoid multiprocessing error ??? I read it on stackoverflow
        for i, player2 in enumerate(policies):
            for j, player1 in enumerate(policies):
                if i < j: # only play each pair once
                    continue
                print(f"{(i)*(i+1)/2+j+1:.0f}/{n*(n+1)/2:.0f}: {policies_names[i]} vs {policies_names[j]}")

                scraping_data.run_and_update_battle_pool(player2, player1,
                    folder=f"data/parallel/{n_processes}_{n_battles_per_process}",
                    n_to_emulate_per_process=n_battles_per_process,
                    n_processes=n_processes,
                    max_battles_in_file=n_battles_per_file,
                    verbose=True)
# Running sequentially excluded PrunedTreeSearch
elif mode == "sequential":
    policies = [RandomPlayer(), OneTurnLookahead(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), Heuristical()]
    policies_names = [policy.__class__.__name__ for policy in policies]
    n = len(policies)
    n_battles = 100
    n_battles_per_file = 100
    for i, player2 in enumerate(policies):
        for j, player1 in enumerate(policies):
            if i < j: # only play each pair once
                continue
            print(f"{(i)*(i+1)/2+j+1:.0f}/{n*(n+1)/2:.0f}: {policies_names[i]} vs {policies_names[j]}")

            scraping_data.run_and_update_battle(player2, player1,
                folder=f"data/sequential/all_except_pts",
                n_to_emulate=n_battles,
                max_battles_in_file=n_battles_per_file,
                verbose=True)
# Running sequentially with PrunedTreeSearch
elif mode == "pts_sequential":
    policies = [RandomPlayer(), OneTurnLookahead(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), Heuristical()]
    policies_names = [policy.__class__.__name__ for policy in policies]
    n = len(policies)
    n_battles = 100
    n_battles_per_file = 100
    pts_depth = 3
    pts_instances = 20
    pts_player = PrunedTreeSearch(max_depth=pts_depth, instances=pts_instances, parallel=True)
    for i, player2 in enumerate(policies):
        print(f"{i+1:.0f}/{n:.0f}: PrunedTreeSearch_{pts_depth}_{pts_instances} vs {policies_names[i]}")
        scraping_data.run_and_update_battle(pts_player, player2,
                folder=f"data/sequential/pts_{pts_depth}_{pts_instances}",
                n_to_emulate=n_battles,
                max_battles_in_file=n_battles_per_file,
                verbose=True)

