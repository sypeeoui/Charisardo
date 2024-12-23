import sys, os
from multiprocessing import freeze_support
# add relative parent directory to path
sys.path.append(os.path.dirname(os.getcwd()))


from vgc.behaviour.BattlePolicies import *
import utils.scraping_data as scraping_data
from policies.PrunedTreeSearch import PrunedTreeSearch
from policies.Heuristical import Heuristical
from policies.WeightedGreedy import WeightedGreedy, WeightedGreedy2

# Define the policies to compare
policies = [RandomPlayer(), OneTurnLookahead(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), PrunedTreeSearch(), Heuristical()]
policies_names = [policy.__class__.__name__ for policy in policies]
n = len(policies)

mode = "wg_parallel"

# Running in parallel, pruned tree search must be run sequentially to avoid conflicts
if mode == "parallel":
    policies = [Heuristical(), RandomPlayer(), OneTurnLookahead(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), WeightedGreedy()] # PrunedTreeSearch() must be run sequentially
    policies_names = [policy.__class__.__name__ for policy in policies]
    n = len(policies)
    n_processes = 20
    n_battles_per_process = 20
    n_battles_per_file = n_processes * n_battles_per_process
    if __name__ == '__main__': # to avoid multiprocessing error ??? I read it on stackoverflow
        for i, player2 in enumerate(policies):
            for j, player1 in enumerate(policies):
                if i < j: # only play each pair once
                    continue
                # if i != j:
                #     continue
                print(f"{(i)*(i+1)/2+j+1:.0f}/{n*(n+1)/2:.0f}: {policies_names[i]} vs {policies_names[j]}")

                scraping_data.run_and_update_battle_pool(player2, player1,
                    folder=f"data/parallel/{n_processes}_{n_battles_per_process}",
                    n_to_emulate_per_process=n_battles_per_process,
                    n_processes=n_processes,
                    max_battles_in_file=n_battles_per_file,
                    verbose=True)

# Running sequentially excluding PrunedTreeSearch
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
    policies = [OneTurnLookahead(), RandomPlayer(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), Heuristical()]
    policies_names = [policy.__class__.__name__ for policy in policies]
    n = len(policies)
    n_battles = 100
    n_battles_per_file = 100
    is_parallel = True
    pts_depth = 2
    pts_instances = 14
    pts_player = PrunedTreeSearch(max_depth=pts_depth, instances=pts_instances, parallel=is_parallel)
    if __name__ == '__main__':
        for i, player2 in enumerate(policies):
            sub_string = "parallel" if is_parallel else "sequential"
            sub_string += f"_{pts_depth}_{pts_instances}" if is_parallel else f"_{pts_depth}"
            print(f"{i+1:.0f}/{n:.0f}: PrunedTreeSearch_{sub_string} vs {policies_names[i]}")

            scraping_data.run_and_update_battle(pts_player, player2,
                    folder=f"data/sequential/pts_{sub_string}",
                    n_to_emulate=n_battles,
                    max_battles_in_file=n_battles_per_file,
                    verbose=True)


elif mode == "wg_sequential":
    policies = [RandomPlayer(), OneTurnLookahead(), TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), Heuristical(), WeightedGreedy(), WeightedGreedy2()]
    policies_names = [policy.__class__.__name__ for policy in policies]
    n = len(policies)
    n_battles = 100
    n_battles_per_file = 100
    wg_player = WeightedGreedy2()
    for i, player2 in enumerate(policies):
        print(f"{i+1:.0f}/{n:.0f}: WeightedGreedy vs {policies_names[i]}")

        scraping_data.run_and_update_battle(wg_player, player2,
                folder=f"data/sequential/wg",
                n_to_emulate=n_battles,
                max_battles_in_file=n_battles_per_file,
                verbose=True)
        
elif mode == "wg_parallel":
    policies = [OneTurnLookahead(), RandomPlayer() , TypeSelector(), BreadthFirstSearch(),
             PrunedBFS(), TunedTreeTraversal(), Heuristical(), WeightedGreedy(), WeightedGreedy2()]
    policies_names = [policy.__class__.__name__ for policy in policies]
    n = len(policies)
    n_processes = 20
    n_battles_per_process = 20
    n_battles_per_file = n_processes * n_battles_per_process
    wg_player = WeightedGreedy2()
    if __name__ == '__main__':
        for i, player2 in enumerate(policies):
            print(f"{i+1:.0f}/{n:.0f}: WeightedGreedy vs {policies_names[i]}")

            scraping_data.run_and_update_battle_pool(wg_player, player2,
                folder=f"data/parallel/wg",
                n_to_emulate_per_process=n_battles_per_process,
                n_processes=n_processes,
                max_battles_in_file=n_battles_per_file,
                verbose=True)
            