import numpy as np
import matplotlib.pyplot as plt
import time
import vgc.behaviour.BattlePolicies as pol
from vgc.datatypes.Objects import PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.balance.meta import StandardMetaData
from vgc.behaviour.TeamBuildPolicies import RandomTeamBuilder
from vgc.behaviour.TeamSelectionPolicies import RandomTeamSelectionPolicy
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator

n_battles = 100
policies1 = [pol.Minimax()]
policies = [pol.RandomPlayer(), pol.OneTurnLookahead(), pol.TypeSelector(), pol.BreadthFirstSearch(),
             pol.PrunedBFS(), pol.TunedTreeTraversal()]

policies_names = ["Random", "OneTurnAhead", "TypeSelector", "BreadthFirst", "PrunedBFS", "TunedTree"]
policies_names1 = ["Minimax"]


n = len(policies)
policies_matrix = np.zeros((1, n))
time_matrix = np.zeros((1, n))
turns_matrix = np.zeros((1, n))
std_matrix = np.zeros((1, n))

# roster generation
roster = RandomPkmRosterGenerator().gen_roster()
team_builder = RandomTeamBuilder()
team_builder.set_roster(roster)

for i, player0 in enumerate(policies1):
    for j, player1 in enumerate(policies):

        print(str (i*n + j + 1) + "/" + str(n*n))
        print(policies_names[i] + " vs " + policies_names[j])

        won_matches = 0     

        t = False
        ep = 0

        start = time.time()

        turns = np.zeros(n_battles)
        while ep < n_battles:
            # random team building 
            team0 = team_builder.get_action(StandardMetaData())
            team1 = team_builder.get_action(StandardMetaData())
            team0 = PkmTeam(pkms = team0.pkm_list[0:3])
            team1 = PkmTeam(pkms = team1.pkm_list[0:3])
            
            env = PkmBattleEnv((team0, team1), debug=False, encode=(player0.requires_encode(), player1.requires_encode()))

            s, _ = env.reset()
            env.render()
            ep += 1
            while not t:
                turns[ep-1] += 1
                if time.time() - start > 1000:
                    t = True
                    print("time problem")
                    break
                o0 = s[0]
                o1 = s[1]
                a = [player0.get_action(o0), player1.get_action(o1)]
                s, _, t, _, _ = env.step(a)
                env.render()

            t = False
            won_matches += not(env.winner)    

        player0.close()

        stop = time.time()

        if stop - start > 1000:
            policies_matrix[i, j] = np.nan
            time_matrix[i, j] = np.nan
            turns_matrix[i, j] = np.nan

        else:
            winning_rate = won_matches/n_battles
            policies_matrix[i, j] = winning_rate
            time_matrix[i, j] = stop - start
            turns_matrix[i, j] = np.mean(turns)
            std_matrix[i, j] = np.std(turns)
            

print(policies_matrix)
print(time_matrix)


plt.figure(1)
plt.imshow(policies_matrix, cmap='binary', interpolation='nearest', vmin=0, vmax=1)  
plt.colorbar(label="winning rate of player 0")  

plt.xticks(ticks=np.arange(len(policies_names)), labels=policies_names, rotation=45)  
plt.yticks(ticks=np.arange(len(policies_names1)), labels=policies_names1)  

for i in range(policies_matrix.shape[0]):  
    for j in range(policies_matrix.shape[1]): 
        plt.text(j, i, f"{policies_matrix[i, j]:.2f}", ha='center', va='center', color='red', fontsize=8)

plt.ylabel("Player 0 strategy")
plt.xlabel("Player 1 strategy")

plt.tight_layout()

plt.figure(2)
plt.imshow(time_matrix, cmap='PuRd', interpolation='nearest')  
plt.colorbar(label="mean time to perform battle")  

plt.xticks(ticks=np.arange(len(policies_names)), labels=policies_names, rotation=45)  
plt.yticks(ticks=np.arange(len(policies_names1)), labels=policies_names1)  

for i in range(time_matrix.shape[0]):  
    for j in range(time_matrix.shape[1]): 
        plt.text(j, i, f"{time_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

plt.ylabel("Player 0 strategy")
plt.xlabel("Player 1 strategy")

plt.tight_layout()

plt.figure(3)
plt.imshow(turns_matrix, cmap='PuBu', interpolation='nearest')  
plt.colorbar(label="mean turns for battle")  

plt.xticks(ticks=np.arange(len(policies_names)), labels=policies_names, rotation=45)  
plt.yticks(ticks=np.arange(len(policies_names1)), labels=policies_names1)  

for i in range(turns_matrix.shape[0]):  
    for j in range(turns_matrix.shape[1]): 
        plt.text(j, i, f"{turns_matrix[i, j]:.2f} \n std: {std_matrix[i, j]:.2f}", ha='center', va='center', color='red', fontsize=8)

plt.ylabel("Player 0 strategy")
plt.xlabel("Player 1 strategy")

plt.tight_layout()

plt.show()
        










