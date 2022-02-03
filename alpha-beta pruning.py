import math
import random as r

std_id = input("Enter student ID : \n")
total_turns = int(std_id[0]) * 2

ini_HP = int(std_id[-2:][::-1])

minRange, maxRange = [int(char) for char in input().split()]
# print(minRange, maxRange)

total_branch = int(std_id[2])

leaves = []
for i in range(0, total_turns ** total_branch + 1):
    leaves.append(r.randrange(minRange, maxRange + 1))



# print(leaves)
print("1. Depth and Branch ratio is", total_turns, ":", total_branch)
print("2. Terminal States(Leaf Nodes) are ", *leaves)


cnt = 0

'''leaves = [ 19,22,9,2,26,16,16,27,16]
print(leaves)'''


def alpha_beta_pruning(depth, pos, mxPlayer, leaf, alp, beta):
    global cnt
    global total_turns
    global total_branch

    l = len(leaves)


    if depth == total_turns:
        return leaf[pos]

    if mxPlayer:

        bestVal = -math.inf

        for j in range(0, total_branch):

            val = alpha_beta_pruning(depth + 1, pos * total_branch + j, False, leaf, alp, beta)
            bestVal = max(bestVal, val)
            alp = max(alp, bestVal)

            if beta <= alp:
                corner_case = int((l - 1) / (total_turns - depth))
                cur_pos = pos * total_branch + j
                if corner_case - 1 != cur_pos:
                    cnt += (total_branch - (cur_pos % total_branch))-1
                break

        return bestVal

    else:
        bestVal = math.inf

        for j in range(0, total_branch):

            val = alpha_beta_pruning(depth + 1, pos * total_branch + j, True, leaf, alp, beta)
            bestVal = min(bestVal, val)
            beta = min(beta, bestVal)


            if beta <= alp:
                corner_case = int((l - 1) / (total_turns - depth))
                cur_pos = pos * total_branch + j
                if corner_case - 1 != cur_pos:
                    cnt += (total_branch - (cur_pos % total_branch)) - 1
                break
        return bestVal


algo = alpha_beta_pruning(0, 0, True, leaves, -math.inf, math.inf)
print("3. Left life(HP) of the defender after maximum damage caused by the attacker is", ini_HP - algo)
print("4. After Alpha-Beta Pruning Leaf Node Comparisons ", (len(leaves)-cnt))
