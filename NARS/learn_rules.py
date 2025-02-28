import numpy as np
import pandas

from Config import p_pool_size, n_PTRs, scope, train_epoch, print_step_results
from Utils import Pattern_pool, match_ultimate

p_pool = Pattern_pool(p_pool_size)

df = pandas.read_csv("./rule_based_predictions.csv")
for i in range(len(df)):
    if i > train_epoch:
        break
    r1 = np.array(df)[i, :]
    for j in range(scope):
        r2 = np.array(df)[i + j, :]
        if print_step_results:
            print(match_ultimate(r1, r2, p_pool, n_PTRs), "|", 1 if r1[11] == r2[11] else 0)

print("---train_finished---")

with open("./rules.txt", "w", encoding="utf-8") as f:
    if print_step_results:
        for each in p_pool.pattern_pool:
            print(each.statements, "\033[32mf=" + str(round(each.f, 3)) + "\033[0m",
                  "\033[33mc=" + str(round(each.c, 3)) + "\033[0m")
            f.write(",".join(each.statements) + " | " + str(round(each.f, 3)) + ", " + str(round(each.c, 3)) + "\n")

np.save("./p_pool", p_pool, allow_pickle=True)
