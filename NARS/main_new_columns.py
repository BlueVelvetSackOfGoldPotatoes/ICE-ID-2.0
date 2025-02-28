import random

import numpy as np
import pandas

from Config import p_pool_size, n_PTRs
from Utils import Pattern_pool, match_new

p_pool = Pattern_pool(p_pool_size)
df = pandas.read_csv("./random_sample.csv")
for _ in range(10000):
    A, B = random.sample(range(len(df)), 2)
    r1 = np.array(df)[A, :]
    r2 = np.array(df)[B, :]
    print(match_new(r1, r2, p_pool, n_PTRs), "|", 1 if r1[7] == r2[7] else 0)

for each in p_pool.pattern_pool:
    print(each.statements, "\033[32mf=" + str(round(each.f, 3)) + "\033[0m",
          "\033[33mc=" + str(round(each.c, 3)) + "\033[0m")
