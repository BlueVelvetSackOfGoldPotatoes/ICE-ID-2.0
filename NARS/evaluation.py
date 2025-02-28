import numpy as np

from Config import n_PTRs, print_step_results
from Utils import match_ultimate

if __name__ == "__main__":
    p_pool = np.load("./p_pool.npy", allow_pickle=True).item()
    r1 = "470497,1880,Helga Jónsdóttir,Helga,,Jónsdóttir,,1833.0,Kona,kona hans,G,4.0,4781.0,320.0,385.0,14,Eiginkona,0.31065989847715736,10.0,0.0"
    r2 = "497208,1890,Helga Jónsdóttir,Helga,,Jónsdóttir,,1833.0,Kona,húsmóðir,E,4.0,4784.0,320.0,385.0,14,Heimavinnandi,0.31065989847715736,11.0,0.0"

    r1 = np.array(r1.split(","))
    r2 = np.array(r2.split(","))

    tmp = match_ultimate(r1, r2, p_pool, n_PTRs, True)
    if print_step_results:
        print("\033[32mEvaluation | label\033[0m:", tmp, "|", 1 if r1[11] == r2[11] else 0)
    if (tmp > 0.5 and r1[11] == r2[11]) or (tmp < 0.5 and r1[11] != r2[11]):
        print("\033[32mResult\033[0m: Succeed")
    else:
        print("\033[32mResult\033[0m: Failed")
