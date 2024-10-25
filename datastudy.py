import os
import csv
from config import *
import pickle
import parseutils as p
from parseutils import sd
import time
import numpy as np
from parameters import *
from collections import defaultdict

with open ('./data_small.pkl', 'rb') as pkl:
    dataset = pickle.load(pkl)

peep_stats = defaultdict(int)
rr_stats = defaultdict(int)
tv_stats = defaultdict(int)
fio2_stats = defaultdict(int)

for patient_course in dataset:
    print("new patient")
    for hour_stats in patient_course:
        static_state, dynamic_state, actions, next_state = hour_stats
        set_peep, set_rr, set_tv, set_fio2, vasopressor = actions
        peep_stats[set_peep] += 1
        rr_stats[set_rr] += 1
        tv_stats[set_tv] += 1
        fio2_stats[set_fio2] += 1
        print(hour_stats)

peep_vals = sorted(peep_stats.keys())
rr_vals = sorted(rr_stats.keys())
tv_vals = sorted(tv_stats.keys())
fio2_vals = sorted(fio2_stats.keys())

# print("PEEP")
# for p in peep_vals:
#     print(p, peep_stats[p])

# print("RR")
# for r in rr_vals:
#     print(r, rr_stats[r])

# print("TV")
# for t in tv_vals:
#     print(t, tv_stats[t])

# print("FIO2")
# for f in fio2_vals:
#     print(f, fio2_stats[f])

# actionstats = {
#     "PEEP": peep_stats,
#     "RR": rr_stats,
#     "TV": tv_stats,
#     "FIO2": fio2_stats
# }

# with open('./action_stats.pkl', 'wb') as f:
#     pickle.dump(actionstats, f)