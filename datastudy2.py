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
import statistics
import matplotlib.pyplot as plt
import sys


def get_boundaries(fp):

    with open (fp, 'rb') as pkl:
        dataset = pickle.load(pkl)

    peeplist = []
    rrlist = []
    tvlist = []
    fio2list = []

    for episode in dataset:
        for step in episode:
            sstate, dstate, action, nstate = step
            peep, rr, tv, fio2, _ = action
            peeplist.append(peep)
            rrlist.append(rr)
            tvlist.append(tv)
            fio2list.append(fio2)

    peeplist = np.array(peeplist)
    rrlist = np.array(rrlist)
    tvlist = np.array(tvlist)
    fio2list = np.array(fio2list)

    with open('./boundaries2.txt', 'w') as b:
        b.write(f"Total hours: {len(peeplist)}\n")
        b.write("PEEP\n")
        p50 = np.percentile(peeplist,50)
        b.write(f"50 percentile {p50}\n")
        b.write(f"Values <= {p50} : {np.sum(peeplist <= p50)}\n")
        b.write(f"values above {np.sum(peeplist > p50)}\n")
 
        b.write("rr\n")
        p50 = np.percentile(rrlist,50)
        b.write(f"50 percentile {p50}\n")
        b.write(f"Values <= {p50} : {np.sum(rrlist <= p50)}\n")
        b.write(f"values above {np.sum(rrlist > p50)}\n")
 
        b.write("tv\n")
        p50 = np.percentile(tvlist,50)
        b.write(f"50 percentile {p50}\n")
        b.write(f"Values <= {p50} : {np.sum(tvlist <= p50)}\n")
        b.write(f"values above {np.sum(tvlist > p50)}\n")

        b.write("fio2\n")
        p50 = np.percentile(fio2list,50)
        b.write(f"50 percentile {p50}\n")
        b.write(f"Values <= {p50} : {np.sum(fio2list <= p50)}\n")
        b.write(f"Values above {np.sum(fio2list > p50)}\n")


if __name__ == "__main__":
    
    # if len(sys.argv) < 2:
    #     print("Need filepath")
    #     sys.exit(1)

    # fp = sys.argv[1]

    get_boundaries('./data/data.pkl')
    