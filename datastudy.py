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

    with open('./boundaries.txt', 'w') as b:
        b.write(f"Total hours: {len(peeplist)}\n")
        b.write("PEEP\n")
        p33 = np.percentile(peeplist,33)
        b.write(f"33 percentile {p33}\n")
        b.write(f"Values below {np.sum(peeplist < p33)}\n")
        p67 = np.percentile(peeplist,67)
        b.write(f"67 percentile {p67}\n")
        b.write(f"values above {np.sum(peeplist > p67)}\n")
        b.write(f"values between: {np.sum((p33 <= peeplist) & (peeplist <= p67))}\n")

        b.write("rr\n")
        p33 = np.percentile(rrlist,33)
        b.write(f"33 percentile {p33}\n")
        b.write(f"Values below {np.sum(rrlist < p33)}\n")
        p67 = np.percentile(rrlist,67)
        b.write(f"67 percentile {p67}\n")
        b.write(f"Values above {np.sum(rrlist > p67)}\n")
        b.write(f"Values between: {np.sum((p33 <= rrlist) & (rrlist <= p67))}\n")

        b.write("tv\n")
        p33 = np.percentile(tvlist,33)
        b.write(f"33 percentile {p33}\n")
        b.write(f"Values below {np.sum(tvlist < p33)}\n")
        p67 = np.percentile(tvlist,67)
        b.write(f"67 percentile {p67}\n")
        b.write(f"Values above {np.sum(tvlist > p67)}\n")
        b.write(f"Values between: {np.sum((p33 <= tvlist) & (tvlist <= p67))}\n")

        b.write("fio2\n")
        p33 = np.percentile(fio2list,33)
        b.write(f"33 percentile {p33}\n")
        b.write(f"Values below {np.sum(fio2list < p33)}\n")
        p67 = np.percentile(fio2list,67)
        b.write(f"67 percentile {p67}\n")
        b.write(f"Values above {np.sum(fio2list > p67)}\n")
        b.write(f"Values between: {np.sum((p33 <= fio2list) & (fio2list <= p67))}\n")


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Need filepath")
        sys.exit(1)

    fp = sys.argv[1]

    get_boundaries(fp)
    