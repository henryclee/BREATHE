import os
import csv
from config import *
import pickle
import parseutils as p
import time


with open("filelist.pkl", 'rb') as f:
    filelist = pickle.load(f)

print(f"Loaded filelist, there are {len(filelist)} files")

# Stats dictionary, so we can refer to stats by name instead of index #
sd = {}
slist = ['hr', 'invasive', 'noninvasive', 'highflow', 'discharge', 'icuout', 'death', 
        'elix_van', 'gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'gender', 'age', 'anchor_year', 'race', 
        'first_careunit', 'los', 'pinsp_draeger', 'pinsp_hamilton', 'ppeak', 'set_peep', 'total_peep', 'pcv_level', 'rr', 'set_rr',
        'total_rr', 'set_tv', 'total_tv', 'set_fio2', 'set_ie_ratio', 'set_pc_draeger', 'set_pc', 'height', 'weight', 
        'calculated_bicarbonate', 'pCO2', 'pH', 'pO2', 'so2', 'vasopressor', 'crrt', 'heart_rate', 'sbp', 'dbp', 'mbp', 'temperature',
        'spo2', 'glucose', 'sepsis3', 'sofa']

for i,v in enumerate(slist):
    sd[v] = i


# Data - will eventually be the numpy array

data = []

start = time.time()

for fname in filelist:

    # Initialize everything to None:
    stats = [None for _ in range(51)]

    f = open(fname, 'r')
    reader = csv.reader(f)
    
    # Skip the header
    next(reader)
    
    ventilated = False
    # Wait until ventilated
    while not ventilated:
        # Readline
        try:
            row = next(reader)
            if len(row) != 63:
                continue
            cleanline = p.cleanRow(row)
            stats = [new if new is not None else orig for orig, new in zip(stats, cleanline)]
        except StopIteration:
            break
        ventilated = (stats[sd['invasive']]) == 1

    # If not ventilated, check next file
    if not ventilated:
        # print(fname, "not ventilated")
        continue



    # Wait until we get valid stats
    while not p.validStats(stats,sd):
        # Readline
        try:
            row = next(reader)
            if len(row) != 63:
                continue
            cleanline = p.cleanRow(row)
            stats = [new if new is not None else orig for orig, new in zip(stats, cleanline)]
        except StopIteration:
            break

    if not p.validStats(stats,sd):
        continue

    # Should have valid stats from here on out

    episode_data = []
    done = False
    while not done:
        # Readline
        try:
            row = next(reader)
            if len(row) != 63:
                continue
            cleanline = p.cleanRow(row)
            nextstats = [new if new is not None else orig for orig, new in zip(stats, cleanline)]
            # State, Action, (Reward will be calculated later), Next_State
            episode_data.append([p.getStatic(stats, sd), p.getDynamic(stats,sd), p.getAction(stats,sd), p.getDynamic(nextstats, sd)])
            stats = nextstats
            if p.terminate(stats, sd):
                done = True
        except StopIteration:
            break

    if not done:
        continue

    # Terminal states are death and extubation. If the patient is extubated, we need to determine whether the patient
    # is alive at the end of the encounter

    dead = False

    if int(stats[sd['death']]) == 0:
        while not dead:
            # Readline
            try:
                row = next(reader)
                if len(row) != 63:
                    continue
                cleanline = p.cleanRow(row)
                if cleanline[sd['death']] == 1:
                    dead = True    
            except StopIteration:
                break

    f.close()

    # If the patient dies during the rest of the encounter, we change the value in the dynamic state for the next state
    # of the final episode step
    if dead:
        episode_data[-1][-1][-1] = 1

    data.append(episode_data)

print(f"Collected data from {len(data)} patients")

with open('./data_small.pkl', 'wb') as pklfile:
    pickle.dump(data, pklfile)

end = time.time()
print(f"Time taken: {end - start:.4f} seconds")



    

        













