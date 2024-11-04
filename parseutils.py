import os
import csv
from config import *
import pickle

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


def getval(val):
    if val:
        return float(val)
    else:
        return None

# Cleans up the data from a row
def cleanRow(row: list):

    [hr, subject_id, hadm_id, stay_id, intime, outtime, invasive, noninvasive, highflow, discharge_outcome, 
    icuouttime_outcome, death_outcome, elixhauser_vanwalraven, gcs, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable,
    gender, age, anchor_year, insurance, language, marital_status, race, first_careunit, los, pinsp_draeger,
    pinsp_hamilton, ppeak, set_peep, total_peep, pcv_level, rr, set_rr, total_rr, set_tv, total_tv, set_fio2,
    set_ie_ratio, set_pc_draeger, set_pc, height, weight, calculated_bicarbonate, pCO2, pH, pO2, so2, 
    vasopressor, crrt, heart_rate, sbp, dbp, mbp, sbp_ni, dbp_ni, mbp_ni, temperature, spo2, glucose, sepsis3,
    sofa] = row

    hr = getval(hr)
    invasive = int(invasive)
    noninvasive = int(noninvasive)
    highflow = int(highflow)
    discharge_outcome = int(discharge_outcome)
    icuouttime_outcome = int(icuouttime_outcome)
    death_outcome = int(death_outcome)
    elixhauser_vanwalraven = getval(elixhauser_vanwalraven)

    gcs = getval(gcs)
    gcs_motor = getval(gcs_motor)
    gcs_verbal = getval(gcs_verbal)
    gcs_eyes = getval(gcs_eyes)
    age = getval(age)
    height = getval(height)
    weight = getval(weight)

    ppeak = getval(ppeak)
    set_peep = getval(set_peep)
    total_peep = getval(total_peep)
    rr = getval(rr)
    set_rr = getval(set_rr)
    total_rr = getval(total_rr)
    set_tv = getval(set_tv)
    total_tv = getval(total_tv)
    set_fio2 = getval(set_fio2)
    set_ie_ratio = getval(set_ie_ratio)
    vasopressor = getval(vasopressor)
    crrt = getval(crrt)
    heart_rate = getval(heart_rate)
    temperature = getval(temperature)
    spo2 = getval(spo2)
    glucose = getval(glucose)
    sepsis3 = getval(sepsis3)
    sofa = getval(sofa)

    # Blood pressure - prefer invasive, o/w choose non-invasive
    if sbp:
        sbp = getval(sbp)
    elif sbp_ni:
        sbp = getval(sbp_ni)
    else:
        sbp = None

    if dbp:
        dbp = getval(dbp)
    elif dbp_ni:
        dbp = getval(dbp_ni)
    else:
        dbp = None

    if mbp:
        mbp = getval(mbp)
    elif mbp_ni:
        mbp = getval(mbp_ni)
    else:
        mbp = None

    # Clean up gender
    if gender == "F":
        gender = 0
    elif gender == "M":
        gender = 1
    else:
        gender = None

    return [hr, invasive, noninvasive, highflow, discharge_outcome, icuouttime_outcome, death_outcome, 
            elixhauser_vanwalraven, gcs, gcs_motor, gcs_verbal, gcs_eyes, gender, age, anchor_year, race, 
            first_careunit, los, pinsp_draeger, pinsp_hamilton, ppeak, set_peep, total_peep, pcv_level, rr, set_rr,
            total_rr, set_tv, total_tv, set_fio2, set_ie_ratio, set_pc_draeger, set_pc, height, weight, 
            calculated_bicarbonate, pCO2, pH, pO2, so2, vasopressor, crrt, heart_rate, sbp, dbp, mbp, temperature,
            spo2, glucose, sepsis3, sofa]

def validStats(stats):
    reqStats = ['gcs', 'gender', 'age', 'race', 'set_peep', 'set_rr', 'set_tv', 'set_fio2', 'height', 'weight', 
            'vasopressor', 'heart_rate', 'mbp', 'spo2', 'sofa']
    # valid = True
    for s in reqStats:
        i = sd[s]
        if stats[i] == None:
            # print("missing", s)
            # valid = False
            return False
    return True

def getStatic(stats):
    static_state = ['gender', 'age', 'height', 'weight', 'race']
    state = []
    for s in static_state:
        i = sd[s]
        state.append(stats[i])
    return state

def getDynamic(stats):
    # dynamic_state = ['heart_rate', 'mbp', 'spo2', 'gcs', 'sofa', 'invasive','death']
    dynamic_state = ['heart_rate', 'mbp', 'spo2', 'vasopressor', 'gcs', 'sofa', 'invasive','death']
    state = []
    for s in dynamic_state:
        i = sd[s]
        state.append(stats[i])
    return state

def getAction(stats):
    # actionlist = ['set_peep', 'set_rr', 'set_tv', 'set_fio2', 'vasopressor']
    actionlist = ['set_peep', 'set_rr', 'set_tv', 'set_fio2']
    action = []
    for a in actionlist:
        i = sd[a]
        action.append(stats[i])    
    return action

def terminate(stats):
    terminal1 = ['death']
    terminal0 = ['invasive']
    for s in terminal1:
        i = sd[s]
        if stats[i] == 1:
            return True
    for s in terminal0:
        i = sd[s]
        if stats[i] == 0:
            return True
    return False
