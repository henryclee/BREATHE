import os
import csv

datapath = './data'

def getFiles(datapath):

    files = []

    for dirpath, dirnames, filenames in os.walk(datapath):        
        for dirname in dirnames:
            getFiles(dirname)
        dirpath = dirpath.replace('\\', '/')
        for filename in filenames:
            files.append(dirpath + '/' + filename)
            # print(f"File: {dirpath}/{filename}")

    return files

filelist = getFiles(datapath)

# Headers
# 0 hr - HOUR
# 1 subject_id
# 2 hadm_id
# 3 stay_id
# 4 intime
# 5 outtime
# 6 invasive
# 7 noninvasive
# 8 highflow
# 9 discharge_outcome
# 10 icuouttime_outcome
# 11 death_outcome
# 12 elixhauser_vanwalraven
# 13 gcs
# 14 gcs_motor
# 15 gcs_verbal
# 16 gcs_eyes
# 17 gcs_unable
# 18 gender
# 19 anchor_age
# 20 anchor_year
# 21 insurance
# 22 language
# 23 marital_status
# 24 race
# 25 first_careunit
# 26 los
# 27 pinsp_draeger
# 28 pinsp_hamilton
# 29 ppeak
# 30 set_peep
# 31 total_peep
# 32 pcv_level
# 33 rr
# 34 set_rr
# 35 total_rr
# 36 set_tv
# 37 total_tv
# 38 set_fio2
# 39 set_ie_ratio
# 40 set_pc_draeger
# 41 set_pc
# 42 height_inch
# 43 pbw_kg
# 44 calculated_bicarbonate
# 45 pCO2
# 46 pH
# 47 pO2
# 48 so2
# 49 vasopressor
# 50 crrt
# 51 heart_rate
# 52 sbp
# 53 dbp
# 54 mbp
# 55 sbp_ni
# 56 dbp_ni
# 57 mbp_ni
# 58 temperature
# 59 spo2
# 60 glucose
# 61 sepsis3
# 62 sofa_24hours

#FEATURES

#Static State
# 18 gender 0 F / 1 M
# 19 anchor_age
# 24 race - maybe categorical?
# 42 height_inch
# 43 pbw_kg

#Dynamic State
# 0 hr
# 9 discharge_outcome
# 10 icuouttime_outcome
# 11 death_outcome
# 12 elixhauser_vanwalraven
# 13 gcs
# 14 gcs_motor
# 15 gcs_verbal
# 16 gcs_eyes
# 29 ppeak
# 30 set_peep
# 31 total_peep
# 32 pcv_level
# 33 rr
# 35 total_rr
# 37 total_tv
# 45 pCO2
# 46 pH
# 47 pO2
# 48 so2
# 49 vasopressor
# 50 crrt
# 51 heart_rate
# 52 sbp
# 53 dbp
# 54 mbp
# 58 temperature
# 59 spo2
# 60 glucose
# 61 sepsis3
# 62 sofa_24hours

#ACTIONS
# 6 invasive
# 7 noninvasive
# 8 highflow
# 30 set_peep
# 34 set_rr
# 36 set_tv
# 38 set_fio2
# 39 set_ie_ratio
# 41 set_pc

def getval(val):
    if val:
        return float(val)
    else:
        return 0


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
    invasive = getval(invasive)
    noninvasive = getval(noninvasive)
    highflow = getval(highflow)
    discharge_outcome = getval(discharge_outcome)
    icuouttime_outcome = getval(icuouttime_outcome)
    death_outcome = getval(death_outcome)
    elixhauser_vanwalraven = getval(elixhauser_vanwalraven)

    gcs = getval(gcs)
    gcs_motor = getval(gcs_motor)
    gcs_verbal = getval(gcs_verbal)
    gcs_eyes = getval(gcs_eyes)
    age = getval(age)
    height = getval(height)
    weight = getval(weight)

    ppeak = getval(ppeak)
    total_peep = getval(total_peep)
    rr = getval(rr)
    set_rr = getval(set_rr)
    total_rr = getval(total_rr)
    set_tv = getval(set_tv)
    total_tv = getval(total_tv)
    set_fio2 = getval(set_fio2)
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
        sbp = 0

    if dbp:
        dbp = getval(dbp)
    elif dbp_ni:
        dbp = getval(dbp_ni)
    else:
        dbp = 0

    if mbp:
        mbp = getval(mbp)
    elif mbp_ni:
        mbp = getval(mbp_ni)
    else:
        mbp = 0

    # Clean up gender
    if gender == "F":
        gender = 0
    elif gender == "M":
        gender = 1
    else:
        print("Gender value not set to M or F")

    return [ hr, invasive, noninvasive, highflow, discharge_outcome, icuouttime_outcome, death_outcome, 
            elixhauser_vanwalraven, gcs, gcs_motor, gcs_verbal, gcs_eyes, gender, age, anchor_year, race, 
            first_careunit, los, pinsp_draeger, pinsp_hamilton, ppeak, set_peep, total_peep, pcv_level, rr, set_rr,
            total_rr, set_tv, total_tv, set_fio2, set_ie_ratio, set_pc_draeger, set_pc, height, weight, 
            calculated_bicarbonate, pCO2, pH, pO2, so2, vasopressor, crrt, heart_rate, sbp, dbp, mbp, temperature,
            spo2, glucose, sepsis3, sofa ]



for fname in filelist:

    static_states = []
    dynamic_states = []
    actions = []

    f = open(fname, 'r')
    reader = csv.reader(f)
    
    # Skip the header
    next(reader)
    
    # Get static features
    # row = next(reader)
    # cleanline = cleanRow(row)

    # [ hr, invasive, noninvasive, highflow, discharge_outcome, icuouttime_outcome, death_outcome, 
    # elixhauser_vanwalraven, gcs, gcs_motor, gcs_verbal, gcs_eyes, gender, age, anchor_year, race, 
    # first_careunit, los, pinsp_draeger, pinsp_hamilton, ppeak, set_peep, total_peep, pcv_level, rr, set_rr,
    # total_rr, set_tv, total_tv, set_fio2, set_ie_ratio, set_pc_draeger, set_pc, height, weight, 
    # calculated_bicarbonate, pCO2, pH, pO2, sO2, vasopressor, crrt, heart_rate, sbp, dbp, mbp, temperature,
    # spo2, glucose, sepsis3, sofa ] = cleanline


    invasive = 0
    noninvasive = 0

    ventilated = False

    # Wait until ventilated
    while not ventilated:
        # Readline
        try:
            row = next(reader)
            cleanline = cleanRow(row)

            [ hr, invasive, noninvasive, highflow, discharge_outcome, icuouttime_outcome, death_outcome, 
            elixhauser_vanwalraven, gcs, gcs_motor, gcs_verbal, gcs_eyes, gender, age, anchor_year, race, 
            first_careunit, los, pinsp_draeger, pinsp_hamilton, ppeak, set_peep, total_peep, pcv_level, rr, set_rr,
            total_rr, set_tv, total_tv, set_fio2, set_ie_ratio, set_pc_draeger, set_pc, height, weight, 
            calculated_bicarbonate, pCO2, pH, pO2, sO2, vasopressor, crrt, heart_rate, sbp, dbp, mbp, temperature,
            spo2, glucose, sepsis3, sofa ] = cleanline
        except StopIteration:
            break
        ventilated = (invasive + noninvasive != 0)

    if ventilated:
        print(fname)
    
        static_states = [gender, age, height, weight]

        print (static_states)

        features = [heart_rate, mbp, spo2 , gcs, elixhauser_vanwalraven, ]

        print (features)

        actions = []

        print (actions)

        break


    # Go to first hour on vent
    # if invasive == 0 and noninvasive == 0:
    #     for row in reader:

    #         hr, subject_id, hadm_id, stay_id, intime, outtime, invasive, noninvasive, highflow, discharge_outcome, 
    #         icuouttime_outcome, death_outcome, elixhauser_vanwalraven, gcs, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable,
    #         gender, age, anchor_year, insurance, language, marital_status, race, first_careunit, los, pinsp_draeger,
    #         pinsp_hamilton, ppeak, set_peep, total_peep, pcv_level, rr, set_rr, total_rr, set_tv, total_tv, set_fio2,
    #         set_ie_ratio, set_pc_draeger, set_pc, height, weight, calculated_bicarbonate, pCO2, pH, pO2, so2, 
    #         vasopressor, crrt, heart_rate, sbp, dbp, mbp, sbp_ni, dbp_ni, mbp_ni, temperature, spo2, glucose, sepsis3,
    #         sofa_24hours = row
            
    #         if invasive == 1 or noninvasive == 1:
    #             break

    # starthr = hr
# 37 total_tv
# 45 pCO2
# 46 pH
# 47 pO2
# 48 so2
# 49 vasopressor
# 50 crrt
# 51 heart_rate
# 52 sbp
# 53 dbp
# 54 mbp
# 58 temperature
# 59 spo2
# 60 glucose
# 61 sepsis3
# 62 sofa_24hours]

    # Read the rest of the file
    # for row in reader:
        
    #     hr, subject_id, hadm_id, stay_id, intime, outtime, invasive, noninvasive, highflow, discharge_outcome, 
    #     icuouttime_outcome, death_outcome, elixhauser_vanwalraven, gcs, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable,
    #     gender, age, anchor_year, insurance, language, marital_status, race, first_careunit, los, pinsp_draeger,
    #     pinsp_hamilton, ppeak, set_peep, total_peep, pcv_level, rr, set_rr, total_rr, set_tv, total_tv, set_fio2,
    #     set_ie_ratio, set_pc_draeger, set_pc, height, weight, calculated_bicarbonate, pCO2, pH, pO2, so2, 
    #     vasopressor, crrt, heart_rate, sbp, dbp, mbp, sbp_ni, dbp_ni, mbp_ni, temperature, spo2, glucose, sepsis3,
    #     sofa_24hours = row


    # break
    
    # for row in reader:
    #     print(row)
