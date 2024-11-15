# Rewards
SURVIVE = 10
DEATH = -10

# MBP within [70,80]
MINMBP = 70
MAXMBP = 80
GOODBPREWARD = .1
BADBPREWARD = -.1

# SpO2 within [94,98]
MINSPO2 = 94
MAXSPO2 = 98
GOODO2REWARD = .1
BADO2REWARD  = -.1

# How many hours in a window
WINDOW_SIZE = 4

# Discount factor
DF = .95

# Boundaries for actions v1 - 162 states -- too big
# HIPEEP = 8
# LOPEEP = 5
# HIRR = 20
# LORR = 16
# HITV = 500
# LOTV = 400
# HIO2 = 50
# LOO2 = 40

# Boundaries for actions v2
HIPEEP = 5
HIRR = 18
HIO2 = 40
HITV = 500
LOTV = 400





# Define state, action, reward, next state
cols = ['gender', 'age', 'height', 'weight', 
        'hr1', 'mbp1', 'spo21', 'vaso1', 'gcs1', 'sofa1', 'invasive1','death1',
        'hr2', 'mbp2', 'spo22', 'vaso2', 'gcs2', 'sofa2', 'invasive2','death2',
        'hr3', 'mbp3', 'spo23', 'vaso3', 'gcs3', 'sofa3', 'invasive3','death3',
        'hr4', 'mbp4', 'spo24', 'vaso4', 'gcs4', 'sofa4', 'invasive4','death4',
        'action', 'reward',
        'hrn', 'mbpn', 'spo2n', 'vason', 'gcsn', 'sofan', 'invasiven','deathn']

# State
statecols = ['gender', 'age', 'height', 'weight', 
        'hr1', 'mbp1', 'spo21', 'vaso1', 'gcs1', 'sofa1', 'invasive1','death1',
        'hr2', 'mbp2', 'spo22', 'vaso2', 'gcs2', 'sofa2', 'invasive2','death2',
        'hr3', 'mbp3', 'spo23', 'vaso3', 'gcs3', 'sofa3', 'invasive3','death3',
        'hr4', 'mbp4', 'spo24', 'vaso4', 'gcs4', 'sofa4', 'invasive4','death4']
# Static state
sstatecols = ['gender', 'age', 'height', 'weight']
# Dynamic state
dstatecols = [
        'hr1', 'mbp1', 'spo21', 'vaso1', 'gcs1', 'sofa1', 'invasive1','death1',
        'hr2', 'mbp2', 'spo22', 'vaso2', 'gcs2', 'sofa2', 'invasive2','death2',
        'hr3', 'mbp3', 'spo23', 'vaso3', 'gcs3', 'sofa3', 'invasive3','death3',
        'hr4', 'mbp4', 'spo24', 'vaso4', 'gcs4', 'sofa4', 'invasive4','death4']
dstatecols1 = ['hr1', 'mbp1', 'spo21', 'vaso1', 'gcs1', 'sofa1', 'invasive1','death1']
dstatecols2 = ['hr2', 'mbp2', 'spo22', 'vaso2', 'gcs2', 'sofa2', 'invasive2','death2']
dstatecols3 = ['hr3', 'mbp3', 'spo23', 'vaso3', 'gcs3', 'sofa3', 'invasive3','death3']
dstatecols4 = ['hr4', 'mbp4', 'spo24', 'vaso4', 'gcs4', 'sofa4', 'invasive4','death4']
# Action
actioncols = ['action']
# Reward
rewardcols = ['reward']
# Next state
nstatecols = ['hrn', 'mbpn', 'spo2n', 'vason', 'gcsn', 'sofan', 'invasiven','deathn']
# Done
donecols = ['invasiven']

STATEIDX = [cols.index(v) for v in statecols]
SSTATEIDX = [cols.index(v) for v in sstatecols]
DSTATEIDX = [cols.index(v) for v in dstatecols]
DSTATEIDX1 = [cols.index(v) for v in dstatecols1]
DSTATEIDX2 = [cols.index(v) for v in dstatecols2]
DSTATEIDX3 = [cols.index(v) for v in dstatecols3]
DSTATEIDX4 = [cols.index(v) for v in dstatecols4]
ACTIONIDX = [cols.index(v) for v in actioncols]
REWARDIDX = [cols.index(v) for v in rewardcols]
NSTATEIDX = [cols.index(v) for v in nstatecols]
DONEIDX = [cols.index(v) for v in donecols]

ACTIONSPACE = 24