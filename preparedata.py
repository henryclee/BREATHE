import os
import csv
from config import *
import pickle
import parseutils as p
from parseutils import sd
import time
import numpy as np
from parameters import *

with open ('./data_small.pkl', 'rb') as pkl:
    dataset = pickle.load(pkl)

