import os
import csv
from config import *
import pickle

def getFiles(DATAPATH):

    files = []

    for dirpath, dirnames, filenames in os.walk(DATAPATH
):        
        for dirname in dirnames:
            getFiles(dirname)
        dirpath = dirpath.replace('\\', '/')
        for filename in filenames:
            files.append(dirpath + '/' + filename)
            # print(f"File: {dirpath}/{filename}")

    return files

filelist = getFiles(DATAPATH)

with open("filelist.pkl", "wb") as f:
    pickle.dump(filelist, f)