##########################################################################
## Simulator.py  v 0.1
##
## Implements two versions of a multi-level sampler:
##
## 1) Traditional 3 step process
## 2) Streaming process using hashing
##
##
## Original Code written by H. Andrew Schwartz
## for SBU's Big Data Analytics Course 
## Spring 2020
##
## Student Name: Surabhi Sarnot
## Student ID: 112584690

import random
from datetime import datetime

import mmh3
##Data Science Imports:
import numpy as np


##IO, Process Imports:


##########################################################################
##########################################################################
# Task 1.A Typical non-streaming multi-level sampler

def typicalSampler(filename, percent=.01, sample_col=0):
    # Implements the standard non-streaming sampling method
    # Step 1: read file to pull out unique user_ids from file
    # Step 2: subset to random  1% of user_ids
    # Step 3: read file again to pull out records from the 1% user_id and compute mean withdrawn

    unique_user_ids = set()
    for line in filename:
        unique_user_ids.add(line.split(",")[2])
    unique_user_ids = list(unique_user_ids)
    random.shuffle(unique_user_ids)
    sample_users = unique_user_ids[0:int(len(unique_user_ids) * percent)]

    summation = 0
    square_sum = 0
    count = 0

    filename.seek(0)

    for line in filename:
        arr = line.split(",")
        if arr[2] in sample_users:
            amount = float(arr[3])
            summation += amount
            count+=1

    print("Time in milliseconds = ", (datetime.now()-start_time).total_seconds()*1000)
    return np.mean(summation,count), np.std(summation)

##########################################################################
##########################################################################
# Task 1.B Streaming multi-level sampler

def streamSampler(stream, percent=.01, sample_col=0):
    # Implements the standard streaming sampling method:
    #   stream -- iosteam object (i.e. an open file for reading)
    #   percent -- percent of sample to keep
    #   sample_col -- column number to sample over
    #
    # Rules:
    #   1) No saving rows, or user_ids outside the scope of the while loop.
    #   2) No other loops besides the while listed. 

    # mean, standard_deviation = 0.0, 0.0
    ##<<COMPLETE>>

    buckets = 1 / percent
    cnt = 0
    mean = 0
    standard_deviation = 0

    hashValue = random.randint(0, buckets - 1)
    for line in stream:
        arr = line.split(",")
        if mmh3.hash(arr[2]) % buckets == hashValue:
            cnt += 1
            amount = float(arr[3])
            newMean = mean + ((amount - mean) / cnt)
            newStd = (amount - newMean) * (amount - mean) + standard_deviation
            mean = newMean
            standard_deviation = newStd

    return mean, np.sqrt(standard_deviation / cnt)


##########################################################################
##########################################################################
# Task 1.C Timing

files = ['transactions_small.csv', 'transactions_medium.csv', 'transactions_large.csv']
percents = [.02, .005]

if __name__ == "__main__":
    ##<<COMPLETE: EDIT AND ADD TO IT>>
    for perc in percents:
        print("\nPercentage: %.4f\n==================" % perc)
        for f in files:
            print("\nFile: ", f)
            fstream = open(f, "r")
            start_time = datetime.now()
            print("  Typical Sampler: ", typicalSampler(fstream, perc, 2))
            print("Time in milliseconds = ", (datetime.now() - start_time).total_seconds() * 1000)
            fstream.close()
            fstream = open(f, "r")
            start_time = datetime.now()
            print("  Stream Sampler:  ", streamSampler(fstream, perc, 2))
            print("Time in milliseconds = ", (datetime.now() - start_time).total_seconds() * 1000)
