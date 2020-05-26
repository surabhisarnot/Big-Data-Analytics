'''
Team SPHS members:
1) Surabhi Sarnot (112584690)
2) Priyanka Datar (112681258)
3) Himanshu Agrawal (112680639)
4) Sri Sneha Geetha Retineni (112671507)

General description:
 This code is used to find the similarity between two counties based on various factors of each county.
 It takes input from the preprocessed data created using file 'data_preparation.py'
 We are using  concept(using cosine similarity) and running code on Apache spark while using map reduce concepts on HDFS

Concepts used:
1) Similarity (cosine similarity)

Frameworks used:
1) Apache Spark
2) Map Reduce HDFS

Execution Environment:
Google Cloud DataProc 1 master node and 2 worker nodes
Image Version: 1.4 (Debian 9, Hadoop 2.9, Spark 2.4)
Configuration of Assignment 3 CSE 545

Execution command format:
spark-submit county_similarity.py 'hdfs file path'

'hdfs file path' is path to preprocessed_data.csv as mentioned above

'''

import numpy as np
from pprint import pprint
from pyspark import SparkContext, SparkConf
import sys


# Normalize the features in the dataframe using min-max normalization
def normalize(li):
    max_value = max(li)[1]
    min_value = min(li)[1]
    denominator = max_value - min_value
    if denominator == 0:
        li = [(county, 0) for county, value in li]
    else:
        li = [(county, (value - min_value) / denominator) for county, value in li]
    return li


# Find cosine similarity between two counties based on their features
def cosine_similarity(county_list_1, county_list_2):
    d1 = dict()
    for val in county_list_1:
        d1[val[0]] = val[1]
    d2 = dict()
    for val in county_list_2:
        d2[val[0]] = val[1]
    numerator = 0.0
    for val in county_list_1:
        numerator += d1[val[0]] * d2[val[0]]
    denominator = np.sqrt(np.sum(np.square(list(d1.values())))) * np.sqrt(np.sum(np.square(list(d2.values()))))
    return float(numerator) / denominator


# create spark context
conf = SparkConf()
sc = SparkContext(conf=conf)

# load the data
data = sc.textFile(sys.argv[1])

header = data.first()
header_list = header.split(",")

# 1) remove the header row
# 2) filter the county records for the year 2018
# 3) create key value pairs of form(column_name, (county, row_value))
# 4) using groupByKey() group all values of same column (column_name,[(county1, row_value),(county2, row_value)...])
# 5) normalize values of all the columns
# 6) create key value pairs of form(county, (column_name, row_value))
# 7) using groupByKey() group all values of same column (county,[(column_name1, row_value),(column_name2, row_value)...])
county_rdd = data.filter(lambda row: row != header).map(lambda a: [float(idx) for idx in a.split(',')]).filter(
    lambda a: a[1] == 2018).map(lambda a: [(header_list[i], (a[0], a[i])) for i in range(len(a))]).flatMap(
    lambda a: (a[7:])).groupByKey().mapValues(list).map(lambda a: (a[0], normalize(a[1]))).flatMap(
    lambda a: [(a[1][i][0], (a[0], a[1][i][1])) for i in range(len(a[1]))]).groupByKey().mapValues(list)

# cartesian join of county_rdd with itself to get all the pairs of counties which forms our similarity matrix
cartesian_rdd = county_rdd.cartesian(county_rdd)

# filter duplicate records for county pairs of the form (a,b) and (b,a) and keep only (a,b)
# the similarity matrix is an upper triangular matrix
# Here we are using similarity concept - cosine similarity
similarity_rdd = cartesian_rdd.filter(lambda a: a[0][0] < a[1][0]).map(
    lambda a: ((a[0][0], a[1][0]), cosine_similarity(a[0][1], a[1][1])))

# printing top ten most similar and dissimilar county pairs ((county1_code,county2_code), similarity value)
print("Top ten similar county pairs are: ")
pprint(similarity_rdd.takeOrdered(10, lambda v: -1 * v[1]))

print("Top ten dissimilar county pairs are: ")
pprint(similarity_rdd.takeOrdered(10, lambda v: v[1]))

# Printing the similarity value for some pair of counties used to visualize our results
pprint(similarity_rdd.filter(lambda a: a[0][0] == 48323 and a[0][1] == 51610).take(1))