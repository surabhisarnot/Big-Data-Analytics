'''
Team SPHS members:
1) Surabhi Sarnot (112584690)
2) Priyanka Datar (112681258)
3) Himanshu Agrawal (112680639)
4) Sri Sneha Geetha Retineni (112671507)


General description:
 This file is used to run hypothesis testing to determine the most important features affecting education quality index. We run linear regression and comput the beta and p-values using T staticstic for each feature.
 Our target column here is the education_quality_index (EQI) a column as projected from data preprocessing and analysis.
 It takes input from the preprocessed data created using file ' data_preprocess.py'
 We are using the concepts of Hypothesis Testing and Linear Regression and running code on Apache spark while using map reduce concepts on HDFS

Concepts used:
1) Hypothesis Testing
2) Multiple Linear Regression

Frameworks used:
1) Apache Spark
2) Map Reduce HDFS

Execution Environment:
Google Cloud DataProc 1 master node and 2 worker nodes
Image Version: 1.4 (Debian 9, Hadoop 2.9, Spark 2.4)
Configuration of Assignment 3 CSE 545

Execution command format:
spark-submit hypothesis_testing.py 'hdfs file path'

'hdfs file path' is path to preprocessed_data.csv as mentioned above

'''
import csv, numpy as np
from scipy import stats
import sys

import findspark

findspark.init()
from pyspark.sql import SparkSession
from pyspark import SparkContext

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = SparkContext.getOrCreate()

filepath = sys.argv[1]

#Function to format the input data rdd such that key=column name and value is the list of values for the column.
def create_input_rdd(rec):
	maplist=[]
	for key,value in header_dict.value.items():
		index=value
		colname=key
		maplist.append((colname,(rec[index],rec[header_dict.value['EQI']])))
	return maplist
    
	
#Function to compute beta values, p-values and t statistic for each hypothesis	
def computeMeanStdDev(rec):
    column_name = rec[0]
	print(column_name)
    list_of_x = []
    list_of_y = []
    list_of_verified = []
    for row in list(rec[1]):
        x = float(row[0])
        y = float(row[1])
        list_of_x.append(x)
        list_of_y.append(y)
    mean_y = np.mean(list_of_y)
    mean_x = np.mean(list_of_x)
    stddev_y = np.std(list_of_y)
    stddev_x = np.std(list_of_x)
    std_x = [(x1 - mean_x) / stddev_x for x1 in list_of_x]
    std_y = [(y1 - mean_y) / stddev_y for y1 in list_of_y]
    y_array = np.array(std_y)
    x_array = np.array(std_x)
    x_array = x_array.transpose().reshape(x_array.shape[0], 1)
    y_array = y_array.transpose().reshape(y_array.shape[0], 1)
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_array.transpose(), x_array)), x_array.transpose()),y_array)
    beta_xi = beta[0][0] * x_array
    e_is = [(x - y) ** 2 for x, y in zip(y_array, beta_xi)]
    df = len(list_of_y) - (1 + 1)
    rss = np.sum(e_is)
    ssq = rss / df
    t_val = beta[0][0] / np.sqrt(ssq / ((np.std(x_array) ** 2) * len(y_array)))
    p_value = stats.t.sf(np.abs(t_val), df) * 2
    #p_value_corrected = p_value * 1000
    return (column_name, beta[0][0],p_value)

#Input rdd, extract the header and create a header to index mapping,remove headers. Mapping of header to index is cached as a broadcast variable
input = sc.textFile(filepath)
parsedRDD = input.mapPartitions(lambda part: csv.reader(part))
header=parsedRDD.first()
headerToIndex = dict([(header[i], i) for i in range(len(header))])
parsedRDD = parsedRDD.filter(lambda line:line != header)
header_dict=sc.broadcast(headerToIndex)
mapped_rdd=parsedRDD.flatMap(lambda rec:create_input_rdd(rec))

#Run Linear regression for each column
columns_rdd = mapped_rdd.groupByKey().map(lambda rec:computeMeanStdDev(rec))

#take columns with most positive and negative beta values which indicate the correlation with EQI (as data has been standardized and normalized
sort_rdd_desc = columns_rdd.takeOrdered(20, lambda rec: -1 * rec[1])
print(sort_rdd_desc)
sort_rdd_asc = columns_rdd.takeOrdered(20, lambda rec: rec[1])
print(sort_rdd_asc)


