'''
Team SPHS members:
1) Surabhi Sarnot (112584690)
2) Priyanka Datar (112681258)
3) Himanshu Agrawal (112680639)
4) Sri Sneha Geetha Retineni (112671507)


General description:
 This is the spark version of pre-process. We have also attached a .ipynb python version also named: data_preparation.ipynb
 This file is used to preprocess the data and merge all rewuired datasets. The datasets are : college score card data, county wise unemployment data, countywise education attainment data, countywise population data, county FIPS code file for all  US counties.
 We are using Apache Spark RDD transformations and Spark SQL dataframe concepts. We use Map Reduce concepts on HDFS for distributed and parallel processing.

Frameworks used:
1) Apache Spark
2) Map Reduce HDFS

Execution Environment:
Google Cloud DataProc 1 master node and 2 worker nodes
Image Version: 1.4 (Debian 9, Hadoop 2.9, Spark 2.4)
Configuration of Assignment 3 CSE 545

Execution command format:
spark-submit data_preprocess.py 'hdfs file paths for all files needed to merge as seperate args'


'''

import numpy as np
from scipy import stats
import sys
import findspark
findspark.init()
from pyspark.sql import SparkSession


from pyspark import SparkContext

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = SparkContext.getOrCreate()

file_college = sys.argv[1]
file_zip = sys.argv[2]
file_unemp = sys.argv[3]
file1_edu = sys.argv[4]
file_pop = sys.argv[5]

''' Function to remove columns with % of null values > 40%'''	
def deleteNullCols(rec):
	nullCount = 0
	for value in list(rec[1]):
		value=value[2]
		if value=="PrivacySuppressed" or value=='':
			nullCount = nullCount+1
		else:
			if value is None or (type(value) in [int, float] and np.isnan(value)):
				nullCount = nullCount+1
	if nullCount/rdd_size.value > 0.4:
		return False
	return True

'''Function to fill null values with the county level aggregated mean value for that column.'''	
def fillNA(rec):
	value_list=[]
	result_list=[]
	if type(list(rec[1])[0]) in [int,float]:
		for value in list(rec[1]):
			if value=="PrivacySuppressed" or value=='' or value is None or (type(value) in [int, float] and np.isnan(value)):
				continue
			else:
				value_list.append(value)
		mean=np.mean(value_list)
		for value in list(rec[1]):
			if value=="PrivacySuppressed" or value=='' or value is None or (type(value) in [int, float] and np.isnan(value)):
				result_list.append(mean)
			else:
				result_list.append(value)
		return (rec[0],result_list)
	else:
		return (rec[0],rec[1])
        
'''Calculate Education Quality Index'''
def calcEQI(result):
    for rec in result:
        perc_total = rec[1][header_dict.value['UG12MN']]*100/rec[1][header_dict.value['POP_ESTIMATE']
        EQI = 0.4*rec[1][header_dict.value['Percent of adults with a high school diploma only']]+0.6*rec[1][header_dict.value["Percent of adults completing some college or associate's degree"]]+0.8*rec[1][header_dict.value["Percent of adults with a bachelor's degree or higher"]]+perc_total
        EQI = EQI/100
    return (result[0], (result[1].append(EQI)))
		
'''Read all necessary datafiles and join them on the county FIPS code column.'''
df = spark.read.csv(file_college, header=True)
df_zip = spark.read.csv(file_zip, header=True)
updatedDf = df.withColumn("ZIP", (df.ZIP.substr(0,5)))
zipjoineddf = updatedDf.join(df_zip, updatedDf.ZIP  == df_zip.ZIP)
zipjoineddf = zipjoineddf.drop(df_zip.CITY)
zipjoineddf = zipjoineddf.drop(df_zip.ZIP)

df_unemp = spark.read.csv(file_unemp, header=True)
zip_unemp_df = zipjoineddf.join(df_unemp, zipjoineddf.STCOUNTYFP  == df_unemp.STCOUNTYFP)
zip_unemp_df = zip_unemp_df.drop(df_unemp.STCOUNTYFP)

df_edu = spark.read.csv(file1_edu, header=True)
zip_unemp_edu_df = zip_unemp_df.join(df_edu, zip_unemp_df.STCOUNTYFP  == df_edu.FIPSCODE)
zip_unemp_edu_df = zip_unemp_edu_df.drop(df_edu.FIPSCODE)


df_pop = spark.read.csv(file_pop, header=True)
zip_unemp_edu_pop_df = zip_unemp_edu_df.join(df_pop, zip_unemp_edu_df.STCOUNTYFP  == df_pop.FIPStxt)
zip_unemp_edu_pop_df = zip_unemp_edu_pop_df.drop(df_pop.FIPStxt)

zip_unemp_edu_pop_df.write.csv("mydatanew.csv",header=True)

''' Read the merged csv data file as RDD.'''
data = sc.textFile("mydatanew.csv")

header = data.first()
header_list = header.split(",")

''' Filter out the data for year > 2010'''
county_rdd = data.filter(lambda row: row != header).map(lambda a: [idx for idx in a.split(',')]).filter(
    lambda a: int(a[header_dict.value['YEAR']]) >=2010).flatMap(lambda a: [(header_list[i], (a[0],a[header_dict.value['YEAR']],a[i])) for i in range(len(a))]).groupByKey().mapValues(list)

'''Preprocess the data by deleting columns with >40% null values and fill the null values of remaining columns as described above.'''	
county_rdd=county_rdd.filter(lambda rec : deleteNullCols(rec))
	
county_rdd = county_rdd.map(lambda rec:(rec[0],rec[1][0],rec[1][1]),rec[1][2]).groupByKey().map(lambda rec: fillNA(rec))
countsByKey = sc.broadcast(county_rdd.countByKey())

nulldeleted_rdd = mapped_rdd.groupByKey().filter(lambda rec : deleteNullCols(rec))

fillna_rdd=nulldeleted_rdd.map(lambda rec: fillNA(rec))

'''Aggregate the column values at county level.'''
noNArdd = fillna_rdd.reduceByKey(lambda x,y:x+y)
agg_rdd = noNArdd.map(lambda rec: (rec[0], rec[1]/countsByKey.value[rec[0]]))

'''Calculate EQI'''    
agg_rdd = agg_rdd.map(lambda rec: calcEQI(rec))

''' Save the preprocessed file'''
agg_rdd.saveAsTextFile("preprocessed_data.csv")


