import json
import sys
from pprint import pprint

import numpy as np
from pyspark import SparkContext, SparkConf


# compute cosine similarity
def cosine_similarity(curr_item_list):
    similarity = 0.0
    s = set([user for user, rating in curr_item_list]).intersection([user for user, rating in op_item_list_bv.value])
    if len(s) >= 2:
        curr_item_mean = np.mean([x[1] for x in curr_item_list])
        curr_item_list = [(user, rating - curr_item_mean) for user, rating in curr_item_list]
        d1 = dict()
        for val in op_item_list_bv.value:
            d1[val[0]] = val[1]
        d2 = dict()
        for val in curr_item_list:
            d2[val[0]] = val[1]
        numerator = 0.0
        for val in s:
            numerator += d1[val] * d2[val]
        denominator = np.sqrt(np.sum(np.square(list(d1.values())))) * np.sqrt(np.sum(np.square(list(d2.values()))))
        similarity = float(numerator) / denominator
    return similarity


# calculate item rating if user has not already rated the item
def get_rating(new_product, user_product_list, similarity_matrix):
    d1 = dict()
    for value in user_product_list:
        d1[value[0]] = value[1]
    if new_product in d1.keys():
        return d1[new_product]
    d2 = dict()
    for value in similarity_matrix:
        d2[value[0]] = value[1]
    numerator = 0.0
    denominator = 0.0
    for key in d2.keys():
        numerator += d1[key] * d2[key]
        denominator += d2[key]
    try:
        return float(numerator) / denominator
    except Exception as e:
        return 0.0
    return 0.0


conf = SparkConf()
sc = SparkContext(conf=conf)

user_rdd = sc.textFile(sys.argv[1]).map(json.loads).map(
    lambda review: ((review["reviewerID"], review["asin"]), (review["overall"]))).reduceByKey(
    lambda a, b: b).map(lambda a: (a[0][1], (a[0][0], a[1]))).groupByKey().mapValues(list).filter(
    lambda a: len(a[1]) >= 25).flatMap(
    lambda a: [(a[1][i][0], (a[0], a[1][i][1])) for i in range(len(a[1]))]).groupByKey().mapValues(list).filter(
    lambda a: len(a[1]) >= 5)

item_rdd = user_rdd.flatMap(
    lambda a: [(a[1][i][0], (a[0], a[1][i][1])) for i in range(len(a[1]))]).groupByKey().mapValues(list)

for item in eval(sys.argv[2]):
    op_item_list = item_rdd.filter(lambda a: a[0] == item).flatMap(lambda a: a[1]).collect()
    op_item_rating_mean = np.mean([x[1] for x in op_item_list])
    op_item_list = [(user, rating - op_item_rating_mean) for user, rating in op_item_list]
    op_item_list_bv = sc.broadcast(op_item_list)
    similarity_matrix = item_rdd.map(lambda a: (a[0], cosine_similarity(a[1]))).filter(
        lambda a: a[1] > 0).takeOrdered(50, lambda v: -1 * v[1])
    similarity = dict()
    for value in similarity_matrix:
        similarity[value[0]] = value[1]
    print("----------Product ID", item, "----------")
    pprint(
        user_rdd.filter(lambda a: len(set([item for item, rating in a[1]]).intersection(similarity.keys())) >= 2).map(
            lambda a: (a[0], get_rating(item, a[1], [(item, similarity[item]) for item in
                                                     set([item for item, rating in a[1]]).intersection(
                                                         similarity.keys())]))).filter(
            lambda a: a[1] > 0).collect())


# spark-submit a3_p2_sarnot_112584690.py 'hdfs:/user/surabhi/Software_5.json.gz' "['B00EZPXYP4', 'B00CTTEKJW']"

def normalise(li):
    max_value = max(li)[1]
    min_value = min(li)[1]
    denominator = max_value - min_value
    if denominator == 0:
        li = [(county, 0) for county, value in li]
    else:
        li = [(county, (value - min_value) / denominator) for county, value in li]
    return li


def cosine_similarity(countylist, curr_county_list):
    similarity = 0.0
    d1 = dict()
    for val in countylist:
        d1[val[0]] = val[1]
    d2 = dict()
    for val in curr_county_list:
        d2[val[0]] = val[1]
    numerator = 0.0
    for val in countylist:
        numerator += d1[val[0]] * d2[val[0]]
    denominator = np.sqrt(np.sum(np.square(list(d1.values())))) * np.sqrt(np.sum(np.square(list(d2.values()))))
    similarity = float(numerator) / denominator
    return similarity


for county in county_list:
    op_county_list = county_rdd.filter(lambda a: a[0] == county).flatMap(lambda a: a[1]).collect()
    similarity_matrix = county_rdd.map(lambda a: (a[0], cosine_similarity(op_county_list, a[1]))).filter(
        lambda a: a[1] > 0).takeOrdered(50, lambda v: -1 * v[1])
