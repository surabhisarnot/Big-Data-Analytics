from pprint import pprint
import json
import re
from scipy import stats
import numpy as np
import sys
from pyspark import SparkContext, SparkConf


# calculate frequencies of words
def get_word_records(review):
    return_list = []
    try:
        reviewText = review['reviewText']
        list = re.findall(pattern, reviewText)
        d = dict()
        for word in list:
            word = word.lower()
            d[word] = d.get(word, 0) + 1
        length = len(list)
        s = set(list)
        for word in top_1000_words.value:
            frequency = 0.0
            if word in s:
                frequency = float(d.get(word)) / length
            return_list.append((word,
                                (frequency, review['overall'], 1 if review['verified'] else 0)))
    except Exception as e:
        for word in top_1000_words.value:
            return_list.append((word, (0.0, 0.0, 0)))
    return return_list


# single variable linear regression
def linear_regression(word_records):
    x = []
    y = []
    for record in word_records:
        x.append(record[0])
        y.append(record[1])
    X = np.reshape(x, (len(x), 1))
    Y = np.reshape(y, (len(y), 1))
    # standardize
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    X = np.hstack((X, np.ones((len(x), 1))))
    X_transpose = np.transpose(X)
    betas = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X)), X_transpose), Y)
    # degree of freedom(N-(m+1))
    df = len(x) - (1 + 1)
    rss = np.sum(((Y - np.dot(X, betas)) ** 2))
    s_square = rss / df
    se = np.sqrt(s_square / np.sum(((X[:, 0]) ** 2)))
    t = betas[0][0] / se
    pval = stats.t.sf(np.abs(t), df) * 2
    return betas[0][0], pval * 1000


# multivariate regression
def multivariate_regression(word_records):
    x = []
    y = []
    for record in word_records:
        x.append([record[0], record[2]])
        y.append(record[1])
    X = np.reshape(x, (len(x), 2))
    Y = np.reshape(y, (len(y), 1))
    # standardize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Y = (Y - np.mean(Y)) / np.std(Y)
    X = np.hstack((X, np.ones((len(x), 1))))
    X_transpose = np.transpose(X)
    betas = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X)), X_transpose), Y)
    # degree of freedom(N-(m+1))
    df = len(x) - (2 + 1)
    rss = np.sum(((Y - np.dot(X, betas)) ** 2))
    s_square = rss / df
    se = np.sqrt(s_square / np.sum(((X[:, 0]) ** 2)))
    t = betas[0][0] / se
    pval = stats.t.sf(np.abs(t), df) * 2
    # multiplying pvalue by 1000 to adjust p-value according to the Bonferroni correction
    return (betas[0][0], pval * 1000)


conf = SparkConf()
sc = SparkContext(conf=conf)
reviewrdd = sc.textFile(sys.argv[1]).map(json.loads)

# precomplie regular expression
pattern = re.compile(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))')

# identify top 1000 words
top_words = reviewrdd.map(lambda review: review['reviewText'] if 'reviewText' in review else "").flatMap(
    lambda reviewText: re.findall(pattern, reviewText)).map(
    lambda word: (word.lower(), 1)).reduceByKey(lambda a, b: a + b).takeOrdered(1000, lambda v: -1 * v[1])

# create list of top 1000 words and broadcast
top_words_list = []
for word in top_words:
    top_words_list.append(word[0])
top_1000_words = sc.broadcast(top_words_list)

# gather all word frequencies together
word_rdd = reviewrdd.flatMap(lambda review: get_word_records(review)).groupByKey().mapValues(list)

# perform linear regression
rdd_linear = word_rdd.mapValues(linear_regression)
pos_corr = rdd_linear.takeOrdered(20, lambda x: -x[1][0])
neg_corr = rdd_linear.takeOrdered(20, lambda x: x[1][0])

# perform multivariate regression
rdd_multi = word_rdd.mapValues(multivariate_regression)
pos_corr_control = rdd_multi.takeOrdered(20, lambda x: -x[1][0])
neg_corr_control = rdd_multi.takeOrdered(20, lambda x: x[1][0])

# print results
print("Top 20 positively correlated words are:")
pprint(pos_corr)
print("Top 20 negatively correlated words are:")
pprint(neg_corr)
print("Top 20 positively correlated words with control variable as verified are:")
pprint(pos_corr_control)
print("Top 20 negatively correlated words with control variable as verified are:")
pprint(neg_corr_control)

# spark-submit a3_p1_sarnot_112584690.py 'hdfs:/user/surabhi/Software_5.json.gz'
