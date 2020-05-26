##########################################################################
## MRSystemSimulator2020.py  v 0.1
##
## Implements a basic version of MapReduce intended to run
## on multiple threads of a single system. This implementation
## is simply intended as an instructional tool for students
## to better understand what a MapReduce system is doing
## in the backend in order to better understand how to
## program effective mappers and reducers.
##
## MyMapReduce is meant to be inheritted by programs
## using it. See the example "WordCountMR" class for
## an exaample of how a map reduce programmer would
## use the MyMapReduce system by simply defining
## a map and a reduce method.
##
##
## Original Code written by H. Andrew Schwartz
## for SBU's Big Data Analytics Course
## Spring 2020
##
## Student Name: KEY
## Student ID:

##IO, Process Imports:
from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Manager
from pprint import pprint

import mmh3
##Data Science Imports:
import numpy as np


##########################################################################
##########################################################################
# MapReduceSystem:

class MapReduce:
    __metaclass__ = ABCMeta

    def __init__(self, data, num_map_tasks=5, num_reduce_tasks=3, use_combiner=False):
        self.data = data  # the "file": list of all key value pairs
        self.num_map_tasks = num_map_tasks  # how many processes to spawn as map tasks
        self.num_reduce_tasks = num_reduce_tasks  # " " " as reduce tasks
        self.use_combiner = use_combiner  # whether or not to use a combiner within map task

    ###########################################################
    # programmer methods (to be overridden by inheriting class)

    @abstractmethod
    def map(self, k, v):
        print("Need to override map")

    @abstractmethod
    def reduce(self, k, vs):
        print("Need to overrirde reduce")

    ###########################################################
    # System Code: What the map reduce backend handles

    def mapTask(self, data_chunk, namenode_m2r, combiner=False):
        # runs the mappers on each record within the data_chunk and assigns each k,v to a reduce task
        mapped_kvs = []  # stored keys and values resulting from a map
        for (k, v) in data_chunk:
            # run mappers:
            chunk_kvs = self.map(k, v)  # the resulting keys and values after running the map task
            mapped_kvs.extend(chunk_kvs)

        # assign each kv pair to a reducer task
        if combiner:
            dict = {}
            # do the reduce from here before passing to reduceTask
            for m in mapped_kvs:
                if not m[0] in dict:
                    dict[m[0]] = []
                dict[m[0]].append(m[1])

            for k, v in dict.items():
                namenode_m2r.append((self.partitionFunction(k), self.reduce(k, v)))
        else:
            for (k, v) in mapped_kvs:
                namenode_m2r.append((self.partitionFunction(k), (k, v)))

    def partitionFunction(self, k):
        # given a key returns the reduce task to send it
        ##<<COMPLETE>>
        node_number = mmh3.hash(str(k), 0) % self.num_reduce_tasks
        return node_number

    def reduceTask(self, kvs, namenode_fromR):
        # sort all values for each key (can use a list of dictionary)
        vsPerK = dict()
        for (k, v) in kvs:
            try:
                vsPerK[k].append(v)
            except KeyError:
                vsPerK[k] = [v]

        # call reducers on each key with a list of values
        # and append the result for each key to namenoe_fromR
        for k, vs in vsPerK.items():
            if vs:
                fromR = self.reduce(k, vs)
                if fromR:  # skip if reducer returns nothing (no data to pass along)
                    namenode_fromR.append(fromR)

    def runSystem(self):
        # runs the full map-reduce system processes on mrObject

        # the following two lists are shared by all processes
        # in order to simulate the communication
        namenode_m2r = Manager().list()  # stores the reducer task assignment and
        # each key-value pair returned from mappers
        # in the form: [(reduce_task_num, (k, v)), ...]
        namenode_fromR = Manager().list()  # stores key-value pairs returned from reducers
        # in the form [(k, v), ...]

        # Divide up the data into chunks according to num_map_tasks
        # Launch a new process for each map task, passing the chunk of data to it.
        # Hint: The following starts a process
        #      p = Process(target=self.mapTask, args=(chunk,namenode_m2r))
        #      p.start()
        runningProcesses = []

        tasks = self.num_map_tasks
        start_index = 0
        length = len(self.data)

        while tasks > 0:
            ceil = int(np.ceil(length / tasks))
            end_index = start_index + ceil
            length -= ceil
            tasks -= 1
            p = Process(target=self.mapTask,
                        args=(self.data[start_index:(int)(end_index)], namenode_m2r, self.use_combiner))
            start_index = end_index
            runningProcesses.append(p)
            p.start()

        # join map task running processes back
        for p in runningProcesses:
            p.join()
            # print output from map tasks
        print("namenode_m2r after map tasks complete:")
        pprint(sorted(list(namenode_m2r)))

        # "send" each key-value pair to its assigned reducer by placing each
        # into a list of lists, where to_reduce_task[task_num] = [list of kv pairs]
        to_reduce_task = [[] for i in range(self.num_reduce_tasks)]

        for (i, j) in namenode_m2r:
            to_reduce_task[i].append(j)

        # launch the reduce tasks as a new process for each.
        runningProcesses = []
        for kvs in to_reduce_task:
            runningProcesses.append(Process(target=self.reduceTask, args=(kvs, namenode_fromR)))
            runningProcesses[-1].start()

        # join the reduce tasks back
        for p in runningProcesses:
            p.join()
        # print output from reducer tasks
        print("namenode_fromR after reduce tasks complete:")
        pprint(sorted(list(namenode_fromR)))

        # return all key-value pairs:
        return namenode_fromR


##########################################################################
##########################################################################
##Map Reducers:

class WordCountBasicMR(MapReduce):  # [DONE]
    # mapper and reducer for a more basic word count
    # -- uses a mapper that does not do any counting itself
    def map(self, k, v):
        kvs = []
        for w in v.split():
            kvs.append((w.lower(), 1))
        return kvs

    def reduce(self, k, vs):
        return (k, np.sum(vs))

    # an example of another map reducer


class SetDifferenceMR(MapReduce):
    # contains the map and reduce function for set difference
    # Assume that the mapper receives the "set" as a list of any primitives or comparable objects
    def map(self, k, v):
        toReturn = []
        for i in v:
            toReturn.append((i, k))
        return toReturn

    def reduce(self, k, vs):
        if len(vs) == 1 and vs[0] == 'R':
            return k
        else:
            return None


class MeanCharsMR(MapReduce):  # [TODO]
    def map(self, k, v):
        # <<COMPLETE>>
        pairs = []
        s = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z'}
        map_chars = {}
        for ch in v:
            if ch.isalpha():
                ch = ch.lower()
                map_chars[ch] = map_chars.get(ch, 0) + 1
                s.discard(ch)
        for ch in s:
            map_chars[ch] = 0

        for key, value in map_chars.items():
            pairs.append((key, (0.0, 0.0, value, value ** 2, 1)))
        return pairs

    def reduce(self, k, vs):
        summation = 0.0
        sum_of_squares = 0.0
        doc_count = 0

        for v in vs:
            summation += v[2]
            sum_of_squares += v[3]
            doc_count += v[4]

        mean = round((summation / doc_count), 4)
        std = round((abs((sum_of_squares / doc_count) - mean ** 2) ** 0.5), 4)

        return k, ("Mean = " + str(mean), "Standard Deviation = " +str(std), summation, sum_of_squares, doc_count)


##########################################################################
##########################################################################

from scipy import sparse


def createSparseMatrix(X, label):
    sparseX = sparse.coo_matrix(X)
    list = []
    for i, j, v in zip(sparseX.row, sparseX.col, sparseX.data):
        list.append(((label, i, j), v))
    return list


if __name__ == "__main__":  # [Uncomment peices to test]

    ###################
    ##run WordCount:

    print("\n\n*****************\n Word Count\n*****************\n")
    data = [(1, "The horse raced past the barn fell"),
            (2, "The complex houses married and single soldiers and their families"),
            (3, "There is nothing either good or bad, but thinking makes it so"),
            (4, "I burn, I pine, I perish"),
            (5, "Come what come may, time and the hour runs through the roughest day"),
            (6, "Be a yardstick of quality."),
            (7, "A horse is the projection of peoples' dreams about themselves - strong, powerful, beautiful")]
    print("\nWord Count Basic WITHOUT Combiner:")
    mrObjectNoCombiner = WordCountBasicMR(data, 3, 3)
    mrObjectNoCombiner.runSystem()
    print("\nWord Count Basic WITH Combiner:")
    mrObjectWCombiner = WordCountBasicMR(data, 3, 3, use_combiner=True)
    mrObjectWCombiner.runSystem()

    ###################
    ##MeanChars:
    print("\n\n*****************\n Word Count\n*****************\n")
    data.extend([(8,
                  "I believe that at the end of the century the use of words and general educated opinion will have altered so much that one will be able to speak of machines thinking without expecting to be contradicted."),
                 (9, "The car raced past the finish line just in time."),
                 (10, "Car engines purred and the tires burned.")])
    print("\nMean Chars WITHOUT Combiner:")
    mrObjectNoCombiner = MeanCharsMR(data, 4, 3)
    mrObjectNoCombiner.runSystem()
    print("\nMean Chars WITH Combiner:")
    mrObjectWCombiner = MeanCharsMR(data, 4, 3, use_combiner=True)
    mrObjectWCombiner.runSystem()
