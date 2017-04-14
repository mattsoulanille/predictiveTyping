#import tensorflow as tf
import pdb
import numpy as np
from tensorflow_hmm import HMMNumpy

from nltk.corpus import brown
corpus = brown.tagged_words()[0:1000]

from collections import defaultdict

def make_observation_probabilities(corpus):
    tag_count = defaultdict(int)
    transition_count = defaultdict(int)

    for pair in corpus:
        tag_count[pair[1]] += 1
        
        transition_count[(pair[0].lower(), pair[1])] += 1

    floor = 10**-7
    probabilities = defaultdict(lambda: floor)
    for key in transition_count.keys():
        probabilities[key] = float(transition_count[key]) / tag_count[key[1]]
    
    
    words = list(set( [x[0] for x in transition_count.keys()] ))
    return probabilities, words


def make_transition_probabilities(corpus):
    tag_count = defaultdict(int)
    transition_count = defaultdict(int)

    for i in range(len(corpus) - 1):
        tag_count[corpus[i][1]] += 1
        
        transition = (corpus[i][1], corpus[i+1][1])
        transition_count[transition] += 1

    tag_count[corpus[-1][1]] += 1


    # Floor value for transitions
    floor = 10**-6
    probabilities = defaultdict(lambda: floor)
    for key in transition_count.keys():
        probabilities[key] = float(transition_count[key]) / tag_count[key[1]]

    return probabilities, tag_count.keys()



# Given DET, probability that you are "the" (or some other word)
emit_dict, words = make_observation_probabilities(corpus)
trans_dict, tags = make_transition_probabilities(corpus)

def get_point(tagpair):
    return (tags.index(tagpair[0]), tags.index(tagpair[1]))

def get_tagpair(point):
    return (tags[point[0]], tags[point[1]])


trans_list = [ [0 for x in range(len(tags))] for y in range(len(tags)) ]



for tagpair in trans_dict.keys():
    # Row must be first tag
    # Col must be second
    #states
    point = get_point(tagpair)
    trans_list[point[0]][point[1]] = trans_dict[tagpair]

trans = np.array(trans_list)

emi_list = [ [0 for x in range(len(tags))] for y in range(len(words))]

for w in range(len(words)):
    for t in range(len(tags)):
        word = words[w]
        tag = tags[t]
        value = emit_dict[(word, tag)]
        #pdb.set_trace()
        emi_list[w][t] = value

    
emi = np.array(emi_list)


# Temporary: All are equal for now
p0_list = [1.0 / len(tags) for t in range(len(tags))]
p0 = np.array(p0_list)

obs = words

obs_seq = np.array([0,1,2])
# From example.py
np_model = HMMNumpy(trans, p0)

def dptable(V, pathScores, states):
    print(" ".join(("%10d" % i) for i in range(V.shape[0])))
    for i, y in enumerate(pathScores.T):
        print("%.7s: " % states[i])
        print(" ".join("%.7s" % ("%f" % yy) for yy in y))



y = emi[obs_seq]
np_states, np_scores = np_model.viterbi_decode(y)
print("Most likely States: ", [obs[s] for s in np_states])
pathScores = np.array(np.exp(np_scores))
dptable(pathScores, pathScores, tags)



