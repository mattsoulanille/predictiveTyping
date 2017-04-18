import nltk
#from nltk.corpus import brown
#corpus = brown.tagged_words()

from nltk.corpus import treebank
corpus = treebank.tagged_words()


from nltk.tokenize import word_tokenize
from corpus_analysis import *
import pdb

emit_dict = make_observation_probabilities(corpus)
trans_dict = make_transition_probabilities(corpus)

# Get a set of all the words
words = set()
for x in emit_dict.values():
    keys = set(x.keys())
    words = words.union(keys)

# Get a set of all the tags
tags = set(trans_dict.keys())
for x in trans_dict.values():
    keys = set(x.keys())
    tags = tags.union(keys)

    

    
def predict(sentence):
    # *-1 is a none value
    tokenized = nltk.word_tokenize(unicode(sentence))
    tagged = nltk.pos_tag(tokenized)#, tagset='brown')
    tag = tagged[-1][1]
    word = tagged[-1][0]

    nextTags = trans_dict[tag]
    nextTag = [u'UNK', 0]

    # Get most likely tag
    for k in nextTags.keys():
        if nextTags[k] > nextTag[1]:
            nextTag = [k, nextTags[k]]

    nextWords = sorted(emit_dict[nextTag[0]].items(), key=lambda x: -x[1])[0:3]
    return nextWords
    

    #pdb.set_trace()
    
