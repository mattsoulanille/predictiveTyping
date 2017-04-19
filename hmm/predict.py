import nltk, pdb
#from nltk.corpus import brown
#corpus = brown.tagged_words()
from multiprocessing import Process, Pool
from nltk.corpus import treebank
corpus = treebank.tagged_words()


from nltk.tokenize import word_tokenize
import pdb

from model import model


def predict():
    while (True):
        sent = raw_input("Enter sentence: ")
        nextWords = trained.nextOverall(sent)[0:3]
        print nextWords


if __name__ == "__main__":
    trained = model(corpus)


    # s = Process(target=trained.buildSemantic)
    # t = Process(target=trained.buildTransition)
    # o = Process(target=trained.buildObservation)

    predict()
