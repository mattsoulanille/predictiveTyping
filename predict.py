import nltk, pdb
from nltk.corpus import brown

from nltk.tokenize import word_tokenize
import pdb

from model import model


def predict():
    while (True):
        sent = raw_input("Enter sentence: ")
        if (sent != ""):
            nextWords = trained.nextOverall(sent)[0:3]
            for pair in nextWords:
                print str(pair[0]) + "\t" + str(pair[1])


from deepnlp import pos_tagger
if __name__ == "__main__":
    import argparse
    import dill
    parser = argparse.ArgumentParser(description='Predictive typing')
    parser.add_argument('-b', '--build', action="store_true")
    args = parser.parse_args()
    
    filePath = "models/brownCorpus.p"
    #corpus = brown.tagged_words()[0:1000]
    corpus = brown.tagged_words()[0:100000]
    tagger = pos_tagger.load_model(lang = 'en')
    def tagger_function(words):
        return [(x[0], x[1].upper()) for x in tagger.predict(words)]

    trained = model(corpus, tagger_function)

    if (args.build):
        
        trained.build()
        with open(filePath, "wb") as saveFile:
            trained.save(saveFile)
            
    else:
        data = dill.load(open(filePath, "rb"))
        trained.build(data)

        predict()
    
