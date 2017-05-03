import nltk
from nltk.corpus import brown
from deepnlp import pos_tagger
from model import model

corpus_start = 0
corpus_end = 100000


training_corpus = brown.tagged_words()[corpus_start:corpus_end]
#testing_corpus = brown.tagged_words()[test_start:test_end]
#testing_corpus = brown.sents()[100:200]
testing_corpus = brown.sents()[57230:57330]
#training_corpus = brown.tagged_words()[0:100000]


tagger = pos_tagger.load_model(lang = 'en')



def tagger_function(words):
    return [(x[0], x[1].upper()) for x in tagger.predict(words)]

trained = model(training_corpus, tagger_function)


correct = 0
total = 0
for sentence in testing_corpus:
    total += 1
    print "testing " + str(total)

    w = -2
    last = unicode(sentence[w])
    results = [x[0] for x in trained.nextOverall(' '.join(sentence[0:w]))[0:3]]
    guess = unicode(sentence[w]) in results

    if (guess):
        correct += 1

    print "Guess: " + str(results)
    print "Actual: " + str(last)
    print
        
score = correct / float(total) * 100
print "Percent correct: " + str(score)
    
