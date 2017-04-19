import nltk
from collections import defaultdict

class model:
    def __init__(self, corpus, floor=10**-15):
        
        self.corpus = corpus
        self.floor = floor

        
        self.buildSemantic()
        self.buildTransition()
        self.buildObservation()

    def buildSemantic(self):

        def gen_pairs():
            for i in range(len(self.corpus) - 1):
                pair = [self.corpus[i][0], self.corpus[i+1][0]]
                yield pair

        self.semantic_probabilities = self.make_probability(gen_pairs(), self.floor)
    
    def buildTransition(self):

        def gen_transitions():
            for i in range(len(self.corpus) - 1):
                pair = [self.corpus[i][1], self.corpus[i+1][1]]
                yield pair

        
        self.transition_probabilities = self.make_probability(gen_transitions(),
                                                              self.floor)


    def buildObservation(self):
        def gen_observations():
            for t in self.corpus:
                yield (t[1], t[0])

        self.observation_probabilities = self.make_probability(gen_observations(),
                                                               self.floor)

        
    def _next(self, probabilities, t):
        return sorted(probabilities[t].items(), key=lambda x: -x[1])

    def nextSemantic(self, word):
        # returns sorted list of next words
        return self._next(self.semantic_probabilities, word)

    def nextTransition(self, tag):
        return self._next(self.transition_probabilities, tag)
        
    def nextObservation(self, tag):
        return self._next(self.observation_probabilities, tag)
    

    def nextOverall(self, sentence):
        
        from itertools import product
        import pdb
        tokenized = nltk.word_tokenize(unicode(sentence))
        tagged = nltk.pos_tag(tokenized)
        prev_tag = tagged[-1][1]
        prev_word = tagged[-1][0]

        #s = self.nextSemantic(prev_word)[0:100]
        t = self.nextTransition(prev_tag)[0:10]
        #o = self.nextObservation(prev_tag)[0:100]

        probabilities = []
        for next_tag, probability_t in t:
            
            o = self.observation_probabilities[next_tag]

            for next_word, probability_o in o.iteritems():

                probability_s = self.semantic_probabilities[prev_word][next_word]

                total_probability = probability_t * probability_o * probability_s
                
                probabilities.append( (next_word, total_probability) )

        probabilities.sort(key=lambda x: -x[1])
        return probabilities
        


    def make_probability(self, pairs, floor):
        # Calculates C(x -> a) / C(x) for a given set of X and a set of a
        totals = defaultdict(lambda: defaultdict( lambda: floor) )
        probabilities = defaultdict(lambda: defaultdict( lambda: floor) )
        
        startpoints = set() # set(x[0] for x in pairs)
        endpoints = set() # set(x[1] for x in pairs)
        
        for pair in pairs:
            startpoints.add(pair[0])
            endpoints.add(pair[1])
            totals[pair[0]][pair[1]] += 1
            
        for s in startpoints:
            sum_s = sum( totals[s].values() )
            for e in endpoints:
                probabilities[s][e] = totals[s][e] / sum_s
                    
        return probabilities
