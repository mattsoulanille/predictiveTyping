from nltk.corpus import brown

if __name__ == "__main__":
    corpus = brown.tagged_words()[0:100]

from collections import defaultdict


def make_probability(pairs, floor):
    # Calculates C(x -> a) / C(x) for a given set of X and a set of a
    totals = defaultdict(lambda: defaultdict( lambda: floor) )
    probabilities = defaultdict(lambda: defaultdict( lambda: floor) )

    startpoints = set(x[0] for x in pairs)
    endpoints = set(x[1] for x in pairs)

    for pair in pairs:
        totals[pair[0]][pair[1]] += 1

    for s in startpoints:
        for e in endpoints:
            probabilities[s][e] = totals[s][e] / sum( totals[s].values() )

            
    return probabilities

def make_observation_probabilities(corpus, floor=10**-15):
    return make_probability([(t[1], t[0]) for t in corpus], floor)

    


def make_transition_probabilities(corpus, floor=10**-15):
    transitions = []
    for i in range(len(corpus) - 1):
        pair = [corpus[i][1], corpus[i+1][1]]
        transitions.append(pair)
    return make_probability(transitions, floor)
