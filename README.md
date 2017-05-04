# predictiveTyping
This project seeks to use hidden markov models to implement predictive typing. It is a project for cs325.

# Getting Started
## Prerequisites
Python, Pip
## Installing Dependencies
This project uses the [tensorflow_hmm](https://github.com/dwiel/tensorflow_hmm) python library. To install, 
```
pip install tensorflow_hmm
```
This project also uses dill
```
pip install dill
```
This project uses nltk's brown corpus
```
pip install nltk
nltk.download()
```
Then, download the brown corpus.

# Running

## Building a model
Before a model can be run, it must be built. The following builds a model from 10000 words in brown and stores it in ```models/brownCorpus10000.p```. Building a model of 100,000 words can take 10 minutes and easily use 32GB of ram.
```
python predict.py -b -s 10000
```
## Running a build model
To run a previously built model, remove the ```-b``` flag.
```
python predict.py -s 10000
```
This creates a repl in which sentences can be typed.

# Scoring
The ```score.py``` program scores a model of size 100,000 and can be run as follows:
```
python score.py
```
This can easily take 10 minutes and lots of ram.

# Project Report

## Description:
This project was an attempt to create a predicitve typing model similar to iMessage's. We initially were planning on using a Hidden Markov Model to predict the next word given a string of words, but we ended up finding a Brown POS Tagger that made this design choice moot. In the end, our model consisted of three submodels:
* Semantic Model: This model contains the probability of ending up at an end word given a start word for every word in the corpus.
* Transition Model: This model contains the probability of ending up at a tag given a previous tag for every tag in Brown.
* Observation Model: This model contains the probability of going from a tag to a word for every word-tag pair in the corpus.

To calculate the total probability / score of a word suggestion, the model takes the product of all the submodels' scores, however, it squares the semantic probability in order to achieve more accurate predictions. An input to the model is a string that is tagged by the brown tagger. Our model then returns a sorted list of the most likely words (with their probabilities) which we truncate to the three most likely.


## Evaluation of the Model
To evaluate the model, we took sentences from brown, took the first part of them, and asked our model what the next word would be. We have two sets of results for 100 sentences:
* Testing on training data: 20% correct
* Testing on very different data: 5% correct

Improving these almost definitely requires a better algorithm. We could look at trigrams instead of bigrams and look for a better training corpus that includes more similar data since the Brown corpus was divided into semantically different 2000 word chunks. Another improvement would be to tune the weighting of the different submodels using machine learning to generate more semantically correct sentences. As of now, they are hard coded and probably not very accurate due to our lack of knowledge of the subject matter.


