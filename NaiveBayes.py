'''
NaiveBayes class
Functions:
    train():
        - compute class priors : p(ham) and p(spam)
        - count word occurances separately for spam and ham
        - compute probabilities of words given a class p(word|ham), p(word|spam) with Laplace smoothing
    prediction():
        - compute the probability of a message being spam or ham
        - assign the label with the higher probability
'''