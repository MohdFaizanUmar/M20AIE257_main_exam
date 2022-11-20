import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets


# Test case to validate that random state are same
def test_case_random(xtrain1, xtest1, xtrain2, xtest2):
    comparison1 = xtrain1 == xtrain2
    comparison2 = xtest1 == xtest2
    comp1 = comparison1.all()
    comp2 = comparison2.all()
    assert (comp1 and comp2)
    print("Test case passed")

# Test case to validate that random state are different 
def test_case_not_random(xtrain1, xtest1, xtrain2, xtest2):
    comparison1 = xtrain1 == xtrain2
    comparison2 = xtest1 == xtest2
    comp1 = comparison1.all()
    comp2 = comparison2.all()
    assert not (comp1 and comp2)
    print("Test case passed")






