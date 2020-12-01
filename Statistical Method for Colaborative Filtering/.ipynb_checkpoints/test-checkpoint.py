import numpy as np
import em
import common
import naive_em

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here
mix, post = naive_em.estep(X, )