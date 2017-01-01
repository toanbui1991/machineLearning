# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:14:49 2016

@author: User
"""

#you need to know what is majority vote and plural vot.
#you have multiple classifier and decided the lable of 
#an obvervation base on the most decision.
#y_hat = mode{c1(x), c2(x), ..., cm(x)}
#majority means for binary dicision, mean more than 50 percnet.
#plural means multiple classification, choose the most one.
#for binray classification then the ensemble decision become.
#since class1 = 1, class2 = -1 we can rewrite.
#c(x) = sign(sum_of_j(cj(x))) = 1 if sum_of_j(cj(x)) > 0
#else = -1.
#why?
#because if majority then sum > 0 for 50% class1,
#if more than 50% cj is class 2 then sum < 0
#the sign help to transform to 1 or -1.
#if sign(possitive) = 1
#sign(negative) = -1


#now we know how the ensemble method work!
#the question is how it work?
#the answer as follow:
#Assumpe that we have n binary classfier with equal error rate e.
#then the error probability of the ensemble method is the probability mass
#function of binomial distribution function.
#P(y > k) = sum from k to n of (n,k)p^k(1-p)^(n-k)



#example: given 11 binary classification with the error rate e = 0.25
#and each classification is independence then
#what is the probability that the ensemble classication is wrong
#p(y >= 6) = sum frome 6 to 111 of(11, 6)0.25^6(1-0.25)(11-6) = ?

#example code. 
#note to computhe the probability of essemble error with ideal
#assumption we need to know means we calculate the probability
#of binomial mass function of like the formulart about.

#P(y > k) = sum_from_k_to_n of (n, k)*p^k(1-p)^(n-k)
#note comb function is to compute the combination of
#N think take k at at time.
from scipy.misc import comb
import math
def ensemble_error(n_classifier, error):
    k = math.ceil(n_classifier/2.0)
    error_array = [comb(n_classifier, k) * error**k
                   *(1-error)**(n_classifier - k)
                   for k in range(k, n_classifier + 1)]
    return sum(error_array)

ensemble_error(n_classifier = 11, error = 0.25)


#visualization of the the relationship between base error
#and the ensemble error
import numpy as np
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range] 
import matplotlib.pyplot as plt
plt.plot(error_range, ens_errors,
         label='Ensemble error',
         linewidth=2)
plt.plot(error_range, error_range,
         linestyle='--', label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()
         
#note: to visualization between the base error and the ensemble error
#we need x is a range of error, and y of the two line is the base_error
#and the ensemble error.
#first compute the x and the y
import numpy as np
error_range = np.arange(0, 1.01, 0.01)
ens_errors = [ensemble_error(11, error) for error in error_range]
#second plot the two line
import matplotlib.pyplot as plt
plt.plot(error_range, ens_errors,
         label='Ensemble error',
         linewidth=2)
plt.plot(error_range, error_range,
         linestyle='--', label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()

#summary: if the base error rate is smaller than 0.5
#then the ensemble error rate will be best ther than the
#base
#if the base_error is larger thant 0.5 then the ensemble
#error will be worst.

             