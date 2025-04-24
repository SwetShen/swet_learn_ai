"""
In theory, the probability of flipping a
coin is 0.5 on the positive side and 0.5
 on the negative side

But if we flip a coin, there are 9 times
we flip the obverse and 1 time we flip
the reverse, which means the actual probability
of occurrence does not match the theoretical value
"""

# likehood value
print((0.1 ** 1) * (0.9 ** 9))

# if we have different event
l1 = 0.1 ** 1 * 0.9 ** 9
l2 = 0.7 ** 7 * 0.3 ** 3
l3 = 0.4 ** 4 * 0.6 ** 6

# The Ideal Probability Model
best_l = 0.5 ** 10
# The likelihood value closest to
# the ideal probability model is
# called the maximum likelihood value
print(l1 - best_l, l2 - best_l, l3 - best_l)
