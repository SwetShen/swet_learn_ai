"""
Cross entropy involves learning about the magnitude
of information in information theory, followed by
learning about KL divergence (relative entropy)
to measure the gap between models. Finally, the
cross entropy formula is derived from Gibbs' inequality.
"""
import math

P = 0.5
Q = 0.5
cross_entropy = -(P * math.log(Q) + (1 - P) * math.log(1 - Q))
print(cross_entropy)
