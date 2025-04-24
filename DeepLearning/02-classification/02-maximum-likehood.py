"""
Taking coin toss as an example, the known probability model is as follows:
0.1:0.9
0.7:0.3
0.2:0.8
Due to the binary classification of coin toss, the value of x is only 0.1
Therefore, the maximum likelihood formula is:
Maximum likelihood value=sum (log (probability of each event occurring or not occurring))
"""
import math

l1 = 1 * math.log(0.1) + 9 * math.log(0.9)
l2 = 7 * math.log(0.7) + 3 * math.log(0.3)
l3 = 2 * math.log(0.2) + 8 * math.log(0.8)

print(l1, l2, l3)
