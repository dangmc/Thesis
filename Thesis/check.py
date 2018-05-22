import numpy as np


a = [1]*10
b = [2]*10

a = np.array(a)
b = np.array(b)

c = np.concatenate((a, b), axis=0)


a = [3]*10
b = [4]*10

a = np.array(a)
b = np.array(b)
d = np.concatenate((a, b), axis=0)

e = []
e.append(c)
e.append(d)

e = np.array(e)
print(e)
