import numpy as np
from numba import vectorize

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

np.add(a, b)

np.add(a, 100)

c = np.arange(4*4).reshape((4,4))
print('c:', c)

np.add(b, c)



b_col = b[:, np.newaxis]
b_col

np.add(b_col, c)



@vectorize(['int64(int64, int64)'], target='cuda')
def add_ufunc(x, y):
    return x + y

print('a+b:\n', add_ufunc(a, b))
print()
print('b_col + c:\n', add_ufunc(b_col, c))