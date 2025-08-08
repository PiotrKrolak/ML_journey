# https://www.w3schools.com/python/numpy/default.asp

import numpy as np

print("NumPy version: " + np.__version__ + "\n")

# create na NumPy array:
# 0d array
a = np.array(123)
# 1d array
b = np.array([1,2,3,4,5])
# 2d array
c = np.array([[6,5,4],[3,2,1]])
# 3d array
d = np.array([[[1,2,3],[4,5,6]],[[123,234,345],[456,567,678]]])

arrays_names = [a, b, c,d]

for arr in arrays_names:
    print(arr)
    print(type(arr))
    print(arr.ndim)
    print("\n")

# get acces to array
print("Get acces to array:")
print(b[2])
print(c[1,2])
print(d[1,1,1])

print("\nSlicing: ")
print(b[1:3])
print(b[2:-1])

# najpier wycinam a potem wskazuje co ktory krok wypisuje wartosc
print("\nSTEP: ")
print(b[1:4: 2])
print(b[::3])


# https://www.w3schools.com/python/numpy/numpy_data_types.asp

