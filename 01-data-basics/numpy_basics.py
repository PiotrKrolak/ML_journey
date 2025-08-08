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
# n-d array
e = np.array([1,2,3,4,5],  float)

arrays_names = [a, b, c, d, e]

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

# Shape of an Array
print("\nShape of an Array: ")
print(c.shape)

# Reshape array from 1-d to...
arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

new_array = arr.reshape(4,3)
new_array_2 = arr.reshape(2, 3, 2)

print(f"\nReshape array from {arr} to: \n{new_array}")
print(f"\nReshape array from {arr} to: \n{new_array_2}")


# Flattening the arrays
# Flattening array means converting a multidimensional array into a 1D array.
# We can use reshape(-1) to do this.
arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)

print("\nChange 2d array to 1d array: ")
print(newarr)


# enumerate
print("\nenumerate 2d: ")
for idx, x in np.ndenumerate(arr):
  print(idx, x)

arr = np.array([1, 2, 3])

print("\nenumerate 1d: ")
for idx, x in np.ndenumerate(arr):
  print(idx, x)

