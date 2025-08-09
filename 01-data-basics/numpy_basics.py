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


# Stack function
# z dwóch macierzy jednowymiarowych tworzy jedną w ktorej kazdemu wymiarowi odpowiada n'ta pozycja w pierwotnej macierzy
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print("\nStack function: ")
print(f"\nArray 1: \n{arr1}")
print(f"\nArray 2: \n{arr2}")
print(f"\nStack array 1 and array 2: \n{arr}")

# hstack - dodaje dwie jednowymiarowe macierze do jednej jednowymiarowej
arr = np.hstack((arr1, arr2))

print(f"\nhStack array 1 and array 2: \n{arr}")

#dStack dodaje maciesze o rozmiarze n i tworzy jedna n wymiarowa macierz
arr = np.dstack((arr1, arr2))

print(f"\ndStack array 1 and array 2: \n{arr}")

#Array Split
#dziel macierz jednowymiarowa na x macierzy jednowymiarowych
arr = np.array([1, 2, 3, 4, 5, 6])

print(f"\nArray for split: {arr}")

newarr = np.array_split(arr, 3)

print("\nSplit array: ")
print(newarr)

# Split the 2-D array into three 2-D arrays.
print("\nSplit the 2-D array into three 2-D arrays.")
print(newarr)

# The example above returns three 2-D arrays.
# In addition, you can specify which axis you want to do the split around.
# The example below also returns three 2-D arrays, but they are split along the column (axis=1).
print("\nThe example below also returns three 2-D arrays, but they are split along the column (axis=1).")
print(newarr)


# Searcjing Arrays
arr = np.array([1, 2, 3, 4, 5, 4, 4])

print(f"\n\nArray for Searching: {arr}")

# Find the indexes where the value is 4:
value = 4
x = np.where(arr == value)

print(f"return array of index where value is {value}")
print(x)


# Sort Array
print("\nSort Array: ")
arr_num = np.array([3, 2, 0, 1])
arr_str = np.array(['banana', 'cherry', 'apple'])
arr_bool = np.array([True, False, True])
arr = np.array([[3, 2, 4], [5, 0, 1]])


print(f"\nNumeric Array: {arr_num}")
print("Sorted num array: ")
print(np.sort(arr_num))

print(f"\nNumeric Array: {arr_str}")
print("Sorted str array: ")
print(np.sort(arr_str))

print(f"\nNumeric Array: {arr_bool}")
print("Sorted bool array: ")
print(np.sort(arr_bool))

print(f"\nNumeric Array: {arr}")
print("Sorted arr: ")
print(np.sort(arr))


# Filtering Arrays
# Create an array from the elements on index there "True":

arr = np.array([41, 42, 43, 44])

x = [True, False, True, False]

newarr = arr[x]
print(f"\nOryginall array: {arr}")
print(f"filter x: {x}")
print(newarr)