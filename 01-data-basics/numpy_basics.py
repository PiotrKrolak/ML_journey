# https://www.w3schools.com/python/numpy/default.asp

import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
import os

folder = 'distributions_plots'

# check if "distributions_plots" exist - if not create it.
if not os.path.exists(folder):
    os.makedirs(folder)


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



# Create a filter array that will return only values higher than 42:
# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print("\nCreate a filter array that will return only values higher than 42:")

print(f"Array: {arr}")
print(f"Filter array: {filter_arr}")



# Randm Numbers in NumPy

# Generate a random integer from 0 to 100
x = random.randint(100)


# Generate a random float from 0 to 1
y = random.rand()


print("\nGenerate Random Number: ")

print(f"Random intiger form 0 to 100:   {x}")
print(f"Random float from 0 to 1:       {y}")


# Generate Random Array
# array 1d
x = random.randint(100, size=(5))
# array 2d
y = random.randint(100, size=(3,5))

print(f"\nRandom array 1d: \n{x} ")
print(f"\nRandom array 2d: \n{y} ")


# Generate a 1-D array containing 5 random floats
x = random.rand(5)

#  Generate a 2-D array with 3 rows, each row containing 5 random numbers
y = random.rand(3, 5)

print(f"\nGenerate a 1-D array containing 5 random floats: \n{x}")
print(f"\nGenerate a 2-D array with 3 rows, each row containing 5 random numbers: \n{y}")


# Return one of the values
x = random.choice([3, 5, 7, 9])
print("\nReturn one of the values: 3, 5, 7, 9")
print(x)


# Generate a 2-D array that consists of the values in the array parameter (3, 5, 7, and 9)
x = random.choice([3, 5, 7, 9], size=(3, 5))
print("\nGenerate a 2-D array that consists of the values in the array parameter (3, 5, 7, and 9):")
print(x)

# Create random matrix 1d and 2d - contain specyfic number with separate probability for each

# Generate a 1-D array containing 100 values, where each value has to be 3, 5, 7 or 9.
# The probability for the value to be 3 is set to be 0.1
# The probability for the value to be 5 is set to be 0.3
# The probability for the value to be 7 is set to be 0.6
# The probability for the value to be 9 is set to be 0
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))

# Same example as above, but return a 2-D array with 3 rows, each containing 5 values.
y = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))


print(f"\n array 1D with specyfic value and probablility: \n{x}")
print(f"\n array 2D with specyfic value and probablility: \n{y}")




# Randomly shuffle elements of following array
arr = np.array([1, 2, 3, 4, 5])

print(f"\nOryginal Array: {arr}")

random.shuffle(arr)

print("\nShuffle: ")
print(arr)


# Generate a random permutation of elements of following array
arr = np.array([1, 2, 3, 4, 5])
print(f"\nOryginal Array: {arr}")

print("\nPermutation: ")
print(random.permutation(arr))
print(arr)



# Seaborn Module
sns.displot([0, 1, 2, 3, 4, 5])
#plt.show()
plt.savefig(os.path.join(folder, "Seaborn_Module.png"))
plt.close()

sns.displot([0, 1, 2, 3, 4, 5], kind="kde")
#plt.show()
plt.savefig(os.path.join(folder, "Seaborn_Module_kde.png"))
plt.close()

##########################################################################
# Normal Distribution                                                    #
##########################################################################
# It has three parameters:                                               #
# loc - (Mean) where the peak of the bell exists.                        #
# scale - (Standard Deviation) how flat the graph distribution should be.#
# size - The shape of the returned array.                                #
##########################################################################

# Generate a random normal distribution of size 2x3:
x = random.normal(size=(2, 3))
print("\nGenerate a random normal distribution of size 2x3")
print(x)

# Generate a random normal distribution of size 2x3 with mean at 1 and standard deviation of 2
x = random.normal(loc=1, scale=2, size=(2, 3))
print("\nGenerate a random normal distribution of size 2x3 with mean at 1 and standard deviation of 2")
print(x)



# VISUALISATION OF DIFERENT DISTRIBUTIONS

# Visualization of NORMAL DISTRIBUTION
str = "\nVisualization of NORMAL DISTRIBUTION"
print(str)
# y to losowe 1000 liczb w rozkladzie normalnym
y = random.normal(size=1000)

# wykres rozkladu normalnego
sns.displot(y, kind="kde")

#plt.figure(num=str)
plt.title(str)
#plt.show()
plt.savefig(os.path.join(folder, "normal_distribution.png"))
plt.close()


# Binomial Distribution
# Given 10 trials for coin toss generate 10 data points
x = random.binomial(n=10, p=0.5, size=10)

print("\nGiven 10 trials for coin toss generate 10 data points")
print(x)


##################################################################################
# Visualization of Binomial Distribution                                         #
##################################################################################
#It has three parameters:                                                        #
# n - number of trials.                                                          #
# p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each). #
# size - The shape of the returned array.                                        #
##################################################################################

str = "\nVisualization of Binomial Distribution"
print(str)
sns.displot(random.binomial(n=10, p=0.5, size=1000))

#plt.figure(num=str)
plt.title(str)
#plt.show()
plt.savefig(os.path.join(folder, "binormal_distribution.png"))
plt.close()

#####################################################################
# Difference Between Normal and Binomial Distribution               #
#####################################################################
print("\nDifference Between Normal and Binomial Distribution")
data = {
  "normal": random.normal(loc=50, scale=5, size=1000),
  "binomial": random.binomial(n=100, p=0.5, size=1000)
}


sns.displot(data, kind="kde")
plt.title("Difference Between Normal and Binomial Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "normal_vs_binormal.png"))
plt.close()


##################################################################################
# Poisson Distribution                                                           #
##################################################################################
# It has two parameters:                                                         #
# lam - rate or known number of occurrences e.g. 2 for above problem.            #
# size - The shape of the returned array.                                        #
##################################################################################

# Poisson Distribution
x = random.poisson(lam=2, size=10)
str = "Poisson Distribution"
print("\nPoisson Distribution: ")
print(x)


sns.displot(random.poisson(lam=2, size=1000))

plt.title(str)
#plt.show()
plt.savefig(os.path.join(folder, "poisson_dist.png"))
plt.close()


# Diference between normal, binormal i poisson distribution:
data = {
    "normal"    : random.normal(loc=50, scale=7, size=1000),
    "binormal"  : random.binomial(n=1000, p=0.01, size=1000),
    "poisson"   : random.poisson(lam=10, size=1000)
}

sns.displot(data, kind="kde")
plt.title("Difference normal - binormal - poisson distribution")
#plt.show()
plt.savefig(os.path.join(folder, "dif_normal-binormal-poisson.png"))
plt.close()



#################################################################################
# Uniform Distribution                                                          #
#################################################################################
# Used to describe probability where every event has equal chances of occuring. #
#                                                                               #
# It has three parameters:                                                      #
# low - lower bound - default 0 .0.                                             #
# high - upper bound - default 1.0.                                             #
# size - The shape of the returned array.                                       #
#################################################################################

x = random.uniform(size=(2, 3))

print("\nCreate a 2x3 uniform distribution sample: ")
print(x)


# Visualization of Uniform Distribution
sns.displot(random.uniform(size=1000), kind="kde")
plt.title("Uniform Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "uniform_distr.png"))
plt.close()


#################################################################################
# Logistic Distribution                                                         #
#################################################################################
# Logistic Distribution is used to describe growth.                             #
#                                                                               #
# It has three parameters:                                                      #
# loc - mean, where the peak is. Default 0.                                     #
# scale - standard deviation, the flatness of distribution. Default 1.          #
# size - The shape of the returned array.                                       #
#################################################################################


# Draw 2x3 samples from a logistic distribution with mean at 1 and stddev 2.0
x = random.logistic(loc=1, scale=2, size=(2, 3))

print("\nDraw 2x3 samples from a logistic distribution with mean at 1 and stddev 2.0: ")
print(x)


# Visualization of Logistic Distribution
sns.displot(random.logistic(size=1000), kind="kde")

plt.title("Visualization of Logistic Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "logistic_dist.png"))
plt.close()


# Difference Between Logistic and Normal Distribution
data = {
  "normal": random.normal(scale=2, size=1000),
  "logistic": random.logistic(size=1000)
}

sns.displot(data, kind="kde")
plt.title("Difference Between Logistic and Normal Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "dif_logistic-normal.png"))
plt.close()


#####################################################################################################
# Multinomial Distribution                                                                          #
#####################################################################################################
# It has three parameters:                                                                          #
# n - number of times to run the experiment.                                                        #
# pvals - list of probabilties of outcomes (e.g. [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] for dice roll).     #
# size - The shape of the returned array.                                                           #
#####################################################################################################

x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

print("\nMultinomial Distribution")
print(x)


#####################################################################################################
# Exponential Distribution                                                                          #
#####################################################################################################
# It has two parameters:                                                                            #
# scale - inverse of rate ( see lam in poisson distribution ) defaults to 1.0.                      #
# size - The shape of the returned array.                                                           #
#####################################################################################################


# Draw out a sample for exponential distribution with 2.0 scale with 2x3 size:
x = random.exponential(scale=2,size=(2,4))

print("\nDraw out a sample for exponential distribution with 2.0 scale with 2x3 size: ")
print(x)


# Visualization of Exponential Distribution
sns.displot(random.exponential(size=1000), kind="kde")
plt.title("Visualization of Exponential Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "exponential_dist.png"))
plt.close()


#####################################################################################################
# Chi Square Distribution                                                                           #
#####################################################################################################
# Chi Square distribution is used as a basis to verify the hypothesis.                              #
#                                                                                                   #
# It has two parameters:                                                                            #
# df - (degree of freedom).                                                                         #
# size - The shape of the returned array.                                                           #
#####################################################################################################

# Draw out a sample for chi squared distribution with degree of freedom 2 with size 2x3:
x = random.chisquare(df=2, size=(2, 3))

print("\nDraw out a sample for chi squared distribution with degree of freedom 2 with size 2x3:")
print(x)


# Visualization of Chi Square Distribution
sns.displot(random.chisquare(df=1, size=1000), kind="kde")
plt.title("Visualization of Chi Square Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "Chi_Square_dist.png"))
plt.close()


#####################################################################################################
# Rayleigh Distribution                                                                             #
#####################################################################################################
# Rayleigh distribution is used in signal processing.                                               #
#                                                                                                   #
# It has two parameters:                                                                            #
# scale - (standard deviation) decides how flat the distribution will be default 1.0).              #
# size - The shape of the returned array.                                                           #
#####################################################################################################


# Draw out a sample for rayleigh distribution with scale of 2 with size 2x3:
x = random.rayleigh(scale=2, size=(2, 3))

print("\nDraw out a sample for rayleigh distribution with scale of 2 with size 2x3: ")
print(x)


# Visualization of Rayleigh Distribution
print("Visualization of Rayleigh Distribution")

sns.displot(random.rayleigh(size=1000), kind="kde")
plt.title("Visualization of Rayleigh Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "rayleigh_dist.png"))
plt.close()


#####################################################################################################
# Pareto Distribution                                                                               #
#####################################################################################################
# It has two parameter:                                                                             #
# a - shape parameter.                                                                              #
# size - The shape of the returned array.                                                           #
#####################################################################################################


# Draw out a sample for pareto distribution with shape of 2 with size 2x3:
x = random.pareto(a=2, size=(2, 3))

print("\nDraw out a sample for pareto distribution with shape of 2 with size 2x3:")
print(x)


# Visualization of Pareto Distribution
str = "Visualization of Pareto Distribution"

print(str)

sns.displot(random.pareto(a=2, size=1000))
plt.title(str)
#plt.show()
plt.savefig(os.path.join(folder, "pareto.png"))
plt.close()


#####################################################################################################
# Zipf Distribution                                                                                 #
#####################################################################################################
# It has two parameters:                                                                            #
# a - distribution parameter.                                                                       #
# size - The shape of the returned array.                                                           #
#####################################################################################################


# Draw out a sample for zipf distribution with distribution parameter 2 with size 2x3:
x = random.zipf(a=2, size=(2, 3))

print("\nDraw out a sample for zipf distribution with distribution parameter 2 with size 2x3")
print(x)


# Visualization of Zipf Distribution
print("Visualization of Zipf Distribution")

x = random.zipf(a=2, size=1000)
sns.displot(x[x<10])
plt.title("Visualization of Zipf Distribution")
#plt.show()
plt.savefig(os.path.join(folder, "zipf_dist.png"))
plt.close()