import pandas as pd



print("Padnas Version: " + pd.__version__ +"\n\n")


mydataset = {
    'cars'      : ["Mercedes-Benz", "Land Rover", "Ford"],
    'passings'  : [4, 7, 2]
}

myvar = pd.DataFrame(mydataset)

print(myvar)
print("\n\n")


# Series
a = [1, 7, 2]

myvar = pd.Series(a)


print(f"Series of: {a} as pd.Series(a): ")
print(myvar)
print("\n\n")

print("Return the first value of the Series:")
print(myvar[0])
print("\n\n")

# Create my own labels
a = [1, 7, 2]
myvar = pd.Series(a, index = ["x", "y", "z"])

print("Create my own labels: ")
print(myvar)
print("\n")

print("Return the value of 'y':")
print(myvar['y'])
print("\n\n")



# Key/Value Objects as Series
calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories)

print("Key/Value Objects as Series:")
print(myvar)
print("\n\n")


data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

myvar = pd.DataFrame(data)

print("Create a DataFrame from two Series: ")
print(myvar)
print("\n")

print("Return row 1: ")
print(myvar.loc[1])
print("\n\n")