# Dane CSV:
# https://dane.gov.pl/pl/dataset/3357,dane-z-mandatow-karnych/resource/72050/table


import pandas as pd


# Zmienne:
file_CSV = 'data.csv'
file_JSON = 'file.json'

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


# OBSŁUGA DANYCH
# Dane CSV  i JSON:     https://dane.gov.pl/pl/dataset/3357,dane-z-mandatow-karnych/resource/72050/table

df = pd.read_csv(file_CSV)

print(f"Obsługa danych z pliku: {file_CSV}")
print(df)
#print(df.to_string())
print("\n\n")


print("Check the number of maximum returned rows:")
print(pd.options.display.max_rows)
print("\n\n")



# JSON

df = pd.read_json(file_JSON)

print("File JSON: ")
print(df.to_string())
print("\n\n")



df = pd.read_csv(file_CSV)

print("Get a quick overview by printing the first 10 rows of the DataFrame: ")
print(df.head(10))
print("\n\n")

print("Print the first 5 rows of the DataFrame: ")
print(df.head())
print("\n\n")


print("Print the last 5 rows of the DataFrame: ")
print(df.tail())
print("\n\n")


print("Print information about the data: ")
print(df.info())
print("\n\n")



# Cleaning Data
df = pd.read_csv(file_CSV)

new_df = df.dropna()

print("Cleaning Empty Cells")
print("Return a new Data Frame with no empty cells: ")
print(new_df.to_string())
print("\n\n")


df.dropna(inplace = True)
print("Remove all rows with NULL values: ")
print(new_df.to_string())
print("\n\n")


print("Replace NULL values with the number 130.")
df = pd.read_csv(file_CSV)
df.fillna(130, inplace = True)
print("\n")


print("Replace NULL values in the 'Calories' columns with the number 130.")
df = pd.read_csv(file_CSV)
df.fillna({"Calories": 130}, inplace=True)
print("\n")


print("Calculate the MEAN, and replace any empty values with it.")
df = pd.read_csv(file_CSV)
x = df["Calories"].mean()
df.fillna({"Calories": x}, inplace=True)
print("\n")


print("Calculate the MEDIAN, and replace any empty values with it.")
df = pd.read_csv(file_CSV)
x = df["Calories"].median()
df.fillna({"Calories": x}, inplace=True)
print("\n")


print("Calculate the MODE, and replace any empty values with it.")
df = pd.read_csv(file_CSV)
x = df["Calories"].mode()[0]
df.fillna({"Calories": x}, inplace=True)
print("\n")


# Cleaning Data of Wrong Format
# https://www.w3scools.com/python/pandas/pandas_cleaning_wrong_format.asp

df = pd.read_csv(file_CSV)

df['Date'] = pd.to_datetime(df['Date'], format='mixed')

print("Convert to date:")
print(df.to_string())
print("\n\n")

print("Remove rows with a NULL value in the 'Date' column:")
df.dropna(subset=['Date'], inplace = True)
print("\n\n")


# https://www.w3schools.com/python/pandas/pandas_cleaning_wrong_data.asp