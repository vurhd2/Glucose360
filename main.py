import pandas as pd

def import_data(path, glucose_col="Glucose Value (mg/dL)"):
   global glucose 
   glucose = glucose_col

   df = pd.read_csv(path)
   df = df.dropna(subset=[glucose])
   df = df.replace("Low", 40)
   df = df.replace("High", 400)

   df[glucose] = pd.to_numeric(df[glucose])

   return df

def ave_glucose(df):
   return df[glucose].mean()

def summary(df):
   min = df[glucose].min()
   first = df[glucose].quantile(0.25)
   median = df[glucose].median()
   third = df[glucose].quantile(0.75)
   max = df[glucose].max()

   return [min, first, median, third, max]

def intersd(df):
   return df[glucose].std()

def a1c(df):
   return (46.7 + ave_glucose(df)) / 28.7

def gmi(df):
   return (0.02392 * ave_glucose(df)) + 3.31

def std(df):
   return df[glucose].std()

"""
Returns the approximate total amount of minutes the glucose levels were between the given lower and upper bounds (inclusive)
@param df: the data in the form of a Pandas DataFrame
@param low: the lower bound of the acceptable glucose values
@param high: the upper bound of the acceptable glucose values
"""
def time_in_range(df, low, high):
   temp_df = df[df[glucose] <= high and df[glucose] >= low]

def main():
   #df = import_data("datasets/Clarity_Export_00000_Sutherland_Eliza_2023-10-16_235810.csv")
   #df = import_data("datasets/Clarity_Export_00001_Fitzroy_Penelope_2023-10-16_235810.csv")
   #df = import_data("datasets/Clarity_Export_00002_Barrow_Nathaniel_2023-10-16_235810.csv")

   print("summary: " + str(summary(df)))

   print("mean: " + str(ave_glucose(df)))

   print("a1c: " + str(a1c(df)))

   print("gmi: " + str(gmi(df)))

   print("std: " + str(std(df)))

if __name__ == "__main__":
   main()


