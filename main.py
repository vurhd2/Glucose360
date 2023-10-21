import pandas as pd
import glob

"""
Returns a Multiindexed Pandas DataFrame containing all of the csv data found in the directory at the given path
@param path    the path of the directory to be parsed through
@param glucose_col   the header of the column containing the glucose values
"""
def import_directory(path, glucose_col="Glucose Value (mg/dL)"):
   global glucose
   glucose = glucose_col

   csv_files = glob.glob(path + "/*.csv")

   data = pd.DataFrame()
   for file in csv_files:
      df = import_data(file)

      data = pd.concat([data, df])

   data = data.set_index(['id', 'Timestamp (YYYY-MM-DDThh:mm:ss)'])

   return data

"""
Returns a pre-processed Pandas DataFrame containing the data for the csv file at the given path
@param path    the path of the csv file to be pre-processed and read into a Pandas Dataframe
"""
def import_data(path):
   df = pd.read_csv(path)

   id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1] + df["Patient Info"].iloc[2]
   df['id'] = id

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

   df = import_directory("datasets")

   for id, data in df.groupby(level=0):
      print("ID: " + str(id))
      print("summary: " + str(summary(data)))
      print("mean: " + str(ave_glucose(data)))
      print("a1c: " + str(a1c(data)))
      print("gmi: " + str(gmi(data)))
      print("std: " + str(std(data)))

if __name__ == "__main__":
   main()


