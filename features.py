import pandas as pd
import numpy as np
from preprocessing import glucose, time, interval
from scipy.integrate import trapezoid

def mean(df):
   return df[glucose()].mean()

def summary_stats(df):
   min = df[glucose()].min()
   first = df[glucose()].quantile(0.25)
   median = df[glucose()].median()
   third = df[glucose()].quantile(0.75)
   max = df[glucose()].max()

   return [min, first, median, third, max]

def std(df):
   return df[glucose()].std()

def a1c(df):
   return (46.7 + mean(df)) / 28.7

def gmi(df):
   return (0.02392 * mean(df)) + 3.31

"""
Returns the percent of total time the glucose levels were between the given lower and upper bounds (inclusive)
@param df: the data in the form of a Pandas DataFrame
@param low: the lower bound of the acceptable glucose values
@param high: the upper bound of the acceptable glucose values
"""
def percent_time_in_range(df, low=70, high=180):
   in_range_df = df[(df[glucose()] <= high) & (df[glucose()] >= low)]
   time_in_range = len(in_range_df)
   total_time = len(df)
   return (100 * time_in_range / total_time) if total_time > 0 else np.nan

# ------------------------- EVENT-BASED ----------------------------

def AUC(df):
   return trapezoid(df[glucose()],dx = interval())

def iAUC(df, level=70):
   data = df.copy()
   data[glucose()] = data[glucose()] - level
   data.loc[data[glucose()] < 0, glucose()] = 0
   return AUC(data)

def baseline(df):
   return df[time()].iloc[0]

def peak(df):
   return np.max(df[glucose()])

def delta(df):
   return peak(df) - baseline(df)

"""
Returns a Pandas Series containing the Timestamps of glucose excursions
"""
def excursions(df):
   sd = std(df)
   ave = mean(df)

   outlier_df = df[(df[glucose()] >= ave + (2 * sd)) | (df[glucose()] <= ave - (2 * sd))].copy()
   
   # calculate the differences between each of the timestamps
   outlier_df.reset_index(inplace=True)
   outlier_df['timedeltas'] = outlier_df[time()].diff()[1:]
   # find the gaps between the times
   gaps = outlier_df[outlier_df['timedeltas'] > pd.Timedelta(minutes=interval())][time()]

   print(gaps)

   excursions = []
   for i in range(len(gaps) - 1):
      copy = outlier_df[(outlier_df[time()] >= gaps.iloc[i]) & (outlier_df[time()] <= gaps.iloc[i+1])][[time(), glucose()]].copy()
      copy.set_index(time(), inplace=True)
      excursions.append(copy.idxmax())
   
   return pd.Series(excursions)

"""
Takes in a multiindexed Pandas DataFrame containing CGM data for multiple patients/datasets, and
returns a single indexed Pandas DataFrame containing summary metrics in the form of one row per patient/dataset
"""
def create_features(dataset, events=False):
   df = pd.DataFrame()

   for id, data in dataset.groupby('id'):
      features = {}
      summary = summary_stats(data)

      features['id'] = id
      
      features['mean'] = mean(data)
      features['min'] = summary[0]
      features['first quartile'] = summary[1]
      features['median'] = summary[2]
      features['third quartile'] = summary[3]
      features['max'] = summary[4]

      features['intrasd'] = std(data)
      features['intersd'] = std(dataset)

      features['a1c'] = a1c(data)
      features['gmi'] = gmi(data)
      features['percent time in range'] = percent_time_in_range(data)

      if events:
         features['AUC'] = AUC(data)
         features['iAUC'] = iAUC(data)
         features['baseline'] = baseline(data)
         features['peak'] = peak(data)
         features['delta'] = delta(data)

      
      df = pd.concat([df, pd.DataFrame.from_records([features])])

   df = df.set_index(['id'])

   return df