import pandas as pd
import numpy as np
import configparser
from multiprocessing import Pool

config = configparser.ConfigParser()
config.read('config.ini')
ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']

"""
All of the metric-calculating functions are designed for DataFrames that contain only one patient's data.
For example, if 'df' is the outputted DataFrame from 'import_data()', 'LBGI(df)' would not be accurate.
Instead, do 'LBGI(df.loc[PATIENT_ID])'.
"""

def mean(df: pd.DataFrame) -> float:
   """
   Calculates the mean
   @param df   the Pandas DataFrame containing 
   """
   return df[GLUCOSE].mean()

def summary_stats(df: pd.DataFrame) -> list:
    min = df[GLUCOSE].min()
    first = df[GLUCOSE].quantile(0.25)
    median = df[GLUCOSE].median()
    third = df[GLUCOSE].quantile(0.75)
    max = df[GLUCOSE].max()

    return [min, first, median, third, max]

def std(df: pd.DataFrame) -> float:
    return df[GLUCOSE].std()

def a1c(df: pd.DataFrame) -> float:
    return (46.7 + mean(df)) / 28.7

def gmi(df: pd.DataFrame) -> float:
    return (0.02392 * mean(df)) + 3.31

"""
Returns the percent of total time the glucose levels were between the given lower and upper bounds (inclusive)
@param df: the data in the form of a Pandas DataFrame
@param low: the lower bound of the acceptable glucose values
@param high: the upper bound of the acceptable glucose values
"""
def percent_time_in_range(df: pd.DataFrame, low: int = 70, high: int = 180) -> float:
    in_range_df = df[(df[GLUCOSE] <= high) & (df[GLUCOSE] >= low)]
    time_in_range = len(in_range_df)
    total_time = len(df)
    return (100 * time_in_range / total_time) if total_time > 0 else np.nan

def ADRR(df: pd.DataFrame) -> float:
   data = df.copy()

   # Convert time to date
   data['date'] = pd.to_datetime(data[TIME]).dt.date

   data = data.dropna(subset=[GLUCOSE])

   data['bgi'] = (np.log(data[GLUCOSE]) ** 1.084) - 5.381
   data['right'] = 22.7 * np.maximum(data['bgi'], 0) ** 2
   data['left'] = 22.7 * np.minimum(data['bgi'], 0) ** 2

   adrr = data.groupby(['date']).apply(lambda df: np.max(df['left']) + np.max(df['right'])).mean()
   return adrr

def BG_formula(ser: pd.Series) -> pd.Series:
    return 1.509 * (np.power(np.log(ser), 1.084) - 5.381)

def LBGI(df: pd.DataFrame) -> float:
    BG = np.minimum(0, BG_formula(df[GLUCOSE]))
    return np.mean(10 * (BG ** 2))

def HBGI(df: pd.DataFrame) -> float:
    BG = np.maximum(0, BG_formula(df[GLUCOSE]))
    return np.mean(10 * (BG ** 2))

def COGI(df: pd.DataFrame) -> float:
    tir = percent_time_in_range(df)
    tir_score = 0.5 * tir

    tbr = percent_time_in_range(df, 0, 70)
    tbr_score = 0.35 * ((1 - (np.minimum(tbr, 15) / 15)) * 100)

    sd = std(df)
    sd_score = 100
    if sd >= 108:
        sd_score = 0
    elif sd > 18:
        sd_score = (1 - (sd / 108)) * 100
    sd_score *= 0.15
    
    COGI = tir_score + tbr_score + sd_score
    return COGI

def GRADE_formula(df: pd.DataFrame) -> pd.DataFrame:
    df_GRADE = pd.DataFrame()
    df_GRADE[GLUCOSE] = df[GLUCOSE].copy()
    df_GRADE["GRADE"] = ((np.log10(np.log10(df[GLUCOSE] / 18)) + 0.16) ** 2) * 425
    return df_GRADE

def GRADE_eugly(df: pd.DataFrame) -> float:
    df_GRADE = GRADE_formula(df)
    return np.sum(df_GRADE[(df_GRADE[GLUCOSE] >= 70) & (df_GRADE[GLUCOSE] <= 140)]["GRADE"]) / np.sum(df_GRADE["GRADE"]) * 100

def GRADE_hypo(df: pd.DataFrame) -> float:
    df_GRADE = GRADE_formula(df)
    return np.sum(df_GRADE[df_GRADE[GLUCOSE] < 70]["GRADE"]) / np.sum(df_GRADE["GRADE"]) * 100

def GRADE_hyper(df: pd.DataFrame) -> float:
    df_GRADE = GRADE_formula(df)
    return np.sum(df_GRADE[df_GRADE[GLUCOSE] > 140]["GRADE"]) / np.sum(df_GRADE["GRADE"]) * 100

def GRADE(df: pd.DataFrame) -> float:
    df_GRADE = GRADE_formula(df)
    return df_GRADE["GRADE"].mean()

def GRI(df: pd.DataFrame) -> float:
    vlow = percent_time_in_range(df, 0, 53)
    low = percent_time_in_range(df, 54, 69)
    high = percent_time_in_range(df, 181, 250)
    vhigh = percent_time_in_range(df, 251, 500) 

    return min((3 * vlow) + (2.4 * low) + (0.8 * high) + (1.6 * vhigh), 100)

def GVP(df: pd.DataFrame) -> float:
    delta_x = df[TIME].diff().apply(lambda timedelta: timedelta.total_seconds() / 60)
    delta_y = df[GLUCOSE].diff()
    L = np.sum(np.sqrt((delta_x ** 2) + (delta_y ** 2)))
    L_0 = np.sum(delta_x)
    return ((L / L_0) - 1) * 100

def hyper_index(df: pd.DataFrame, limit: int = 140, a: float = 1.1, c: float = 30) -> float:
    BG = df[GLUCOSE].dropna()
    return np.sum(np.power(BG[BG > limit] - limit, a)) / (BG.size * c)

def hypo_index(df: pd.DataFrame, limit: int = 80, b: float = 2, d: float = 30) -> float:
    BG = df[GLUCOSE].dropna()
    return np.sum(np.power(limit - BG[BG < limit], b)) / (BG.size * d)

def IGC(df: pd.DataFrame) -> float:
    return hyper_index(df) + hypo_index(df)

def j_index(df: pd.DataFrame) -> float:
    return 0.001 * ((mean(df) + std(df)) ** 2)

# n is the gap in hours
# the resampling interval should be in minutes
def CONGA(df: pd.DataFrame, n: int = 24) -> float:
    config.read('config.ini')
    interval = int(config["variables"]["interval"])
    period = n * (60 / interval)
    return np.std(df[GLUCOSE].diff(periods=period))

# lag is in days
def MODD(df: pd.DataFrame, lag: int = 1) -> float:
    config.read('config.ini')
    interval = int(config["variables"]["interval"])
    period = lag * 24 * (60 / interval)
    return np.mean(np.abs(df[GLUCOSE].diff(periods=period)))

def mean_absolute_differences(df: pd.DataFrame) -> float:
    return np.mean(np.abs(df[GLUCOSE].diff()))

def median_absolute_deviation(df: pd.DataFrame, constant: float = 1.4826) -> float:
    return constant * np.nanmedian(np.abs(df[GLUCOSE] - np.median(df[GLUCOSE])))

def MAG(df: pd.DataFrame) -> float:
   df.dropna(subset=[GLUCOSE], inplace=True)
   data = df[(df[TIME].dt.minute == (df[TIME].dt.minute).iloc[0]) & (df[TIME].dt.second == (df[TIME].dt.second).iloc[0])][GLUCOSE]
   return np.sum(data.diff().abs()) / data.size

def MAGE(df: pd.DataFrame, short_ma: int = 5, long_ma: int = 32, max_gap: int = 180) -> float:
   data = df.reset_index(drop=True)

   config.read('config.ini')
   interval = int(config["variables"]["interval"])
   print(interval)

   missing = data[GLUCOSE].isnull()
   # create groups of consecutive missing values
   groups = missing.ne(missing.shift()).cumsum()
   # group by the created groups and count the size of each group, then apply it where values are missing
   size_of_groups = data.groupby([groups, missing])[GLUCOSE].transform('size').where(missing, 0)
   # filter groups where size is greater than 0 and take their indexes
   indexes = size_of_groups[size_of_groups.diff() > (max_gap / interval)].index.tolist()

   if not indexes: # no gaps in data larger than max_gap
      return MAGE_helper(df, short_ma, long_ma)
   else: # calculate MAGE per segment and add them together (weighted)
      indexes.insert(0, 0); indexes.append(None)
      mage = 0
      total_duration = 0
      for i in range(len(indexes) - 1):
         segment = data.iloc[indexes[i]:indexes[i+1]]
         segment = segment.loc[segment[GLUCOSE].first_valid_index():].reset_index(drop=True)
         segment_duration = (segment.iloc[-1][TIME] - segment.iloc[0][TIME]).total_seconds(); total_duration += segment_duration
         mage +=  segment_duration * MAGE_helper(segment, short_ma, long_ma)
      return mage / total_duration

def MAGE_helper(df: pd.DataFrame, short_ma: int = 5, long_ma: int = 32) -> float:
   averages = pd.DataFrame()
   averages[GLUCOSE] = df[GLUCOSE]
   averages.reset_index(drop=True, inplace=True)

   # calculate rolling means, iglu does right align instead of center
   averages["MA_Short"] = averages[GLUCOSE].rolling(window=short_ma, min_periods=1).mean()
   averages["MA_Long"] = averages[GLUCOSE].rolling(window=long_ma, min_periods=1).mean()

   # fill in leading NaNs due to moving average calculation
   averages["MA_Short"].iloc[:short_ma-1] = averages["MA_Short"].iloc[short_ma-1]
   averages["MA_Long"].iloc[:long_ma-1] = averages["MA_Long"].iloc[long_ma-1]
   averages["DELTA_SL"] = averages["MA_Short"] - averages["MA_Long"]
   
   # get crossing points
   glu = lambda i: averages[GLUCOSE].iloc[i]
   average = lambda i: averages["DELTA_SL"].iloc[i]
   crosses_list = [{"location": 0, "type": np.where(average(0) > 0, "peak", "nadir")}]

   for index in range(1, averages.shape[0]):
      current_actual = glu(index)
      current_average = average(index)
      previous_actual = glu(index-1)
      previous_average = average(index-1)

      if not (np.isnan(current_actual) or np.isnan(previous_actual) or np.isnan(current_average) or np.isnan(previous_average)):
         if current_average * previous_average < 0:
            type = np.where(current_average < previous_average, "nadir", "peak")
            crosses_list.append({"location": index, "type": type})   
         elif (not np.isnan(current_average) and (current_average * average(crosses_list[-1]["location"]) < 0)):
            prev_delta = average(crosses_list[-1]["location"])
            type = np.where(current_average < prev_delta, "nadir", "peak")
            crosses_list.append({"location": index, "type": type})

   crosses_list.append({"location": None, "type": np.where(average(-1) > 0, "peak", "nadir")})
   crosses = pd.DataFrame(crosses_list)     

   num_extrema = crosses.shape[0] -  1
   minmax = np.tile(np.nan, num_extrema)
   indexes = pd.Series(np.nan, index=range(num_extrema))

   for index in range(num_extrema):
      s1 = int(np.where(index == 0, crosses["location"].iloc[index], indexes.iloc[index-1]))
      s2 = crosses["location"].iloc[index+1]

      values = averages[GLUCOSE].loc[s1:s2]
      if crosses["type"].iloc[index] == "nadir":
         minmax[index] = np.min(values)
         indexes.iloc[index] = values.idxmin() 
      else:
         minmax[index] = np.max(values)
         indexes.iloc[index] = values.idxmax() 
   
   differences = np.transpose(minmax[:, np.newaxis] - minmax)
   sd = np.std(df[GLUCOSE].dropna())
   N = len(minmax)

   # MAGE+
   mage_plus_heights = []
   mage_plus_tp_pairs = []
   j = 0; prev_j = 0
   while j < N:
      delta = differences[prev_j:j+1,j]

      max_v = np.max(delta)
      i = np.argmax(delta) + prev_j

      if max_v >= sd:
         for k in range(j, N):
            if minmax[k] > minmax[j]:
               j = k
            if (differences[j, k] < (-1 * sd)) or (k == N - 1):
               max_v = minmax[j] - minmax[i]
               mage_plus_heights.append(max_v)
               mage_plus_tp_pairs.append([i, j])

               prev_j = k
               j = k
               break
      else:
         j += 1
   
   # MAGE-
   mage_minus_heights = []
   mage_minus_tp_pairs = [] 
   j = 0; prev_j = 0
   while j < N:
      delta = differences[prev_j:j+1,j]
      min_v = np.min(delta) 
      i = np.argmin(delta) + prev_j

      if min_v <= (-1 * sd):
         for k in range(j, N):
            if minmax[k] < minmax[j]:
               j = k
            if (differences[j, k] > sd) or (k == N - 1):
               min_v = minmax[j] - minmax[i]
               mage_minus_heights.append(min_v)
               mage_minus_tp_pairs.append([i, j, k])

               prev_j = k
               j = k
               break
      else:
         j += 1

   plus_first = len(mage_plus_heights) > 0 and ((len(mage_minus_heights) == 0) or (mage_plus_tp_pairs[0][1] <= mage_minus_tp_pairs[0][0]))
   return float(np.where(plus_first, np.mean(mage_plus_heights), np.mean(np.absolute(mage_minus_heights))))

def ROC(df: pd.DataFrame, timedelta: int = 15) -> pd.DataFrame:
   config.read('config.ini')
   interval = int(config["variables"]["interval"])
   if timedelta < interval:
      raise Exception("Given timedelta must be greater than resampling interval.")

   positiondelta = round(timedelta / interval)
   return df[GLUCOSE].diff(periods=positiondelta) / timedelta

def compute_features(id, data):
   features = {}
   summary = summary_stats(data)
   features[ID] = id

   features["mean"] = mean(data)
   features["min"] = summary[0]
   features["first quartile"] = summary[1]
   features["median"] = summary[2]
   features["third quartile"] = summary[3]
   features["max"] = summary[4]

   features["intrasd"] = std(data)
   #features["intersd"] = std(dataset)

   features["mean absolute differences"] = mean_absolute_differences(data)
   features["median absolute deviation"] = median_absolute_deviation(data)

   features["eA1C"] = a1c(data)
   features["gmi"] = gmi(data)
   features["percent time in range"] = percent_time_in_range(data)
   features["ADRR"] = ADRR(data)
   features["LBGI"] = LBGI(data)
   features["HBGI"] = HBGI(data)
   features["COGI"] = COGI(data)

   features["euglycaemic GRADE"] = GRADE_eugly(data)
   features["hyperglycaemic GRADE"] = GRADE_hyper(data)
   features["hypoglycaemic GRADE"] = GRADE_hypo(data)
   features["GRADE"] = GRADE(data)
   features["GRI"] = GRI(data)

   features["hyperglycemia index"] = hyper_index(data)
   features["hypoglycemia index"] = hypo_index(data)
   features["IGC"] = IGC(data)

   features["GVP"] = GVP(data)
   features["j-index"] = j_index(data)

   features["CONGA"] = CONGA(data)
   features["MAG"] = MAG(data)
   features["MODD"] = MODD(data)
   features["MAGE"] = MAGE(data)

   return features


def create_features(dataset: pd.DataFrame) -> pd.DataFrame:
   """
   Takes in a multiindexed Pandas DataFrame containing CGM data for multiple patients/datasets, and
   returns a single indexed Pandas DataFrame containing summary metrics in the form of one row per patient/dataset
   """
   with Pool() as pool:
      features = pool.starmap(compute_features, dataset.groupby(ID))
   features = pd.DataFrame(features).set_index([ID])
   return features
