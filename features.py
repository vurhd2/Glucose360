import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

def mean(df: pd.DataFrame) -> float:
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


"""
Returns a Pandas Series containing the Timestamps of glucose excursions
"""


def excursions(df: pd.DataFrame) -> pd.Series:
    sd = std(df)
    ave = mean(df)

    outlier_df = df[
        (df[GLUCOSE] >= ave + (2 * sd)) | (df[GLUCOSE] <= ave - (2 * sd))
    ].copy()

    # calculate the differences between each of the timestamps
    outlier_df.reset_index(inplace=True)
    outlier_df["timedeltas"] = outlier_df[TIME].diff()[1:]

    # find the gaps between the times
    gaps = outlier_df[outlier_df["timedeltas"] > pd.Timedelta(minutes=INTERVAL)][
        TIME
    ]

    # adding initial and final timestamps so excursions at the start/end are included
    initial = pd.Series(df[TIME].iloc[0] - pd.Timedelta(seconds=1))
    final = pd.Series(df[TIME].iloc[-1] + pd.Timedelta(seconds=1))
    gaps = pd.concat([initial, gaps, final])

    # getting the timestamp of the peak within each excursion
    excursions = []
    for i in range(len(gaps) - 1):
        copy = outlier_df[
            (outlier_df[TIME] >= gaps.iloc[i])
            & (outlier_df[TIME] < gaps.iloc[i + 1])
        ][[TIME, GLUCOSE]].copy()
        copy.set_index(TIME, inplace=True)
        if np.min(copy) > ave:
            # local max
            excursions.append(copy.idxmax())
        else:
            # local min
            excursions.append(copy.idxmin())

    return pd.Series(excursions)

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

def MAD(df: pd.DataFrame) -> float:
    return (df[GLUCOSE] - mean(df)).abs().mean()

# n is the gap in hours
# INTERVAL should be in minutes
def CONGA(df: pd.DataFrame, n: int = 24) -> float:
    period = n * (60 / INTERVAL)
    return np.std(df[GLUCOSE].diff(periods=period))

# lag is in days
def MODD(df: pd.DataFrame, lag: int = 1) -> float:
    period = lag * 24 * (60 / INTERVAL)
    return np.mean(np.abs(df[GLUCOSE].diff(periods=period)))

def mean_absolute_differences(df: pd.DataFrame) -> float:
    return np.mean(np.abs(df[GLUCOSE].diff()))

def median_absolute_deviation(df: pd.DataFrame) -> float:
    return np.median(np.abs(df[GLUCOSE] - np.mean(df[GLUCOSE])))

def MAG(df: pd.DataFrame) -> float:
    time_diff = (df[TIME].iloc[-1] - df[TIME].iloc[0]).total_seconds() / 3600
    return np.sum(df[GLUCOSE].diff().abs()) / time_diff

def MAGE(df: pd.DataFrame, short_ma: int = 9, long_ma: int = 32) -> float:
   averages = pd.DataFrame()
   averages[GLUCOSE] = df[GLUCOSE]

   # calculate rolling means, iglu does right align instead of center
   averages["MA_Short"] = averages[GLUCOSE].rolling(window=short_ma, min_periods=1).mean()
   averages["MA_Long"] = averages[GLUCOSE].rolling(window=long_ma, min_periods=1).mean()

   # fill in leading NaNs due to moving average calculation
   averages["MA_Short"].iloc[:short_ma] = averages["MA_Short"].iloc[short_ma]
   averages["MA_Long"].iloc[:long_ma] = averages["MA_Long"].iloc[long_ma]
   averages["DELTA_SL"] = averages["MA_Short"] - averages["MA_Long"]
   
   # get crossing points
   glu = lambda i: averages[GLUCOSE].iloc[i]
   crosses = pd.DataFrame.from_records([{"location": 0, "type": np.where(glu(0) > 0, "peak", "nadir")}])

   for index in range(1, averages.shape[0]):
      current_actual = glu(index)
      current_average = averages["DELTA_SL"].iloc[index]
      previous_actual = glu(index-1)
      previous_average = averages["DELTA_SL"].iloc[index-1]

      if (((not np.isnan(current_actual)) and (not np.isnan(previous_actual))) and 
          ((not np.isnan(current_average)) and (not np.isnan(previous_average)))):
         if current_average * previous_average < 0:
            type = np.where(current_average < previous_average, "nadir", "peak")
            crosses = pd.concat([crosses, pd.DataFrame.from_records([{"location": index, "type": type}])])     
      elif (not np.isnan(current_average) and (current_average * averages["DELTA_SL"].iloc[crosses["location"].iloc[-1]] < 0)): # VALIDATE THIS LATER
         prev_delta = averages["DELTA_SL"].iloc[crosses["location"].iloc[-1]]
         type = np.where(current_average < prev_delta, "nadir", "peak")
         crosses = pd.concat([crosses, pd.DataFrame.from_records([{"location": index, "type": type}])])

   crosses = pd.concat([crosses, pd.DataFrame.from_records([{"location": -1, "type": np.where(averages["DELTA_SL"].iloc[-1] > 0, "peak", "nadir")}])])     
   crosses.dropna(inplace=True)

   num_extrema = crosses.shape[0] -  1
   minmax = pd.Series(np.nan, index=range(0, num_extrema))
   indexes = pd.Series(np.nan, index=range(0, num_extrema))

   for index in range(num_extrema):
      s1 = np.where(index == 0, crosses["location"].iloc[index], indexes.iloc[index-1])
      s2 = crosses["location"].iloc[index+1]

      values = df[GLUCOSE].iloc[s1:s2].dropna()
      if crosses["type"].iloc[index] == "nadir":
         minmax.iloc[index] = np.min(values)
         indexes.iloc[index] = values.idxmin() + s1 - 1
      else:
         minmax.iloc[index] = np.max(values)
         indexes.iloc[index] = values.idxmax() + s1 - 1
         
   differences = np.transpose(minmax[:, np.newaxis] - minmax)
   sd = np.std(df[GLUCOSE].dropna())
   N = len(minmax)

   # MAGE+
   mage_plus_heights = pd.Series()
   mage_plus_tp_pairs = {}
   j = 0; prev_j = 0
   while j <= N:
      delta = differences[prev_j:j,j]
      max_v = np.max(delta)
      i = np.argmax(delta) + prev_j - 1

      if max_v >= sd:
         for k in range(j, N):
            if minmax[k] > minmax[j]:
               j = k
            if (differences[j, k] < (-1 * sd)) or (k == N):
               max_v = minmax[j] - minmax[i]
               mage_plus_heights = pd.concat([mage_plus_heights, max_v])
               mage_plus_tp_pairs[len(mage_plus_tp_pairs)] = [i, j]

               prev_j = k
               j = k
               break
      else:
         j += 1
   
   # MAGE-
   mage_minus_heights = pd.Series()
   mage_minus_tp_pairs = {}
   j = 0; prev_j = 0
   while j <= N:
      delta = differences[prev_j:j,j]
      min_v = np.min(delta)
      i = np.argmin(delta) + prev_j - 1

      if min_v <= (-1 * sd):
         for k in range(j, N):
            if minmax[k] < minmax[j]:
               j = k
            if (differences[j, k] > sd) or (k == N):
               min_v = minmax[j] - minmax[i]
               mage_minus_heights = pd.concat([mage_minus_heights, min_v])
               mage_minus_tp_pairs[len(mage_minus_tp_pairs)] = [i, j, k]

               prev_j = k
               j = k
               break
      else:
         j += 1

   plus_first = np.where(mage_plus_heights.size > 0 and ((mage_minus_heights.size == 0) or (mage_plus_tp_pairs[0][1] <= mage_minus_tp_pairs[0][0])), True, False)
   return np.where(plus_first, np.mean(mage_plus_heights), np.mean(mage_minus_heights.abs()))

# ------------------------- EVENT-BASED ----------------------------


def AUC(df: pd.DataFrame) -> float:
    return trapezoid(df[GLUCOSE], dx=INTERVAL)


def iAUC(df: pd.DataFrame, level: int = 70) -> float:
    data = df.copy()
    data[GLUCOSE] = data[GLUCOSE] - level
    data.loc[data[GLUCOSE] < 0, GLUCOSE] = 0
    return AUC(data)


def baseline(df: pd.DataFrame) -> float:
    return df[TIME].iloc[0]


def peak(df: pd.DataFrame) -> float:
    return np.max(df[GLUCOSE])


def delta(df: pd.DataFrame) -> float:
    return peak(df) - baseline(df)

"""
Takes in a multiindexed Pandas DataFrame containing CGM data for multiple patients/datasets, and
returns a single indexed Pandas DataFrame containing summary metrics in the form of one row per patient/dataset
"""
def create_features(dataset: pd.DataFrame, events: bool = False) -> pd.DataFrame:
    df = pd.DataFrame()

    for id, data in dataset.groupby("id"):
        features = {}
        summary = summary_stats(data)
        features["id"] = id

        features["mean"] = mean(data)
        features["min"] = summary[0]
        features["first quartile"] = summary[1]
        features["median"] = summary[2]
        features["third quartile"] = summary[3]
        features["max"] = summary[4]

        features["intrasd"] = std(data)
        features["intersd"] = std(dataset)

        features["mean absolute differences"] = mean_absolute_differences(data)
        features["median absolute deviation"] = median_absolute_deviation(data)

        features["a1c"] = a1c(data)
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
        features["MAD"] = MAD(data)
        features["MAG"] = MAG(data)
        features["MODD"] = MODD(data)
        #features["MAGE"] = MAGE(data)

        if events:
            features["AUC"] = AUC(data)
            features["iAUC"] = iAUC(data)
            features["baseline"] = baseline(data)
            features["peak"] = peak(data)
            features["delta"] = delta(data)

        df = pd.concat([df, pd.DataFrame.from_records([features])])

    df = df.set_index(["id"])

    return df
