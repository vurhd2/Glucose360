import pandas as pd
import numpy as np
import configparser
from multiprocessing import Pool
import os
from scipy.integrate import trapezoid

dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, "config.ini")
config = configparser.ConfigParser()
config.read(config_path)
ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']

"""
All of the metric-calculating functions are designed for DataFrames that contain only one patient's data.
For example, if 'df' is the outputted DataFrame from 'import_data()', 'LBGI(df)' would not be accurate.
Instead, do 'LBGI(df.loc[PATIENT_ID])'.
"""

def mean(df: pd.DataFrame) -> float:
   """Calculates the mean glucose level for the given CGM trace

   :param df: a Pandas DataFrame containing the preprocessed CGM data to calculate the mean glucose for
   :type df: 'pandas.DataFrame'
   :return: the mean glucose level of the given CGM trace
   :rtype: float
   """
   return df[GLUCOSE].mean()

def summary_stats(df: pd.DataFrame) -> list[float]:
    """Calculates summary statistics (minimum, first quartile, median, third quartile, and maximum) for the given CGM trace

    :param df: a Pandas DataFrame containing the preprocessed CGM data to calculate the five-point summary for
    :type df: 'pandas.DataFrame'
    :return: a list containing the five-point summary for the given CGM trace
    :rtype: list[float]
    """
    min = df[GLUCOSE].min()
    first = df[GLUCOSE].quantile(0.25)
    median = df[GLUCOSE].median()
    third = df[GLUCOSE].quantile(0.75)
    max = df[GLUCOSE].max()

    return [min, first, median, third, max]

def SD(df: pd.DataFrame) -> float:
    """Calculates the standard deviation for the given CGM trace

    :param df: a Pandas DataFrame containing the preprocessed CGM data to calculate the standard deviation for
    :type df: 'pandas.DataFrame'
    :return: the standard deviation of the given CGM trace
    :rtype: float
    """
    return df[GLUCOSE].std()

def CV(df: pd.DataFrame) -> float:
   """Calculates the coefficient of variation (CV) for the given CGM trace

   :param df: a Pandas DataFrame containing the preprocessed CGM data to calculate the CV for
   :type df: 'pandas.DataFrame'
   :return: the CV of the given CGM trace
   :rtype: float
   """
   return SD(df) / mean(df) * 100

def eA1c(df: pd.DataFrame) -> float:
    """Calculates the estimated A1c (eA1c) for the given CGM trace

    :param df: a Pandas DataFrame containing the preprocessed CGM data to calculate the eA1c for
    :type df: 'pandas.DataFrame'
    :return: the eA1c of the given CGM trace
    :rtype: float
    """
    return (46.7 + mean(df)) / 28.7

def GMI(df: pd.DataFrame) -> float:
    """Calculates the Glucose Management Indicator (GMI) for the given CGM trace

    :param df: a Pandas DataFrame containing the preprocessed CGM data to calculate the GMI for
    :type df: 'pandas.DataFrame'
    :return: the GMI of the given CGM trace
    :rtype: float
    """
    return (0.02392 * mean(df)) + 3.31

def percent_time_in_range(df: pd.DataFrame, low: int = 70, high: int = 180) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were between the given lower and upper bounds (inclusive)
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :param low: the lower bound of the acceptable glucose values, defaults to 70
   :type low: int, optional
   :param high: the upper bound of the acceptable glucose values, defaults to 180
   :type high: int, optional
   :return: the percentage of total time the glucose levels within the given CGM trace were between the given bounds (inclusive)
   :rtype: float
   """
   valid_df = df.dropna(subset=[GLUCOSE])
   in_range_df = valid_df[(valid_df[GLUCOSE] <= high) & (valid_df[GLUCOSE] >= low)]
   time_in_range = len(in_range_df)
   total_time = len(valid_df)
   return (100 * time_in_range / total_time) if total_time > 0 else np.nan

def percent_time_in_tight_range(df: pd.DataFrame):
   """Returns the percent of total time the given CGM trace's glucose levels were within 70-140 mg/dL (inclusive)

   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were within 70-140 mg/dL (inclusive)
   :rtype: float
   """
   return percent_time_in_range(df, low = 70, high = 140)

def percent_time_above_range(df: pd.DataFrame, limit: int = 180) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were above a given threshold (inclusive)
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :param limit: the threshold for calculating the percent time above, defaults to 180
   :type limit: int, optional
   :return: the percentage of total time the glucose levels within the given CGM trace were above the given threshold (inclusive)
   :rtype: float
   """
   return percent_time_in_range(df, low = limit, high = 400)

def percent_time_below_range(df: pd.DataFrame, limit: int = 70) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were below a given threshold (inclusive)
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :param limit: the threshold for calculating the percent time below, defaults to 70
   :type limit: int, optional
   :return: the percentage of total time the glucose levels within the given CGM trace were below the given threshold (inclusive)
   :rtype: float
   """
   return percent_time_in_range(df, low = 40, high = limit)

def percent_time_in_hypoglycemia(df: pd.DataFrame) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were within literature-defined
   ranges that indicate hypoglycemia
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were in ranges indicating hypoglycemia (< 70 mg/dL)
   :rtype: float
   """
   return percent_time_below_range(df, 70)

def percent_time_in_level_1_hypoglycemia(df: pd.DataFrame) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were within literature-defined
   ranges that indicate level 1 hypoglycemia
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were in ranges indicating level 1 hypoglycemia (54-70 mg/dL)
   :rtype: float
   """
   return percent_time_in_range(df, 54, 69)

def percent_time_in_level_2_hypoglycemia(df: pd.DataFrame) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were within literature-defined
   ranges that indicate level 2 hypoglycemia
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were in ranges indicating level 2 hypoglycemia (< 54 mg/dL)
   :rtype: float
   """
   return percent_time_below_range(df, 53)

def percent_time_in_hyperglycemia(df: pd.DataFrame) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were within literature-defined
   ranges that indicate hyperglycemia
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were in ranges indicating hyperglycemia (> 180 mg/dL)
   :rtype: float
   """
   return percent_time_above_range(df, 180)

def percent_time_in_level_0_hyperglycemia(df: pd.DataFrame) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were within 
   ranges that indicate level 0 hyperglycemia
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were in ranges indicating level 0 hyperglycemia (140-180 mg/dL)
   :rtype: float
   """
   return percent_time_in_range(df, 140, 180)

def percent_time_in_level_1_hyperglycemia(df: pd.DataFrame) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were within literature-defined
   ranges that indicate level 1 hyperglycemia
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were in ranges indicating level 1 hyperglycemia (180-250 mg/dL)
   :rtype: float
   """
   return percent_time_in_range(df, 181, 249)

def percent_time_in_level_2_hyperglycemia(df: pd.DataFrame) -> float:
   """Returns the percent of total time the given CGM trace's glucose levels were within literature-defined
   ranges that indicate level 2 hyperglycemia
   
   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the percentage of total time the glucose levels within the given CGM trace were in ranges indicating level 2 hyperglycemia (> 250 mg/dL)
   :rtype: float
   """
   return percent_time_above_range(df, 250)

def ADRR(df: pd.DataFrame) -> float:
   """Calculates the Average Daily Risk Range (ADRR) for the given CGM trace.

   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :return: the ADRR for the given CGM trace
   :rtype: float
   """
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
    """Calculates the Average Daily Risk Range (ADRR) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the ADRR for the given CGM trace
    :rtype: float
    """
    return 1.509 * (np.power(np.log(ser), 1.084) - 5.381)

def LBGI(df: pd.DataFrame) -> float:
    """Calculates the Low Blood Glucose Index (LBGI) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the LBGI for the given CGM trace
    :rtype: float
    """
    BG = np.minimum(0, BG_formula(df[GLUCOSE]))
    return np.mean(10 * (BG ** 2))

def HBGI(df: pd.DataFrame) -> float:
    """Calculates the High Blood Glucose Index (HBGI) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the HBGI for the given CGM trace
    :rtype: float
    """
    BG = np.maximum(0, BG_formula(df[GLUCOSE]))
    return np.mean(10 * (BG ** 2))

def COGI(df: pd.DataFrame) -> float:
    """Calculates the Continuous Glucose Monitoring Index (COGI) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the COGI for the given CGM trace
    :rtype: float
    """
    tir = percent_time_in_range(df)
    tir_score = 0.5 * tir

    tbr = percent_time_in_range(df, 0, 70)
    tbr_score = 0.35 * ((1 - (np.minimum(tbr, 15) / 15)) * 100)

    sd = SD(df)
    sd_score = 100
    if sd >= 108:
        sd_score = 0
    elif sd > 18:
        sd_score = (1 - ((sd-18) / 90)) * 100
    sd_score *= 0.15
    
    COGI = tir_score + tbr_score + sd_score
    return COGI

def GRADE_formula(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms each glucose value within the given CGM trace as needed to help calculate 
    the Glycaemic Risk Assessment Diabetes Equation (GRADE).

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the LBGI for the given CGM trace
    :rtype: float
    """
    df_GRADE = pd.DataFrame()
    df_GRADE[GLUCOSE] = df[GLUCOSE].copy()
    df_GRADE["GRADE"] = ((np.log10(np.log10(df[GLUCOSE] / 18)) + 0.16) ** 2) * 425
    return df_GRADE

def GRADE_eugly(df: pd.DataFrame) -> float:
    """Calculates the Glycaemic Risk Assessment Diabetes Equation (GRADE) for 
    solely the glucose values in target range (70-140 mg/dL) within the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the euglycemic GRADE for the given CGM trace
    :rtype: float
    """
    df_GRADE = GRADE_formula(df)
    return np.sum(df_GRADE[(df_GRADE[GLUCOSE] >= 70) & (df_GRADE[GLUCOSE] <= 140)]["GRADE"]) / np.sum(df_GRADE["GRADE"]) * 100

def GRADE_hypo(df: pd.DataFrame) -> float:
    """Calculates the Glycaemic Risk Assessment Diabetes Equation (GRADE) for 
    solely the glucose values in hypoglycemic range (<70 mg/dL) within the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the hypoglycemic GRADE for the given CGM trace
    :rtype: float
    """
    df_GRADE = GRADE_formula(df)
    return np.sum(df_GRADE[df_GRADE[GLUCOSE] < 70]["GRADE"]) / np.sum(df_GRADE["GRADE"]) * 100

def GRADE_hyper(df: pd.DataFrame) -> float:
    """Calculates the Glycaemic Risk Assessment Diabetes Equation (GRADE) for 
    solely the glucose values in hyperglycemic range (>140 mg/dL) within the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the hyperglycemic GRADE for the given CGM trace
    :rtype: float
    """
    df_GRADE = GRADE_formula(df)
    return np.sum(df_GRADE[df_GRADE[GLUCOSE] > 140]["GRADE"]) / np.sum(df_GRADE["GRADE"]) * 100

def GRADE(df: pd.DataFrame) -> float:
    """Calculates the Glycaemic Risk Assessment Diabetes Equation (GRADE) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the GRADE for the given CGM trace
    :rtype: float
    """
    df_GRADE = GRADE_formula(df)
    return df_GRADE["GRADE"].mean()

def GRI(df: pd.DataFrame) -> float:
    """Calculates the Glycemia Risk Index (GRI) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the GRI for the given CGM trace
    :rtype: float
    """
    vlow = percent_time_in_range(df, 0, 53)
    low = percent_time_in_range(df, 54, 69)
    high = percent_time_in_range(df, 181, 250)
    vhigh = percent_time_in_range(df, 251, 500) 

    return min((3 * vlow) + (2.4 * low) + (0.8 * high) + (1.6 * vhigh), 100)

def GVP(df: pd.DataFrame) -> float:
    """Calculates the Glucose Variability Percentage (GVP) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the GVP for the given CGM trace
    :rtype: float
    """
    copy_df = df.dropna(subset=["Glucose"])
    delta_x = pd.Series(5, index=np.arange(copy_df.shape[0]), dtype="float", name='orders')
    delta_y = copy_df.reset_index()["Glucose"].diff()
    L = np.sum(np.sqrt((delta_x ** 2) + (delta_y ** 2)))
    L_0 = np.sum(delta_x)
    return L / L_0

def hyper_index(df: pd.DataFrame, limit: int = 140, a: float = 1.1, c: float = 30) -> float:
    """Calculates the Hyperglycemia Index for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :param limit: upper limit of target range (above which would hyperglycemia), defaults to 140 mg/dL
    :type limit: int, optional
    :param a: exponent utilized for Hyperglycemia Index calculation, defaults to 1.1
    :type a: float, optional
    :param c: constant to help scale Hyperglycemia Index the same as other metrics (e.g. LBGI, HBGI, and GRADE), defaults to 30
    :type c: float, optional
    :return: the Hyperglycemia Index for the given CGM trace
    :rtype: float
    """
    BG = df[GLUCOSE].dropna()
    return np.sum(np.power(BG[BG > limit] - limit, a)) / (BG.size * c)

def hypo_index(df: pd.DataFrame, limit: int = 80, b: float = 2, d: float = 30) -> float:
    """Calculates the Hypoglycemia Index for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :param limit: lower limit of target range (above which would hypoglycemia), defaults to 80 mg/dL
    :type limit: int, optional
    :param b: exponent utilized for Hypoglycemia Index calculation, defaults to 2
    :type b: float, optional
    :param d: constant to help scale Hypoglycemia Index the same as other metrics (e.g. LBGI, HBGI, and GRADE), defaults to 30
    :type d: float, optional
    :return: the Hypoglycemia Index for the given CGM trace
    :rtype: float
    """
    BG = df[GLUCOSE].dropna()
    return np.sum(np.power(limit - BG[BG < limit], b)) / (BG.size * d)

def IGC(df: pd.DataFrame) -> float:
    """Calculates the Index of Glycemic Control (IGC) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the IGC for the given CGM trace
    :rtype: float
    """
    return hyper_index(df) + hypo_index(df)

def j_index(df: pd.DataFrame) -> float:
    """Calculates the J-Index for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the J-Index for the given CGM trace
    :rtype: float
    """
    return 0.001 * ((mean(df) + SD(df)) ** 2)

def CONGA(df: pd.DataFrame, n: int = 24) -> float:
    """Calculates the Continuous Overall Net Glycemic Action (CONGA) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :param n: the difference in time (in hours) between observations used to calculate CONGA, defaults to 24
    :type n: int, optional
    :return: the CONGA for the given CGM trace
    :rtype: float
    """
    config.read('config.ini')
    interval = int(config["variables"]["interval"])
    period = n * (60 / interval)
    return np.std(df[GLUCOSE].diff(periods=period))

# lag is in days
def MODD(df: pd.DataFrame, lag: int = 1) -> float:
    """Calculates the Mean Difference Between Glucose Values Obtained at the Same Time of Day (MODD) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :param lag: the difference in time (in days) between observations used to calculate MODD, defaults to 1
    :type lag: int, optional
    :return: the MODD for the given CGM trace
    :rtype: float
    """
    config.read('config.ini')
    interval = int(config["variables"]["interval"])
    period = lag * 24 * (60 / interval)
    
    return np.mean(np.abs(df[GLUCOSE].diff(periods=period)))

def mean_absolute_differences(df: pd.DataFrame) -> float:
    """Calculates the Mean Absolute Differences for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the mean absolute differences for the given CGM trace
    :rtype: float
    """
    return np.mean(np.abs(df[GLUCOSE].diff()))

def median_absolute_deviation(df: pd.DataFrame, constant: float = 1.4826) -> float:
    """Calculates the Median Absolute Deviation for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :param constant: factor to multiply median absolute deviation by, defaults to 1.4826
    :type constant: float, optional
    :return: the median absolute deviation for the given CGM trace
    :rtype: float
    """
    return constant * np.nanmedian(np.abs(df[GLUCOSE] - np.nanmedian(df[GLUCOSE])))

def MAG(df: pd.DataFrame) -> float:
   """Calculates the Mean Absolute Glucose (MAG) for the given CGM trace.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the MAG for the given CGM trace
    :rtype: float
    """
   df.dropna(subset=[GLUCOSE], inplace=True)
   data = df[(df[TIME].dt.minute == (df[TIME].dt.minute).iloc[0]) & (df[TIME].dt.second == (df[TIME].dt.second).iloc[0])][GLUCOSE]
   return np.sum(data.diff().abs()) / data.size

def MAGE(df: pd.DataFrame, short_ma: int = 5, long_ma: int = 32, max_gap: int = 180) -> float:
   """Calculates the Mean Amplitude of Glycemic Excursions (MAGE) for the given CGM trace.

   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :param short_ma: number of data points utilized to calculate short moving average values, defaults to 5
   :type short_ma: int, optional
   :param long_ma: number of data points utilized to calculate long moving average values, defaults to 32
   :type long_ma: int, optional
   :param max_gap: number of minutes a gap between CGM data points can be without having to split the MAGE calculation into multiple segments, defaults to 180
   :type max_gap: int, optional
   :return: the MAGE for the given CGM trace
   :rtype: float
   """
   data = df.reset_index(drop=True)

   config.read('config.ini')
   interval = int(config["variables"]["interval"])

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
   """Calculates the Mean Amplitude of Glycemic Excursions (MAGE) for a specific segment of a CGM trace.
   Algorithm for calculating MAGE is based on iglu's implementation, and this method is a helper for the MAGE() function.

   :param df: a Pandas DataFrame containing preprocessed CGM data without significant gaps (as defined in the MAGE() function)
   :type df: 'pandas.DataFrame'
   :param short_ma: number of data points utilized to calculate short moving average values, defaults to 5
   :type short_ma: int, optional
   :param long_ma: number of data points utilized to calculate long moving average values, defaults to 32
   :type long_ma: int, optional
   :return: the MAGE for the given segment of a CGM trace
   :rtype: float
   """
   averages = pd.DataFrame()
   averages[GLUCOSE] = df[GLUCOSE]
   averages.reset_index(drop=True, inplace=True)

   if short_ma < 1 or long_ma < 1:
      raise Exception("Moving average spans must be positive, non-zero integers.")

   if short_ma >= long_ma:
      raise Exception("Short moving average span must be smaller than the long moving average span.")
   
   if averages.shape[0] < long_ma:
      return np.nan

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

def m_value(df: pd.DataFrame, r: int = 100) -> float:
   """Calculates the M-value for the given CGM trace.

   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :param r: a reference value utilized for calculating the M-value, defaults to 100
   :type r: int, optional
   :return: the M-value for the given CGM trace
   :rtype: float
   """
   return (abs(10 * np.log10(df[GLUCOSE] / r)) ** 3).mean()

def ROC(df: pd.DataFrame, timedelta: int = 15) -> pd.Series:
   """Returns a Pandas Series with the rate of change in glucose values at every data point

   :param df: a Pandas DataFrame containing preprocessed CGM data
   :type df: 'pandas.DataFrame'
   :param timedelta: difference in time (in minutes) to utilize when calculating differences between data points, defaults to 15
   :type timedelta: int, optional
   :return: a Pandas Series with the rate of change in glucose values at every data point
   :rtype: 'pandas.Series'
   """
   config.read('config.ini')
   interval = int(config["variables"]["interval"])
   if timedelta < interval:
      raise Exception("Given timedelta must be greater than resampling interval.")

   positiondelta = round(timedelta / interval)
   return df[GLUCOSE].diff(periods=positiondelta) / timedelta

def number_readings(df: pd.DataFrame):
   return df[GLUCOSE].count()

def FBG(df: pd.DataFrame) -> float:
    # Ensure time is in datetime
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date

    daily_fbg_means = []
    for day, day_df in df.groupby('date'):
        # Filter data for readings between 6:00 and 7:00 AM
        morning_df = day_df[(day_df[TIME].dt.hour == 6)]
        morning_df = morning_df.sort_values(by=TIME)

        if len(morning_df) >= 6:
            # Take the first 6 readings within 6:00-7:00 AM
            first_6 = morning_df.head(6)
            daily_fbg_means.append(first_6[GLUCOSE].mean())
    
    return np.nan if not daily_fbg_means else np.mean(daily_fbg_means)

def LSBG(df: pd.DataFrame) -> float:
    """Calculates the Lowest Sleeping Blood Glucose (LSBG).
    
    Defined as the mean of the six lowest consecutive glucose measures between
    23:30 and 06:30 (spanning midnight) for each day, averaged over all days.
    
    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the LSBG for the given CGM trace
    :rtype: float
    """
    # Drop rows with missing glucose values
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date

    daily_lowest_means = []
    unique_dates = sorted(df['date'].unique())

    # For each date d, nighttime window: d 23:30 -> (d+1) 06:30
    for d in unique_dates:
        start_night = pd.to_datetime(d) + pd.Timedelta(hours=23, minutes=30)
        end_night = pd.to_datetime(d) + pd.Timedelta(days=1, hours=6, minutes=30)

        night_df = df[(df[TIME] >= start_night) & (df[TIME] < end_night)].sort_values(by=TIME)
        
        if len(night_df) >= 6:
            # Compute rolling mean of 6 consecutive readings
            rolling_means = night_df[GLUCOSE].rolling(window=6).mean()
            # Get the minimum of those rolling means for the night
            min_rolling_mean = rolling_means.min()
            if not np.isnan(min_rolling_mean):
                daily_lowest_means.append(min_rolling_mean)

    return np.nan if not daily_lowest_means else np.mean(daily_lowest_means)

def mean_24h(df: pd.DataFrame) -> float:
    """Calculates the Mean 24-h starting from 23:30 to the next day's 23:30 for each day.
    
    For each date d, the 24-hour window is from d 23:30 to (d+1) 23:30.
    We compute the mean glucose within this window for each day, then average across all days.
    
    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the mean 24-h BG value for the given CGM trace
    :rtype: float
    """
    # Drop rows with missing glucose values
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date

    daily_means = []
    unique_dates = sorted(df['date'].unique())

    for d in unique_dates:
        start_period = pd.to_datetime(d) + pd.Timedelta(hours=23, minutes=30)
        end_period = start_period + pd.Timedelta(days=1)  # next day's 23:30

        period_df = df[(df[TIME] >= start_period) & (df[TIME] < end_period)]

        if not period_df.empty:
            daily_mean = period_df[GLUCOSE].mean()
            daily_means.append(daily_mean)

    return np.nan if not daily_means else np.mean(daily_means)

def mean_24h_auc(df: pd.DataFrame) -> float:
    """Calculates the mean 24-hour AUC (Area Under the Curve) using the trapezoidal rule,
    with a 24-hour period defined from 23:30 of one day to 23:30 of the next day.
    
    For each date d, the 24-hour period is from d 23:30 to (d+1) 23:30.
    The AUC for that day is the integral of glucose over time.
    We then average over all days to get the mean 24-hour AUC.
    
    Integration is done using scipy.integrate.trapezoid with actual timestamps as 'x'.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the mean 24-hour AUC for the given CGM trace (23:30–23:30)
    :rtype: float
    """
    # Drop rows with missing glucose values
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date
    
    daily_aucs = []
    unique_dates = sorted(df['date'].unique())

    for d in unique_dates:
        start_period = pd.to_datetime(d) + pd.Timedelta(hours=23, minutes=30)
        end_period = start_period + pd.Timedelta(days=1)

        day_df = df[(df[TIME] >= start_period) & (df[TIME] < end_period)].sort_values(by=TIME)

        if len(day_df) < 2:
            # Not enough data points to form a meaningful trapezoid
            continue
        
        # Compute time array in hours relative to start_period
        times_in_hours = (day_df[TIME] - start_period).dt.total_seconds() / 3600.0
        glucose_values = day_df[GLUCOSE].values

        # Use scipy's trapezoid to integrate glucose over the 24-hour period
        auc = trapezoid(glucose_values, x=times_in_hours)
        daily_aucs.append(auc)
    
    return np.nan if not daily_aucs else np.mean(daily_aucs)

def mean_daytime(df: pd.DataFrame) -> float:
    """
    Calculates the mean daytime glucose, defined as the mean of all measures 
    between 06:30 and 23:30 for each day, averaged across all days.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the mean daytime glucose level
    :rtype: float
    """
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date

    daily_means = []
    for d, day_df in df.groupby('date'):
        start_period = pd.to_datetime(d) + pd.Timedelta(hours=6, minutes=30)
        end_period = pd.to_datetime(d) + pd.Timedelta(hours=23, minutes=30)

        daytime_df = day_df[(day_df[TIME] >= start_period) & (day_df[TIME] < end_period)]

        if not daytime_df.empty:
            daily_mean = daytime_df[GLUCOSE].mean()
            daily_means.append(daily_mean)

    return np.nan if not daily_means else np.mean(daily_means)

def mean_nocturnal(df: pd.DataFrame) -> float:
    """
    Calculates the mean nocturnal glucose, defined as the mean of all measures 
    between 23:30 and 06:30 for each day, averaged across all days.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the mean nocturnal glucose level
    :rtype: float
    """
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date

    daily_means = []
    for d, day_df in df.groupby('date'):
        # Define the nighttime window
        start_period = pd.to_datetime(d) + pd.Timedelta(hours=23, minutes=30)
        end_period = pd.to_datetime(d) + pd.Timedelta(days=1, hours=6, minutes=30)

        night_df = day_df[(day_df[TIME] >= start_period) & (day_df[TIME] < end_period)]

        # If no readings in that interval for this particular night, skip it
        if not night_df.empty:
            daily_mean = night_df[GLUCOSE].mean()
            daily_means.append(daily_mean)

    return np.nan if not daily_means else np.mean(daily_means)

def auc_daytime(df: pd.DataFrame) -> float:
    """
    Calculates the mean daytime AUC (Area Under the Curve) of glucose 
    between 06:30 and 23:30 for each day, and then averages these daily AUCs.

    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the mean daytime AUC
    :rtype: float
    """
    # Drop rows with missing glucose values
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date

    daily_aucs = []
    for d, day_df in df.groupby('date'):
        start_period = pd.to_datetime(d) + pd.Timedelta(hours=6, minutes=30)
        end_period = pd.to_datetime(d) + pd.Timedelta(hours=23, minutes=30)

        daytime_df = day_df[(day_df[TIME] >= start_period) & (day_df[TIME] < end_period)].sort_values(by=TIME)

        if len(daytime_df) < 2:
            # Not enough data points for integration
            continue

        # Compute time array in hours relative to start_period
        times_in_hours = (daytime_df[TIME] - start_period).dt.total_seconds() / 3600.0
        glucose_values = daytime_df[GLUCOSE].values

        # Use scipy's trapezoid to integrate glucose over the daytime period
        auc = trapezoid(glucose_values, x=times_in_hours)
        daily_aucs.append(auc)

    return np.nan if not daily_aucs else np.mean(daily_aucs)

def nocturnal_auc(df: pd.DataFrame) -> float:
    """
    Calculates the mean nocturnal AUC (Area Under the Curve) of glucose 
    between 23:30 and 06:30 for each day, and then averages these daily AUCs.

    For each date d, we define the nocturnal period as d 23:30 to (d+1) 06:30.
    
    :param df: a Pandas DataFrame containing preprocessed CGM data
    :type df: 'pandas.DataFrame'
    :return: the mean nocturnal AUC
    :rtype: float
    """
    df = df.dropna(subset=[GLUCOSE]).copy()
    df['date'] = df[TIME].dt.date

    daily_aucs = []
    unique_dates = sorted(df['date'].unique())

    for d in unique_dates:
        start_period = pd.to_datetime(d) + pd.Timedelta(hours=23, minutes=30)
        end_period = start_period + pd.Timedelta(hours=6, minutes=30)

        night_df = df[(df[TIME] >= start_period) & (df[TIME] < end_period)].sort_values(by=TIME)

        if len(night_df) < 2:
            # Not enough data points for integration
            continue

        # Compute time array in hours relative to start_period
        times_in_hours = (night_df[TIME] - start_period).dt.total_seconds() / 3600.0
        glucose_values = night_df[GLUCOSE].values

        # Use scipy's trapezoid to integrate glucose over the nocturnal period
        auc = trapezoid(glucose_values, x=times_in_hours)
        daily_aucs.append(auc)

    return np.nan if not daily_aucs else np.mean(daily_aucs)


def compute_features(id: str, data: pd.DataFrame) -> dict[str, any]:
   """Calculates statistics and metrics for a single patient within the given DataFrame

   :param id: the patient to calculate features for
   :type id: str
   :param data: Pandas DataFrame containing preprocessed CGM data for one or more patients
   :type data: 'pandas.DataFrame'
   :return: a dictionary (with each key referring to the name of a statistic or metric)
   :rtype: dict[str, any]
   """
   summary = summary_stats(data)

   features = {
      ID: id,
      "ADRR": ADRR(data),
      "COGI": COGI(data),
      "CONGA": CONGA(data),
      "CV": CV(data),
      "Daytime AUC": auc_daytime(data),
      "eA1c": eA1c(data),
      "FBG": FBG(data),
      "First Quartile": summary[1],
      "GMI": GMI(data),
      "GRADE": GRADE(data),
      "GRADE (euglycemic)": GRADE_eugly(data),
      "GRADE (hyperglycemic)": GRADE_hyper(data),
      "GRADE (hypoglycemic)": GRADE_hypo(data),
      "GRI": GRI(data),
      "GVP": GVP(data),
      "HBGI": HBGI(data),
      "Hyperglycemia Index": hyper_index(data),
      "Hypoglycemia Index": hypo_index(data),
      "IGC": IGC(data),
      "J-Index": j_index(data),
      "LBGI": LBGI(data),
      "LSBG": LSBG(data),
      "MAG": MAG(data),
      "MAGE": MAGE(data),
      "Maximum": summary[4],
      "Mean": mean(data),
      "Mean 24h Glucose": mean_24h(data),
      "Mean 24h AUC": mean_24h_auc(data),
      "Mean Absolute Differences": mean_absolute_differences(data),
      "Mean Daytime": mean_daytime(data),
      "Mean Nocturnal": mean_nocturnal(data),
      "Median": summary[2],
      "Median Absolute Deviation": median_absolute_deviation(data),
      "Minimum": summary[0],
      "MODD": MODD(data),
      "M-Value": m_value(data),
      "Nocturnal AUC": nocturnal_auc(data),
      "Number of Readings": number_readings(data),
      "Percent Time Above Range (180)": percent_time_above_range(data),
      "Percent Time Below Range (70)": percent_time_below_range(data),
      "Percent Time in Hyperglycemia": percent_time_in_hyperglycemia(data),
      "Percent Time in Hyperglycemia (level 0)": percent_time_in_level_0_hyperglycemia(data),
      "Percent Time in Hyperglycemia (level 1)": percent_time_in_level_1_hyperglycemia(data),
      "Percent Time in Hyperglycemia (level 2)": percent_time_in_level_2_hyperglycemia(data),
      "Percent Time in Hypoglycemia": percent_time_in_hypoglycemia(data),
      "Percent Time in Hypoglycemia (level 1)": percent_time_in_level_1_hypoglycemia(data),
      "Percent Time in Hypoglycemia (level 2)": percent_time_in_level_2_hypoglycemia(data),
      "Percent Time In Range (70-180)": percent_time_in_range(data),
      "Percent Time In Tight Range (70-140)": percent_time_in_tight_range(data),
      "SD": SD(data),
      "Third Quartile": summary[3],
   }
   return features


def create_features(dataset: pd.DataFrame) -> pd.DataFrame:
   """Takes in a multiindexed Pandas DataFrame containing CGM data for multiple patients/datasets, and
   returns a single indexed Pandas DataFrame containing summary metrics in the form of one row per patient/dataset

   :param dataset: a Pandas DataFrame containing the CGM data to calculate metrics for
   :type dataset: 'pandas.DataFrame'
   :return: a Pandas DataFrame with each row representing a patient in 'dataset' and each column representing a specific statistic or metric
   :rtype: 'pandas.DataFrame'
   """
   with Pool() as pool:
      features = pool.starmap(compute_features, dataset.groupby(ID))
   features = pd.DataFrame(features).set_index([ID])
   return features
