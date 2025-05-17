import os, glob
import pandas as pd
import configparser
import zipfile, tempfile
from importlib import resources
import numpy as np

# Initialize config at module level


# globals for glucose values to replace "Low" and "High" with in the CGM data
LOW = 40
HIGH = 400

def get_config_path() -> str:
    """Returns the full path to the config.ini file in the working directory."""
    return os.path.join(os.getcwd(), "config.ini")

def save_config(config: configparser.ConfigParser):
    """Helper to save the current state of the config back to 'config.ini' in the working directory."""
    with open(get_config_path(), 'w') as configfile:
        config.write(configfile)

def load_config() -> configparser.ConfigParser:
    """Loads the working-directory 'config.ini' if present; otherwise loads
    the default config from the package and writes it out to the working dir.
    Returns a ConfigParser object."""
    config = configparser.ConfigParser()
    wd_config_path = get_config_path()
    if os.path.exists(wd_config_path):
        # Use the config file in the working directory
        config.read(wd_config_path)
    else:
        # Otherwise read the default config from the package
        with resources.files("glucose360").joinpath("config.ini").open("r") as f:
            config.read_file(f)
        # Write it to the working directory for future use/modifications
        save_config(config)
    return config

config = load_config()
INTERVAL = int(config["variables"]["interval"])
ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']

def import_data(
    path: str,
    name: str = None,
    sensor: str = "dexcom",
    id_template: str = None,
    glucose: str = None,
    time: str = None,
    interval: int | None = None,
    max_gap: int = 45,
    output = print
) -> pd.DataFrame:
    """Returns a Multiindexed Pandas DataFrame containing all of the csv data found at the given path.
    The path can lead to a directory, .zip file, or a .csv file. The returned DataFrame holds columns 
    for timestamps and glucose values, and is indexed by patient identifications

    :param path: the path of the directory/zip/csv to be parsed through
    :type path: str
    :param sensor: the CGM device model used (either dexcom, freestyle libre pro, or freestyle libre 2 / freestyle libre 3), defaults to 'dexcom'
    :type sensor: str, optional
    :param id_template: regex dictating how to parse each CSV file's name for the proper patient identification, defaults to None
    :type id_template: str, optional
    :param glucose: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type glucose: str, optional
    :param time: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type time: str, optional
    :param interval: the resampling interval (in minutes) that the data should follow. If None, uses the interval from config, defaults to None
    :type interval: int | None, optional
    :param max_gap: the maximum amount of minutes a gap in the data can be interpolated, defaults to 45 
       (filling in a gap with a longer duration would be considered extrapolation)
    :type max_gap: int, optional
    :return: A Pandas DataFrame containing the preprocessed data found at the given path. This DataFrame holds columns for timestamps, glucose values, weekday/weekend chunking, and waking/sleeping time chunking.
    :rtype: pandas.DataFrame 

    :Example:
    >>> path_to_data = "datasets/patient_data.csv"
    >>> df = import_data(path_to_data)
    """
    # If interval is specified, update the config. Otherwise use the config value
    if interval is not None:
        updated_config = config['variables']
        updated_config['interval'] = str(interval)
        config["variables"] = updated_config
        # Write changes to working directory config
        save_config(config)
    else:
        interval = int(config['variables']['interval'])

    # get file extension of where the given path points
    ext = os.path.splitext(path)[1]

    # path leads to directory
    if ext == "":
       if not os.path.isdir(path):
          raise ValueError("Directory does not exist")
       else:
          return _import_directory(path, sensor, id_template, glucose, time, interval, max_gap, output)
    
    # check if path leads to .zip or .csv
    if ext.lower() in [".csv", ".zip"]:
       if not os.path.isfile(path):
          raise ValueError("File does not exist")
    else:
       raise ValueError("Invalid file type")
   
    # path leads to .csv
    if ext.lower() == ".csv":
       return _import_csv(path, sensor, id_template, glucose, time, interval, max_gap)

    # otherwise has to be a .zip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
      # create a temporary directory to pull from
      with tempfile.TemporaryDirectory() as temp_dir:
         zip_ref.extractall(temp_dir)
         dir = name or path.split("/")[-1].split(".")[0]
         return _import_directory((temp_dir + "/" + dir), sensor, id_template, glucose, time, interval, max_gap, output)
    
def _import_directory(
    path: str,
    sensor: str = "dexcom",
    id_template: str = None,
    glucose: str = None,
    time: str = None,
    interval: int | None = None,
    max_gap: int = 45,   
    output = print
) -> pd.DataFrame:
    """Returns a Multiindexed Pandas DataFrame containing all of the csv data found at the given path.
    The path must lead to a directory containing .csv files. The returned DataFrame holds columns 
    for timestamps and glucose values, and is indexed by patient identifications

    :param path: the path of the directory to be parsed through
    :type path: str
    :param sensor: the CGM device model used (either dexcom, freestyle libre pro, or freestyle libre 2 / freestyle libre 3), defaults to 'dexcom'
    :type sensor: str, optional
    :param id_template: regex dictating how to parse each CSV file's name for the proper patient identification, defaults to None
    :type id_template: str, optional
    :param glucose: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type glucose: str, optional
    :param time: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type time: str, optional
    :param interval: the resampling interval (in minutes) that the data should follow, defaults to 5
    :type interval: int, optional
    :param max_gap: the maximum amount of minutes a gap in the data can be interpolated, defaults to 45 
       (filling in a gap with a longer duration would be considered extrapolation)
    :type max_gap: int, optional
    :return: A Pandas DataFrame containing the preprocessed data found at the given path. This DataFrame holds columns for timestamps, glucose values, weekday/weekend chunking, and waking/sleeping time chunking.
    :rtype: pandas.DataFrame 
    """
    csv_files = glob.glob(path + "/*.csv")
    num_files = len(csv_files)

    if num_files == 0:
       raise Exception("No CSV files found.")
    
    output(f"{num_files} .csv files were found in the specified directory.")
    
    data: list[pd.DataFrame] = []
    num_valid_files = num_files
    for file in csv_files:
       try:
          data.append(_import_csv(file, sensor, id_template, glucose, time, interval, max_gap))
       except:
          num_valid_files -= 1
   
    output(f"{num_valid_files} .csv files were successfully imported.")

    if len(data) == 0: raise Exception("CSV files found, but none were valid.")  
    df = pd.concat(data)

    output(f"{df.index.unique().size} sections were found in the imported data.")

    return df

def _import_csv(
    path: str,
    sensor: str = "dexcom", 
    id_template: str = None,
    glucose: str = None,
    time: str = None,
    interval: int | None = None, 
    max_gap: int = 45
) -> pd.DataFrame:
    """Returns a Multiindexed Pandas DataFrame containing all of the csv data found at the given path.
    The path must lead to a .csv file. The returned DataFrame holds columns 
    for timestamps and glucose values, and is indexed by patient identifications

    :param path: the path of the csv file to be parsed through
    :type path: str
    :param sensor: the CGM device model used (either 'dexcom', 'freestyle libre pro', 'freestyle libre 2', 'freestyle libre 3', or 'columns'), defaults to 'dexcom'
    :type sensor: str, optional
    :param id_template: regex dictating how to parse the CSV file's name for the proper patient identification, 
       or the name of the patient identification column if using the 'columns' sensor, defaults to None
    :type id_template: str, optional
    :param glucose: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type glucose: str, optional
    :param time: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type time: str, optional
    :param interval: the resampling interval (in minutes) that the data should follow, defaults to 5
    :type interval: int, optional
    :param max_gap: the maximum amount of minutes a gap in the data can be interpolated, defaults to 45 
       (filling in a gap with a longer duration would be considered extrapolation)
    :type max_gap: int, optional
    :return: A Pandas DataFrame containing the preprocessed data found at the given path. This DataFrame holds columns for timestamps, glucose values, weekday/weekend chunking, and waking/sleeping time chunking.
    :rtype: pandas.DataFrame 
    """
    data = pd.DataFrame()
    if sensor == "dexcom":
       data = _import_csv_dexcom(path, id_template, glucose, time)
    elif sensor == "freestyle libre 2" or sensor == "freestyle libre 3":
       data = _import_csv_freestyle_libre_23(path, id_template, glucose, time)
    elif sensor == "freestyle libre pro":
       data = _import_csv_freestyle_libre_pro(path, id_template, glucose, time)
    elif sensor == "columns":
       data = _import_csv_columns(path, id_template, glucose, time)
   
    preprocessed_data = preprocess_data(data, interval, max_gap)
    return preprocessed_data

def _import_csv_columns(
   path: str,
   id_col: str = None,
   glucose_col: str = None,
   time_col: str = None,
) -> pd.DataFrame:
   """Returns a Pandas DataFrame containing all of the csv data found at the given path.
   The path must lead to a .csv file with three columns (identification, timestamp, and glucose value) containing CGM data. The returned DataFrame holds columns 
   for timestamps, glucose values, and the patient identification

   :param path: the path of the csv file to be parsed through
   :type path: str
   :param id_col: the name of the column containing the patient identification(s), defaults to None
   :type id_col: str, optional
   :param glucose_col: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used), defaults to None
   :type glucose_col: str, optional
   :param time_col: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used), defaults to None
   :type time_col: str, optional
   :return: A Pandas DataFrame containing the raw data found at the given path. This DataFrame holds columns for timestamps, glucose values, and the patient identification.
   :rtype: pandas.DataFrame 
   """
   df = pd.read_csv(path)
   glucose = glucose_col or "Glucose Value (mg/dL)"
   time = time_col or "Timestamp (YYYY-MM-DDThh:mm:ss)"
   id = id_col or "ID"

   df.rename(columns={glucose: GLUCOSE, time: TIME, id: ID}, inplace=True)
   return df

def _import_csv_dexcom(
    path: str,
    id_template: str = None,
    glucose_col: str = None,
    time_col: str = None,
) -> pd.DataFrame:
    """Returns a Pandas DataFrame containing all of the Dexcom csv data found at the given path.
    The path must lead to a .csv file containing CGM data from a Dexcom device. The returned DataFrame holds columns 
    for timestamps, glucose values, and the patient identification

    :param path: the path of the Dexcom csv file to be parsed through
    :type path: str
    :param id_template: regex dictating how to parse the CSV file's name for the proper patient identification, defaults to None
    :type id_template: str, optional
    :param glucose_col: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type glucose_col: str, optional
    :param time_col: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type time_col: str, optional
    :return: A Pandas DataFrame containing the raw data found at the given path. This DataFrame holds columns for timestamps, glucose values, and the patient identification.
    :rtype: pandas.DataFrame 
    """
    df = pd.read_csv(path)
    glucose = glucose_col or "Glucose Value (mg/dL)"
    time = time_col or "Timestamp (YYYY-MM-DDThh:mm:ss)"
    id = _retrieve_id_dexcom(path.split("/")[-1], df, id_template)

    df.rename(columns={glucose: GLUCOSE, time: TIME}, inplace=True)
    df[ID] = id
    return df

def _import_csv_freestyle_libre_23(
   path: str,
   id_template: str = None,
   glucose_col: str = None,
   time_col: str = None,  
) -> pd.DataFrame:
   """Returns a Pandas DataFrame containing all of the FreeStyle Libre 2 or 3 csv data found at the given path.
    The path must lead to a .csv file containing CGM data from FreeStyle Libre 2 or FreeStyle Libre 3 devices. The returned DataFrame holds columns 
    for timestamps, glucose values, and the patient identification

    :param path: the path of the FreeStyle Libre 2 or 3 csv file to be parsed through
    :type path: str
    :param id_template: regex dictating how to parse the CSV file's name for the proper patient identification, defaults to None
    :type id_template: str, optional
    :param glucose_col: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type glucose_col: str, optional
    :param time_col: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type time_col: str, optional
    :return: A Pandas DataFrame containing the raw data found at the given path. This DataFrame holds columns for timestamps, glucose values, and the patient identification.
    :rtype: pandas.DataFrame 
    """
   glucose = glucose_col or "Historic Glucose mg/dL"
   time = time_col or "Device Timestamp"
   return _import_csv_freestyle_libre(path, id_template, glucose, time)

def _import_csv_freestyle_libre_pro(
    path: str,
    id_template: str = None,
    glucose_col: str = None,
    time_col: str = None,
) -> pd.DataFrame:
    """Returns a Pandas DataFrame containing all of the FreeStyle Libre Pro csv data found at the given path.
    The path must lead to a .csv file containing CGM data from a FreeStyle Libre Pro device. The returned DataFrame holds columns 
    for timestamps, glucose values, and the patient identification

    :param path: the path of the FreeStyle Libre Pro csv file to be parsed through
    :type path: str
    :param id_template: regex dictating how to parse the CSV file's name for the proper patient identification, defaults to None
    :type id_template: str, optional
    :param glucose_col: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type glucose_col: str, optional
    :param time_col: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used), defaults to None
    :type time_col: str, optional
    :return: A Pandas DataFrame containing the raw data found at the given path. This DataFrame holds columns for timestamps, glucose values, and the patient identification.
    :rtype: pandas.DataFrame 
    """
    glucose = glucose_col or "Historic Glucose(mg/dL)"
    time = time_col or "Meter Timestamp"
    return _import_csv_freestyle_libre(path, id_template, glucose, time)

def _import_csv_freestyle_libre(
   path: str,
   id_template: str,
   glucose_col: str,
   time_col: str
) -> pd.DataFrame:
   """Returns a Pandas DataFrame containing all of the FreeStyle Libre csv data found at the given path.
    The path must lead to a .csv file containing CGM data from FreeStyle Libre 2/3/Pro devices. The returned DataFrame holds columns 
    for timestamps, glucose values, and the patient identification. 

    :param path: the path of the FreeStyle Libre csv file to be parsed through
    :type path: str
    :param id_template: regex dictating how to parse the CSV file's name for the proper patient identification
    :type id_template: str
    :param glucose_col: the name of the column containing the glucose values in the .csv files (if different than the default for the CGM sensor being used)
    :type glucose_col: str
    :param time_col: the name of the column containing the timestamps in the .csv files (if different than the default for the CGM sensor being used)
    :type time_col: str
    :return: A Pandas DataFrame containing the raw data found at the given path. This DataFrame holds columns for timestamps, glucose values, and the patient identification.
    :rtype: pandas.DataFrame 
    """
   id = pd.read_csv(path, nrows=1)["Patient report"].iloc[0] if not id_template else _id_from_filename(path.split("/")[-1], id_template)
   df = pd.read_csv(path, skiprows=2)

   df.rename(columns={glucose_col: GLUCOSE, time_col: TIME}, inplace=True)
   df[ID] = id
   return df

def _retrieve_id_dexcom(name: str, df: pd.DataFrame, id_template: str = None) -> str:
   """Returns the appropriate identification for the given raw Dexcom CGM data based on the given template.
   If the template is None, the identification will be pulled from the patient information fields from within the CSV.
   Otherwise, the filename will be parsed accordingly.

   :param name: the name of the file to parse for an identification
   :type name: str
   :param df: a Pandas DataFrame containing the raw data from a Dexcom CSV file
   :type df: pandas.DataFrame
   :param id_template: regex indicating how to parse the filename for the identification, defaults to None
   :type id_template: str, optional
   :return: the proper identification for the raw data in the given dataframe
   :rtype: str
   """
   if id_template and "first" not in id_template and "last" not in id_template and "patient_identifier" not in id_template:
      # need to parse file name for id
      return _id_from_filename(name, id_template)

   # use Dexcom fields for id instead
   first = df["Patient Info"].iloc[0]
   last = df["Patient Info"].iloc[1]
   patient_identifier = df["Patient Info"].iloc[2]
   id = df["Patient Info"].iloc[0] + df["Patient Info"].iloc[1] 
   if id_template: id = id_template.format(first=first, last=last, patient_identifier=patient_identifier)
   return id

def _id_from_filename(name: str, id_template: str):
   """Parses the given filename for an identification using a regex template.

   :param name: the filename to parse for an identification
   :type name: str
   :param id_template: regex indicating how to parse the filename for the identification
   :type id_template: str
   :return: the identification from the filename
   :rtype: str
   """
   import re
   pattern = re.compile(fr"{id_template}")
   match = pattern.search(name)
   if match is None:
      raise Exception("The RegEx ID template passed does not match the file name.")
   id = str(match.group("id"))
   try: 
      section = str(match.group("section"))
      id += f" ({section})"
   except:
      print(f"'Section' not defined for patient {id}.")
   return id

def preprocess_data(
   df: pd.DataFrame,
   interval: int | None = None,
   max_gap: int = 45
) -> pd.DataFrame:
   """Returns a Pandas DataFrame containing the preprocessed CGM data within the given dataframe.
   As part of the preprocessing phase, the data will be converted into the proper data types, resampled, interpolated, chunked, and
   indexed by identification (alongside all 'Low's and 'High's being replaced and all edge null values being dropped)
   
   :param df: the Pandas DataFrame containing the CGM data to preprocess
   :type df: pandas.DataFrame
   :param interval: the resampling interval (in minutes) that the data should follow. If None, uses the interval from config, defaults to None
   :type interval: int | None, optional
   :param max_gap: the maximum duration (in minutes) of a gap in the data that should be interpolated, defaults to 45
   :type max_gap: int, optional
   :return: A Pandas DataFrame containing the preprocessed CGM data. This DataFrame is indexed by identification and holds columns for
      timestamps, glucose values, day chunking, and time chunking.
   :rtype: pandas.DataFrame 

   :Example:
   >>> # 'df' is a Pandas DataFrame already containing your CGM data, with columns for glucose values, timestamps, and identification
   >>> preprocessed_df = preprocess_data(df)
   """
   # Create a copy of the DataFrame to avoid SettingWithCopyWarning
   df = df.copy()
   
   df = df.dropna(subset=[GLUCOSE])
   df.loc[:, GLUCOSE] = df[GLUCOSE].replace("Low", LOW)
   df.loc[:, GLUCOSE] = df[GLUCOSE].replace("High", HIGH)
   df.reset_index(drop=True, inplace=True)

   df[TIME] = pd.to_datetime(df[TIME])
   df[GLUCOSE] = pd.to_numeric(df[GLUCOSE])

   df = df[[TIME, GLUCOSE, ID]].copy()
   if interval is None:
       interval = int(config['variables']['interval'])
   df = _resample_data(df, interval, max_gap)
   df = df.loc[df[GLUCOSE].first_valid_index():df[GLUCOSE].last_valid_index()]
   df = _chunk_day(_chunk_time(df))
   df.set_index(ID, inplace=True)
   return df

def _resample_data(df: pd.DataFrame, minutes: int = 5, max_gap: int = 45) -> pd.DataFrame:
    """Resamples and (if needed) interpolates the given default-indexed DataFrame.
    Used mostly to preprocess the data in the csv files being imported in _import_csv().

    :param df: the DataFrame to be resampled and interpolated
    :type df: pandas.DataFrame
    :param minutes: the length of the interval to be resampled into (in minutes), defaults to 5
    :type minutes: int
    :param max_gap: the maximum duration (in minutes) of gaps that should be interpolated, defaults to 45
    :type max_gap: int
    :return: A Pandas DataFrame containing the resampled and interpolated. This DataFrame holds columns for timestamps, glucose values, and the patient identification.
    :rtype: pandas.DataFrame
    """
    id = df.at[0, ID]

    # Sort the DataFrame by datetime
    resampled_df = df.sort_values(by=[TIME])
    resampled_df = resampled_df.set_index(TIME)

    interval = str(minutes) + "min"
    # generate the times that match the frequency
    resampled_df = resampled_df.asfreq(interval)
    # add in the original points that don't match the frequency (just for linear time-based interpolation)
    resampled_df.reset_index(inplace=True)
    resampled_df = (pd.concat([resampled_df, df])).drop_duplicates(subset=[TIME])
    resampled_df.sort_values(by=[TIME], inplace=True)

    # interpolate the missing values
    resampled_df.set_index(TIME, inplace=True)
    resampled_df = _interpolate_data(resampled_df, max_gap)
    
    # now that the values have been interpolated, remove the points that don't match the frequency
    resampled_df = resampled_df.asfreq(interval)
    resampled_df[ID] = id # resampled data points might have empty ID values
    resampled_df.reset_index(inplace=True)

    return resampled_df

def _interpolate_data(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Only linearly interpolates NaN glucose values for time gaps that are less than the given number of minutes.
    Used mainly in preprocessing for csv files that are being imported in _import_csv().
    
    :param df: the Pandas DataFrame containing the CGM data to interpolate
    :type df: pandas.DataFrame
    :param max_gap: the maximum minute length of gaps that should be interpolated
    :type max_gap: int
    :return: a Pandas DataFrame with interpolated CGM data
    :rtype: pandas.DataFrame
    """
    config.read('config.ini')
    interval = int(config["variables"]["interval"])

    # based heavily on https://stackoverflow.com/questions/67128364/how-to-limit-pandas-interpolation-when-there-is-more-nan-than-the-limit

    s = df[GLUCOSE].notnull()
    s = s.ne(s.shift()).cumsum()

    m = df.groupby([s, df[GLUCOSE].isnull()])[GLUCOSE].transform('size').where(df[GLUCOSE].isnull())
    df[GLUCOSE] = df[GLUCOSE].interpolate(method="time", limit_area="inside").mask(m >= int(max_gap / interval))

    return df

def _chunk_time(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a new column to the given DataFrame specifying whether the values occur during a waking or sleeping period

    :param df: the Pandas DataFrame to add the new column to (must contain a column for timestamps)
    :type df: pandas.DataFrame
    :return: the Pandas DataFrame with the added column for time chunking
    :rtype: pandas.DataFrame
    """
    times = df[TIME] - df[TIME].dt.normalize()
    is_waking = (times >= pd.Timedelta(hours=8)) & (times <= pd.Timedelta(hours=22))
    df["Time Chunking"] = is_waking.replace({True: "Waking", False: "Sleeping"})
    return df

def _chunk_day(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a new column to the given DataFrame specifying whether the values occur during a weekday or the weekend

    :param df: the Pandas DataFrame to add the new column to (must contain a column for timestamps)
    :type df: pandas.DataFrame
    :return: the Pandas DataFrame with the added column for day chunking
    :rtype: pandas.DataFrame
    """
    is_weekend = df[TIME].dt.dayofweek > 4
    df["Day Chunking"] = is_weekend.replace({True: "Weekend", False: "Weekday"})
    return df

def segment_data(path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits patients' data into multiple segments based on a given .csv file containing ID's and DateTimes.
    
    :param path: path of the .csv file containing identifications and timestamps indicating where to split the given DataFrame
    :type path: str
    :param df: the DataFrame to split based on the given .csv file
    :type df: pandas.DataFrame
    :return: a Pandas DataFrame with the data split accordingly
    :rtype: pandas.DataFrame
    """
    # Read the segments CSV file
    segments = pd.read_csv(path)
    segments[TIME] = pd.to_datetime(segments[TIME])
    
    # Sort segments by TIME
    segments.sort_values(['ID', TIME], inplace=True)
    
    # Create a copy of the original dataframe to avoid modifying it directly
    df_copy = df.copy()
    df_copy[TIME] = pd.to_datetime(df_copy[TIME])
    df_copy = df_copy.reset_index()
    
    # Initialize a Segment column in the original dataframe
    df_copy['Segment'] = 0
    
    # Use a dictionary to keep track of segment counters
    segment_counters = {id: 1 for id in segments['ID'].unique()}
    
    # Iterate over each row in the segments dataframe
    for _, segment_row in segments.iterrows():
        id = segment_row['ID']
        time = segment_row[TIME]
        
        # Create a mask for rows before the current segment time
        mask = (df_copy['ID'] == id) & (df_copy[TIME] < time) & (df_copy['Segment'] == 0)
        
        # Update the segment counter for those rows
        df_copy.loc[mask, 'Segment'] = segment_counters[id]
        
        # Increment the segment counter
        segment_counters[id] += 1
    
    # Assign the segment counter to rows after the last date
    for id in segment_counters.keys():
        mask = (df_copy['ID'] == id) & (df_copy['Segment'] == 0)
        df_copy.loc[mask, 'Segment'] = segment_counters[id]
    
    # Combine ID and Segment to form the new ID
    df_copy['ID'] = df_copy['ID'].astype(str) + '_' + df_copy['Segment'].astype(str)
    
    # Drop the Segment column as it's no longer needed
    df_copy.drop(columns=['Segment'], inplace=True)
    
    return df_copy