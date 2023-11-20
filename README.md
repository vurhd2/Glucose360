# Python-CGM-Package

**Necessary dependencies**:
- pandas 2.1.1
- numpy 1.26.1,
- scipy 1.11.3,
- matplotlib 3.8.0,
- seaborn 0.13.0

**Getting Started**:
All data should be in the form of Dexcom-formatted .csv files, which all share the same column names for the glucose values and times.

1. **Load and Preprocess** To simply import and preprocess this data, utilize `import_directory(path)` from *preprocessing.py*, where `path` is the location of a folder housing all the csv files you wish to import. This function will return a preprocessed (resampled + interpolated) Pandas DataFrame containing glucose value and time columns that are indexed by `id`, a concatenation of the patient's first and last name. Going forward, this DataFrame should be known as `df`.
    1. There are also other parameters for the name of the glucose value column, the name of the time column, and the integer length of the resampling interval, `glucose_col`, `time_col`, and `interval` respectively.
    2. If no values are passed here, the code will automatically assume that `glucose_col` is *"Glucose Value (mg/dL)"* and `time_col` is *"Timestamp (YYYY-MM-DDThh:mm:ss)"*, as well as resample in 5-minute intervals.       

2. **Event Filtering** If you need to look at certain timeframes (or *events*) for each patient's data, utilize `retrieve_event_data(df, events, before, after, desc)` from *events.py*. This will return a DataFrame (in a similar format to the one generated at the start) containing the patient data only within the specific event timeframes as well as an added description column.
    1. `df`: see #1 above
    2. `events` is a **default-indexed** (don't set an index!) DataFrame with 5 columns. Creating a dictionary with the appropriate names/values and then converting it to a DataFrame through Pandas' `DataFrame.from_records()` seems to work fairly well for these purposes.
        1. an `id`, indicating which patient this event refers to (see #1 above, column name **MUST** be *"id"*)
        2. a DateTime, indicating when the event supposedly occurs for the patient (column name should be the same as the time column when importing data)
        3. an integer indicating how many hours before the DateTime should be included
        4. an integer indicating how many hours after the DateTime should be included
        5. a description
    3. `before` is the name of your column indicating how many hours before the DateTime should be included. If not specified, your column name should simply be *"before"*
    4. `after` is the name of your column indicating how many hours after the DateTime should be included. If not specified, your column name should simply be *"after"*
    5. `desc` is the name of your column describing this event (to help differentiate). If not specified, your column name should simply be *"description"*

3. **Feature Calculation** Bulk features/metrics can be calculated for these patients by running `create_features(df)` from *features.py*. This method should return a Pandas DataFrame where each row is indexed by `id` and each column is a specific feature/metric.
    1. If attempting to calculate these metrics for data only within specific timeframes (see #2 for **events** above), utilize `create_event_features(df, events, before, after, desc)`. This will return a similar DataFrame as `create_features(df)`, but just for those event intervals.
        1. See #1 above for `df`
        2. See #2 above for `events`, `before`, `after`, and `desc`

4. **Plots** Three types of plots are currently implemented: *daily*, *spaghetti*, and *AGP*.
    1.  `daily_plot_all(df, events, save)` can be used to generate daily plots for all patients in the imported DataFrame, or `daily_plot(df, id, events, save)` can be used to only generate a daily plot for a specific patient.
        1. See #1 above for `df` and `id`
        2. If `events` is **optionally** provided (see #2 above), the plot will display vertical lines indicating where the "event" occurred. Otherwise, it is perfectly fine to provide nothing (or `None`) for the `events` argument.
        3. `save` is a boolean indicating whether you would like to save the generated plots onto your local drive. By default, `save` is `False`, but can be set to `True` if needed.
    2.  `spaghetti_plot_all(df, save)` can be used to generate spaghetti plots for all patients in the imported DataFrame, or `daily_plot(df, id, save)` can be used to only generate a spaghetti plot for a specific patient.
    3.  `AGP_plot_all(df, save)` can be used to generate AGP report plots for all patients in the imported DataFrame, or `AGP_plot(df, id, save)` can be used to only generate an AGP report plot for a specific patient. The AGP report plots generated have a theme similar to older versions of the AGP report, not the most recent versions.
