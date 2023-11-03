import pandas as pd
import seaborn as sns
import preprocessing as pp
import matplotlib.pyplot as plt

"""
Plots and saves all of the patients data in separate graphs
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
"""
def daily_plot_all(df, events, save=False):
   sns.set_theme()
   for id, data in df.groupby('id'):
      daily_plot(df, events, id, save)

"""
Plots and saves only the given patient's data
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
@param id   the id of the patient whose data is graphed
"""
def daily_plot(df, events, id, save=False):
   data = df.loc[id]

   data[pp.time()] = pd.to_datetime(data[pp.time()])

   plot = sns.relplot(data=data, kind="line", x=pp.time(), y=pp.glucose())

   # plotting vertical lines to represent the events
   if events is not None:
      event_data = events.set_index('id').loc[id]
      for ax in plot.axes.flat:
         if isinstance(event_data, pd.DataFrame):
            for index, row in event_data.iterrows():
               ax.axvline(pd.to_datetime(row[pp.time()]), color="orange")
         else:
            ax.axvline(pd.to_datetime(event_data[pp.time()]), color="orange")

   plt.ylim(35, 405)
   plt.show()

   if save:
      plot.savefig("./plots/" + str(id) + 'Daily.png')

"""
Sequentially produces spaghetti plots for all the given patients
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
"""
def spaghetti_plot_all(df, save=False):
   sns.set_theme()
   for id, data in df.groupby('id'):
      spaghetti_plot(data, id, save)

"""
Graphs a spaghetti plot for the given patient
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
@param id   the id of the patient whose data should be plotted
"""
def spaghetti_plot(df, id, save=False):
   data = df.loc[id]

   data.reset_index(inplace=True)

   # Convert timestamp column to datetime format
   data[pp.time()] = pd.to_datetime(data[pp.time()])

   data['Day'] = data[pp.time()].dt.date

   times = data[pp.time()] - data[pp.time()].dt.normalize()
   # need to be in a DateTime format so seaborn can tell how to scale the x axis labels
   data['Time'] = pd.to_datetime(['1/1/1970' for i in range(data[pp.time()].size)]) + times

   data.sort_values(by=[pp.time()], inplace=True)

   plot = sns.relplot(data=data, kind="line", x='Time', y=pp.glucose(), hue='Day')

   plt.xticks(pd.to_datetime([f"1/1/1970T{hour:02d}:00:00" for hour in range(24)]), (f"{hour:02d}:00" for hour in range(24)))
   plt.xticks(rotation=45)
   plt.ylim(35, 405)
   plt.show() # might result in an empty plot based on osx or matplotlib version apparently

   if save:
      plt.savefig("./plots/" + str(id) + 'Spaghetti.png', bbox_inches='tight')

"""
Plots and saves all of the patients data in separate graphs
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
"""
def AGP_plot_all(df, save=False):
   sns.set_theme()
   for id, data in df.groupby('id'):
      AGP_plot(data, id, save)

"""
Plots and saves only the given patient's data
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
@param id   the id of the patient whose data is graphed
"""
def AGP_plot(df, id, save):
   if pp.interval() > 5:
      raise Exception("Data needs to have measurement intervals at most 5 minutes long")

   data = df.loc[id]
   data.reset_index(inplace=True)

   data[[pp.time(), pp.glucose()]] = pp.resample_data(data[[pp.time(), pp.glucose()]])
   times = data[pp.time()] - data[pp.time()].dt.normalize()
   # need to be in a DateTime format so seaborn can tell how to scale the x axis labels below
   data['Time'] = pd.to_datetime(['1/1/1970' for i in range(data[pp.time()].size)]) + times

   data.set_index('Time', inplace=True)

   agp_data = pd.DataFrame()
   for time, measurements in data.groupby('Time'):
      metrics = {
         'Time': time,
         '5th': measurements[pp.glucose()].quantile(0.05),
         '25th': measurements[pp.glucose()].quantile(0.25),
         'Median': measurements[pp.glucose()].median(),
         '75th': measurements[pp.glucose()].quantile(0.75),
         '95th': measurements[pp.glucose()].quantile(0.95)
      }
      agp_data = pd.concat([agp_data, pd.DataFrame.from_records([metrics])])
   
   agp_data = pd.melt(agp_data, id_vars=['Time'], 
                      value_vars=['5th', '25th', 'Median', '75th', '95th'],
                      var_name='Metric', value_name=pp.glucose())

   agp_data.sort_values(by=['Time'], inplace=True)

   plot = sns.relplot(data=agp_data, kind="line", x='Time', y=pp.glucose(), hue='Metric', hue_order=['95th', '75th', 'Median', '25th', '5th'])
   plt.xticks(pd.to_datetime([f"1/1/1970T{hour:02d}:00:00" for hour in range(24)]), (f"{hour:02d}:00" for hour in range(24)))
   plt.xticks(rotation=45)
   plt.ylim(35, 405)

   for ax in plot.axes.flat:
      ax.axhline(70, color="green")
      ax.axhline(180, color="green")

   plt.show()
   
   if save:
      plt.savefig("./plots/" + str(id) + 'AGP.png', bbox_inches='tight')
