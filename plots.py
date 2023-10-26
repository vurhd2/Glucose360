import pandas as pd
import seaborn as sns
from preprocessing import glucose, time
import matplotlib.pyplot as plt

"""
Plots and saves all of the patients data in separate graphs
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
"""
def daily_plot_all(df, events, save=False):
   sns.set_theme()
   for id, data in df.groupby('id'):
      plot = sns.lineplot(data=data, x=time(), y=glucose())

      events.set_index('id', inplace=True)

      event_times = pd.to_datetime(events.loc[id][time()])
      print(event_times)
      plt.vlines(x = event_times, ymin = 0, ymax = 1, colors='red', linestyles='dashed')

      #for row in events.loc[id]:
         #plot.axvline(row[time()], color='r')

      events.reset_index(inplace=True)

      plt.show()

      if save:
         plot.savefig("./plots/" + str(id) + 'Daily.png')

"""
Plots and saves only the given patient's data
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
@param id   the id of the patient whose data is graphed
"""
def daily_plot(df, id, save=False):
   data = df.loc[id]
   plot = sns.lineplot(data=data, x=time(), y=glucose())

   plot.axvline(pd.to_datetime("2023-08-08 7:45:35"))

   plt.show()

   if save:
      plot.savefig("./plots/" + str(id) + 'Daily.png')

"""
"""
def spaghetti_plot_all(df):
   sns.set_theme()
   for id, data in df.groupby('id'):
      spaghetti_plot(data, id)

"""
"""
def spaghetti_plot(df, id):
   data = df.loc[id]
   data.reset_index(inplace=True)
   
   datetimes = pd.to_datetime(data[time()])
   date = datetimes.dt.date
   times = datetimes.dt.time

   time_data = pd.DataFrame()
   time_data['Day'] = date
   time_data['Time'] = times.astype(str)
   time_data[glucose()] = data[glucose()]

   time_data.set_index('Day', inplace=True)
   time_data.sort_values(by=['Time', 'Day'], ascending=[True, True], inplace=True)

   print(time_data)

   plot = sns.relplot(data=time_data, kind="line", x='Time', y=glucose(), hue='Day')

   plot.savefig("./plots/" + str(id) + 'Spaghetti.png')
