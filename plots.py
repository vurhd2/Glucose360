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
      daily_plot(df, events, id, save)

"""
Plots and saves only the given patient's data
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
@param id   the id of the patient whose data is graphed
"""
def daily_plot(df, events, id, save=False):
   data = df.loc[id]

   event_data = events.set_index('id').loc[id]

   # Convert timestamp column to datetime format
   data[time()] = pd.to_datetime(data[time()])

   #data['Day'] = data[time()].dt.date

   plot = sns.relplot(data=data, kind="line", x=time(), y=glucose())

   for ax in plot.axes.flat:
      if isinstance(event_data, pd.DataFrame):
         for index, row in event_data.iterrows():
            ax.axvline(pd.to_datetime(row[time()]), color="orange")
      else:
         ax.axvline(pd.to_datetime(event_data[time()]), color="orange")

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
   data[time()] = pd.to_datetime(data[time()])

   data['Day'] = data[time()].dt.date

   times = data[time()] - data[time()].dt.normalize()
   # need to be in a DateTime format so seaborn can tell how to scale the x axis labels
   data['Time'] = pd.to_datetime(['1/1/1970' for i in range(data[time()].size)]) + times

   data.sort_values(by=[time()], inplace=True)

   plot = sns.relplot(data=data, kind="line", x='Time', y=glucose(), hue='Day')

   plt.xticks(pd.to_datetime([f"1/1/1970T{hour:02d}:00:00" for hour in range(24)]), (f"{hour:02d}:00" for hour in range(24)))
   plt.xticks(rotation=45)
   plt.show() # might result in an empty plot based on osx or matplotlib version apparently

   if save:
      plt.savefig("./plots/" + str(id) + 'Spaghetti.png', bbox_inches='tight')
