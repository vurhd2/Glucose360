import pandas as pd
import seaborn as sns
from preprocessing import glucose, time
import matplotlib.pyplot as plt

"""
Plots and saves all of the patients data in separate graphs
@param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
"""
def daily_plot_all(df):
   sns.set_theme()
   for id, data in df.groupby('id'):
      plot = sns.relplot(data=data, kind="line", x=time(), y=glucose())

      plot.savefig("./plots/" + str(id) + '.png')

"""
Plots and saves only the given patient's data
@param df   a default DataFrame with only DateTime and Glucose columns
"""
def daily_plot(df):
   plot = sns.relplot(data=df, kind="line", x=time(), y=glucose())

   plot.savefig("./plots/" + str(id) + '.png')
