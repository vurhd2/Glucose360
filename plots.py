import pandas as pd
import seaborn as sns
import preprocessing as pp
import matplotlib.pyplot as plt
import configparser
import json

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from events import retrieve_event_data

import plotly.tools as tls

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

def daily_plot_all(
    df: pd.DataFrame,
    events: pd.DataFrame = None,
    combine: bool = False,
    save: bool = False,
):
    """
    Graphs (and possibly saves) daily plots for all of the patients in the given DataFrame
    @param df         a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
    @param events     a DataFrame containing event timeframes for some (or all) of the given patients
    @param chunk_day  a boolean indicating whether to split weekdays and weekends
    @param save       a boolean indicating whether to download the graphs locally
    """
    sns.set_theme()
    for id, data in df.groupby("id"):
        daily_plot(data, id, events, combine, save)

def daily_plot(
    df: pd.DataFrame,
    id: str,
    events: pd.DataFrame = None,
    combine: bool = False,
    save: bool = False,
):
    """
    Only graphs (and possibly saves) a daily plot for the given patient
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
    @param id   the id of the patient whose data is graphed
    @param events  a DataFrame containing event timeframes for some (or all) of the given patients
    @param combine  a boolean indicating whether to show only one large plot for data from all days
    @param save a boolean indicating whether to download the graphs locally
    """
    data = df.loc[id]

    data[TIME] = pd.to_datetime(data[TIME])
    data.reset_index(inplace=True)
    if not combine: 
       data["Day"] = data[TIME].dt.normalize()
       data["Time"] = pd.Timestamp("1970-01-01T00") + (data[TIME] - data["Day"])

    plot = sns.relplot(
        data=data,
        kind="line",
        x=TIME if combine else "Time",
        y=GLUCOSE,
        col=None if combine else "Day",
        col_wrap=None if combine else 4,
        height=5 if combine else 30,
        aspect=2
    )
    plot.figure.subplots_adjust(top=0.9)
    plot.figure.suptitle(f"Glucose (mg/dL) vs. Timestamp for {id}")
    plot.figure.set_size_inches(10, 6)
    if not combine: plot.figure.subplots_adjust(hspace=0, wspace=0)

    # plotting vertical lines to represent the events
    if events is not None:
      if isinstance(events, pd.DataFrame):
         event_data = events[events["id"] == id] if events is not None else None
         if event_data is not None:
            event_types = event_data['Type'].unique()
            with open('event_colors.json') as colors_file:
               color_dict = json.load(colors_file)
               colors = list(color_dict.values())
               color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
               for index, row in event_data.iterrows():
                  plt.axvline(pd.to_datetime(row[TIME]), color=color_map[row['Type']], label=row['Type'])
      elif events["id"] == id:
         plt.axvline(pd.to_datetime(events[TIME]), color="orange", label=events['Type'])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.35, 0.5), loc='center right')
    plt.tight_layout()

    if not combine:
      plt.xticks(pd.to_datetime([f"1/1/1970T{hour:02d}:00:00" for hour in range(0, 24, 2)]), [f"{hour:02d}" for hour in range(0, 24, 2)])

    for ax in plot.axes.flat:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45 if combine else 90)
    
    plt.ylim(35, 405)
    plt.show() if not save else plot.savefig("./plots/" + str(id) + "Daily.png")

def event_plot_all(df: pd.DataFrame, events: pd.DataFrame):
    sns.set_theme()
    event_data = retrieve_event_data(df, events)
    event_data.set_index('Description', inplace=True)
    for desc, data in event_data.groupby('Description'):
        event = events[events["Description"] == desc]
        event_plot(data, event)

def event_plot(event_data: pd.DataFrame, event: pd.Series):
    pd.set_option('display.max_colwidth', None)
    plot = sns.relplot(
        data=event_data,
        kind="line",
        x=TIME,
        y=GLUCOSE,
    )
    plot.figure.subplots_adjust(top=0.9)
    plot.figure.suptitle(event['Description'].iloc[0])
    plot.figure.set_size_inches(10, 6)

    not_supported_event_types = ['hypo level 1 episode', 'hypo level 2 episode',
                                 'hyper level 1 episode', 'hyper level 2 episode',
                                 'hypo excursion', 'hyper excursion']

    if not (event['Type'].iloc[0] in not_supported_event_types):
      plt.axvline(pd.to_datetime(event[TIME].iloc[0]), color="orange", label=event['Type'].iloc[0])

    plt.legend(loc='right', bbox_to_anchor=(1.0,1.05))
    plt.ylim(35, 405)
    plt.show()

def spaghetti_plot_all(df: pd.DataFrame, chunk_day: bool = False, height: int = 600):
    """
    Sequentially produces spaghetti plots for all the given patients
    @param df         a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
    @param chunk_day  a boolean indicating whether to split weekdays and weekends
    @param height     the height (in pixels) of the resulting graph
    """
    for id, data in df.groupby("id"):
        spaghetti_plot(data, id, chunk_day, height)

def spaghetti_plot(
    df: pd.DataFrame, id: str, chunk_day: bool = False, height: int = 600, app=False
):
    """
    Graphs a spaghetti plot for the given patient
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
    @param id   the id of the patient whose data should be plotted
    @param chunk_day  a boolean indicating whether to split weekdays and weekends
    @param save a boolean indicating whether to download the graphs locally
    """
    data = df.loc[id]
    data["Day"] = data[TIME].dt.date
    times = data[TIME] - data[TIME].dt.normalize()
    data["Time"] = (pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times)
    data.sort_values(by=[TIME], inplace=True)

    fig = px.line(data, x="Time", y=GLUCOSE, color="Day", title=f"Spaghetti Plot for {id}", height=height, facet_col="Day Chunking" if chunk_day else None)
    if app: return fig
    fig.show()

def AGP_plot_all(df: pd.DataFrame, height: int = 600):
    """
    Displays (and possibly saves) AGP Plots for each patient in the given DataFrame
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns containing all patient data
    @param save a boolean indicating whether to download the graphs locally
    """
    for id, data in df.groupby("id"):
        AGP_plot(data, id, height)


def AGP_plot(df: pd.DataFrame, id: str, height: int = 600, app=False):
    """
    Displays (and possibly saves) an AGP Plot for only the given patient in the DataFrame
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns containing all patient data
    @param id   the id of the single patient whose data is being graphed
    @param save a boolean indicating whether to download the graphs locally
    """
    if INTERVAL > 5:
         raise Exception("Data needs to have measurement intervals at most 5 minutes long")

    data = df.loc[id]  
    data.reset_index(inplace=True)

    data[[TIME, GLUCOSE]] = pp.resample_data(data[[TIME, GLUCOSE]])
    times = data[TIME] - data[TIME].dt.normalize()
    # need to be in a DateTime format so seaborn can tell how to scale the x axis labels below
    data["Time"] = (pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times)

    data.set_index("Time", inplace=True)

    agp_data = pd.DataFrame()
    for time, measurements in data.groupby("Time"):
      metrics = {
         "Time": time,
         "5th": measurements[GLUCOSE].quantile(0.05),
         "25th": measurements[GLUCOSE].quantile(0.25),
         "Median": measurements[GLUCOSE].median(),
         "75th": measurements[GLUCOSE].quantile(0.75),
         "95th": measurements[GLUCOSE].quantile(0.95),
      }
      agp_data = pd.concat([agp_data, pd.DataFrame.from_records([metrics])])

    agp_data.sort_values(by=["Time"], inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="5th", x=agp_data["Time"], y=agp_data["5th"], line=dict(color="#869FCE")))
    fig.add_trace(go.Scatter(name="25th", x=agp_data["Time"], y=agp_data["25th"], fill="tonexty", fillcolor="#C9D4E9", line=dict(color="#97A8CB")))
    fig.add_trace(go.Scatter(name="Median",x=agp_data["Time"], y=agp_data["Median"], fill="tonexty", fillcolor="#97A8CB", line=dict(color="#183260")))
    fig.add_trace(go.Scatter(name="75th", x=agp_data["Time"], y=agp_data["75th"], fill="tonexty", fillcolor="#97A8CB", line=dict(color="#97A8CB")))
    fig.add_trace(go.Scatter(name="95th", x=agp_data["Time"], y=agp_data["95th"], fill="tonexty", fillcolor="#C9D4E9", line=dict(color="#869FCE")))

    fig.add_hline(y=70, line_color="green")
    fig.add_hline(y=180, line_color="green")
    fig.update_layout(title={"text": f"AGP Plot for {id}"}, height=height, yaxis_range = [35,405])

    if app: return fig
    fig.show()