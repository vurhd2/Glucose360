import pandas as pd
import seaborn as sns
import preprocessing as pp
import matplotlib.pyplot as plt
import configparser
import json
from events import retrieve_event_data

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

def daily_plot_all(
    df: pd.DataFrame,
    events: pd.DataFrame = None,
    chunk_day: bool = False,
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
        daily_plot(data, id, events, chunk_day, save)

def daily_plot(
    df: pd.DataFrame,
    id: str,
    events: pd.DataFrame = None,
    chunk_day: bool = False,
    save: bool = False,
):
    """
    Only graphs (and possibly saves) a daily plot for the given patient
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
    @param id   the id of the patient whose data is graphed
    @param events  a DataFrame containing event timeframes for some (or all) of the given patients
    @param chunk_day  a boolean indicating whether to split weekdays and weekends
    @param save a boolean indicating whether to download the graphs locally
    """
    data = df.loc[id]

    data[TIME] = pd.to_datetime(data[TIME])
    data.reset_index(inplace=True)

    plot = sns.relplot(
        data=data,
        kind="line",
        x=TIME,
        y=GLUCOSE,
        col="Day Chunking" if chunk_day else None,
    )
    plot.figure.subplots_adjust(top=0.9)
    plot.figure.suptitle(f"Glucose (mg/dL) vs. Timestamp for {id}")
    plot.figure.set_size_inches(10, 6)

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

    for ax in plot.axes.flat:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

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

def spaghetti_plot_all(df: pd.DataFrame, chunk_day: bool = False, save: bool = False):
    """
    Sequentially produces spaghetti plots for all the given patients
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
    @param chunk_day  a boolean indicating whether to split weekdays and weekends
    @param save a boolean indicating whether to download the graphs locally
    """
    sns.set_theme()
    for id, data in df.groupby("id"):
        spaghetti_plot(data, id, chunk_day, save)

def spaghetti_plot(
    df: pd.DataFrame, id: str, chunk_day: bool = False, save: bool = False
):
    """
    Graphs a spaghetti plot for the given patient
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns
    @param id   the id of the patient whose data should be plotted
    @param chunk_day  a boolean indicating whether to split weekdays and weekends
    @param save a boolean indicating whether to download the graphs locally
    """
    data = df.loc[id]

    data.reset_index(inplace=True)

    # Convert timestamp column to datetime format
    data[TIME] = pd.to_datetime(data[TIME])

    data["Day"] = data[TIME].dt.date

    times = data[TIME] - data[TIME].dt.normalize()
    # need to be in a DateTime format so seaborn can tell how to scale the x axis labels
    data["Time"] = (
        pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times
    )

    data.sort_values(by=[TIME], inplace=True)

    plot = sns.relplot(
        data=data,
        kind="line",
        x="Time",
        y=GLUCOSE,
        hue="Day",
        col="Day Chunking" if chunk_day else None,
    )

    plot.fig.subplots_adjust(top=0.9)
    plot.fig.suptitle(f"Spaghetti Plot for {id}")

    plt.xticks(
        pd.to_datetime([f"1/1/1970T{hour:02d}:00:00" for hour in range(24)]),
        (f"{hour:02d}:00" for hour in range(24)),
    )
    plt.ylim(35, 405)
    for ax in plot.axes.flat:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.show() if not save else plt.savefig("./plots/" + str(id) + "Spaghetti.png", bbox_inches="tight")

def AGP_plot_all(df: pd.DataFrame, save: bool = False):
    """
    Displays (and possibly saves) AGP Plots for each patient in the given DataFrame
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns containing all patient data
    @param save a boolean indicating whether to download the graphs locally
    """
    sns.set_theme()
    for id, data in df.groupby("id"):
        AGP_plot(data, id, save)


def AGP_plot(df: pd.DataFrame, id: str, save: bool = False):
    """
    Displays (and possibly saves) an AGP Plot for only the given patient in the DataFrame
    @param df   a Multiindexed DataFrame grouped by 'id' and containing DateTime and Glucose columns containing all patient data
    @param id   the id of the single patient whose data is being graphed
    @param save a boolean indicating whether to download the graphs locally
    """
    if INTERVAL > 5:
        raise Exception(
            "Data needs to have measurement intervals at most 5 minutes long"
        )

    data = df.loc[id]
    data.reset_index(inplace=True)

    data[[TIME, GLUCOSE]] = pp.resample_data(data[[TIME, GLUCOSE]])
    times = data[TIME] - data[TIME].dt.normalize()
    # need to be in a DateTime format so seaborn can tell how to scale the x axis labels below
    data["Time"] = (
        pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times
    )

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

    agp_data = pd.melt(
        agp_data,
        id_vars=["Time"],
        value_vars=["5th", "25th", "Median", "75th", "95th"],
        var_name="Metric",
        value_name=GLUCOSE,
    )

    agp_data.sort_values(by=["Time"], inplace=True)

    plot = sns.relplot(
        data=agp_data,
        kind="line",
        x="Time",
        y=GLUCOSE,
        hue="Metric",
        hue_order=["95th", "75th", "Median", "25th", "5th"],
        palette=["#869FCE", "#97A8CB", "#183260", "#97A8CB", "#869FCE"],
    )

    plot.fig.subplots_adjust(top=0.9)
    plot.fig.suptitle(f"AGP Plot for {id}")

    plt.xticks(
        pd.to_datetime([f"1/1/1970T{hour:02d}:00:00" for hour in range(24)]),
        (f"{hour:02d}:00" for hour in range(24)),
    )
    plt.xticks(rotation=45)
    plt.ylim(35, 405)

    for ax in plot.axes.flat:
        ax.axhline(70, color="green")
        ax.axhline(180, color="green")

        # shading between lines
        plt.fill_between(
            ax.lines[0].get_xdata(),
            ax.lines[0].get_ydata(),
            ax.lines[1].get_ydata(),
            color="#C9D4E9",
        )
        plt.fill_between(
            ax.lines[1].get_xdata(),
            ax.lines[1].get_ydata(),
            ax.lines[3].get_ydata(),
            color="#97A8CB",
        )
        plt.fill_between(
            ax.lines[3].get_xdata(),
            ax.lines[3].get_ydata(),
            ax.lines[4].get_ydata(),
            color="#C9D4E9",
        )

    plt.show() if not save else plt.savefig("./plots/" + str(id) + "AGP.png", bbox_inches="tight")