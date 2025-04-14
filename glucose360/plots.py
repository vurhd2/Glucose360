import pandas as pd
import numpy as np
import glucose360.preprocessing as pp
import configparser
import json
from importlib import resources
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from glucose360.preprocessing import load_config
from glucose360.features import percent_time_in_range, percent_time_in_tight_range, mean, CV, GMI
from pathlib import Path

# Initialize config at module level
config = load_config()
INTERVAL = int(config["variables"]["interval"])
ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
BEFORE = config['variables']['before']
AFTER = config['variables']['after']
TYPE = config['variables']['type']
DESCRIPTION = config['variables']['description']

def daily_plot_all(
    df: pd.DataFrame,
    events: pd.DataFrame = None,
    save: str = None,
    height: int = 2000
):
    """Graphs daily plots for all of the patients in the given DataFrame. 
    Also saves these plots as PDFs and HTMLs if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param events: a DataFrame containing any events to mark on the daily plots, defaults to None
    :type events: 'pandas.DataFrame', optional
    :param save: path of the location where the saved PDF and HTML versions of the plots are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot(s), defaults to 2000
    :type height: int, optional
    """
    for id, data in df.groupby(ID):
        daily_plot(data, id, height, events, save)

def daily_plot(
   df: pd.DataFrame,
   id: str,
   height: int = 2000,
   events: pd.DataFrame = None,
   save: bool = False,
   app: bool = False
):
   """Graphs a daily (time-series) plot for the given patient within the given DataFrame. 
    Also saves this plot as both a PDF and HTML file if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param id: the identification of the patient whose CGM data to plot within the given DataFrame
    :type id: str
    :param events: a DataFrame containing any events to mark on the daily plots, defaults to None
    :type events: 'pandas.DataFrame', optional
    :param save: path of the location where the saved PDF and HTML versions of the plot are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot, defaults to 2000
    :type height: int, optional
    :param app: boolean indicating whether to return the Plotly figure instead of rendering it (used mainly within the web application), defaults to False
    :type app: bool, optional
    :return: None if app is False, otherwise the Plotly figure
    :rtype: 'plotly.graph_objects.Figure' | None
   """
   show_events = not (events is None or events[events[ID] == id].empty)
   data = df.loc[id].copy()
   data[TIME] = pd.to_datetime(data[TIME])
   data["Day"] = data[TIME].dt.date

   days = data["Day"].unique().astype(str).tolist()
   offset = pd.Timedelta(hours=1, minutes=30)

   rendered_types = []
   if show_events:
      event_types = events[TYPE].unique()
      colors_path = Path(__file__).parent / 'event_colors.json'
      with open(colors_path) as colors_file:
         color_dict = json.load(colors_file)
         colors = list(color_dict.values())
         color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}

   fig = make_subplots(rows=len(days), cols=2 if show_events else 1, column_widths=[0.66,0.34] if show_events else None,
                        specs=[[{"type":"scatter"}, {"type":"table"}] for _ in range(len(days))] if show_events else None,
                        horizontal_spacing=0.01 if show_events else None,
                        vertical_spacing=0.05 if show_events else None)

   num_events = 0

   for idx, (day, dataset) in enumerate(data.groupby("Day"), start=1):
      fig.add_trace(go.Scatter(x=dataset[TIME].apply(lambda dt: dt.isoformat()),y=dataset[GLUCOSE],mode='lines+markers',name=str(day), showlegend=False), row=idx, col=1)
      fig.update_xaxes(
          range=[pd.Timestamp(day) - offset, pd.Timestamp(day) + pd.Timedelta(days=1) + offset],
          row=idx, col=1,
          tickfont=dict(size=20),
          title=dict(font=dict(size=35))
      )
      fig.update_yaxes(
          title=dict(text="Glucose Value (mg/dL)"),
          tickfont=dict(size=20),
          row=idx,
          col=1
      )

      if show_events:
         day_events = events[(events[TIME].dt.date == day) & (events[ID] == id)].sort_values(TIME)
         num_events = max(num_events, day_events.shape[0])
         if not day_events.empty:
            table_body = [day_events[TIME].dt.time.astype(str).tolist(), day_events[DESCRIPTION].tolist()]
            fig.add_trace(
               go.Table(
                  columnwidth=[25,25],
                  header=dict(values = [["<b>Time</b>"], ["<b>Description</b>"]],font=dict(size=22)),
                  cells=dict(values=table_body,align=['left', 'left'],font=dict(size=20))
               ), row=idx, col=2
            )

            for _, row in day_events.drop_duplicates(subset=[TIME, TYPE]).iterrows():
               already_rendered = (row[TYPE] in rendered_types)
               if (not already_rendered): rendered_types.append(row[TYPE])
               fig.add_shape(go.layout.Shape(
                  type="line", yref="y domain",
                  x0=pd.to_datetime(row[TIME]), y0=0,
                  x1=pd.to_datetime(row[TIME]), y1=1,
                  line_color=color_map[row[TYPE]], line_dash="dash",
                  name=row[TYPE], legendgroup=row[TYPE], showlegend=(not already_rendered)), row=idx, col=1)

   fig.update_yaxes(range=[min(np.min(data[GLUCOSE]), 60) - 10, max(np.max(data[GLUCOSE]), 180) + 10])
   fig.update_layout(
       title=dict(text=f"Daily Plot for {id}", font=dict(size=40)),
       height=height,
       showlegend=False
   )

   if save: 
      path = os.path.join(save, f"{id}_daily_plot")
      fig.write_image(path + ".pdf", width=1500, height=height)
      fig.write_image(path + ".png", width=1500, height=height)
      fig.write_image(path + ".jpeg", width=1500, height=height)
      fig.write_html(path +".html")

   if app: return fig
   fig.show()

def event_plot_all(df: pd.DataFrame, id: str, events: pd.DataFrame, type: str, save: bool = False):
    """Graphs all event plots of a certain type for the given patient within the given DataFrame. 

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param id: the identification of the patient whose CGM data to plot within the given DataFrame
    :type id: str
    :param events: a DataFrame containing events to be plotted, defaults to None
    :type events: 'pandas.DataFrame'
    :param type: the type of events to plot
    :type type: str
    """
    relevant_events = events[(events[ID] == id) & (events[TYPE] == type)]
    for index, event in relevant_events.iterrows():
        event_plot(df, id, event, relevant_events, save=save)

def event_plot(df: pd.DataFrame, id: str, event: pd.Series, events: pd.DataFrame = None, save: bool = False, app: bool = False):
   """Graphs an event plot for the given patient within the given DataFrame. 

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param id: the identification of the patient whose CGM data to plot within the given DataFrame
    :type id: str
    :param event: the event to be displayed
    :type event: 'pandas.Series'
    :param events: a DataFrame containing any extra events to be marked within the event plot, defaults to None
    :type events: 'pandas.DataFrame', optional
    :param save: boolean indicating whether to save the plot as a file, defaults to False
    :type save: bool, optional
    :param app: boolean indicating whether to return the Plotly figure instead of rendering it (used mainly within the web application), defaults to False
    :type app: bool, optional
    :return: None if app is False, otherwise the Plotly figure
    :rtype: 'plotly.graph_objects.Figure' | None
    """
   if event[ID] != id: raise Exception("Given event does not match the 'id' given.")
   data = df.loc[id].copy()
   event = event.copy()
   event[TIME] = pd.to_datetime(event[TIME])
   before = event[TIME] - pd.Timedelta(minutes=event[BEFORE])
   after = event[TIME] + pd.Timedelta(minutes=event[AFTER])

   data["Day"] = data[TIME].dt.date
   subplot_figs = [go.Scatter(x=dataset[TIME].apply(lambda dt: dt.isoformat()),y=dataset[GLUCOSE],mode='lines+markers', name=str(day)) for day, dataset in data.groupby("Day")]
   fig = go.Figure(data=subplot_figs, layout=go.Layout(title=dict(text=f"Event Plot for {id}", font=dict(size=40)), legend=dict(font=dict(size=20))))

   event_data = events[events[ID] == id] if events is not None else pd.DataFrame()
   if not event_data.empty: create_event_lines(fig, event_data)

   fig.update_xaxes(type="date", range=[before, after], tickfont=dict(size=20), title=dict(font=dict(size=35)))
   fig.update_yaxes(title_text="Glucose Value (mg/dL)", tickfont=dict(size=20), title=dict(font=dict(size=35)))

   if save: 
      path = os.path.join(save, f"{id}_event_plot")
      fig.write_image(path + ".png", width=1500, height=1000)
      fig.write_html(path +".html")

   if app: return fig
   fig.show()

def create_event_lines(fig: go.Figure, events: pd.DataFrame):
   """Marks vertical lines within the given plotly Figure for the given events

   :param fig: the plotly figure to mark vertical lines in
   :type fig: plotly.graph_objects.Figure
   :param events: Pandas DataFrame containing vevents to mark
   :type events: pandas.DataFrame''
   """
   event_types = events[TYPE].unique()
   events[TIME] = pd.to_datetime(events[TIME])
   colors_path = Path(__file__).parent / 'event_colors.json'
   with open(colors_path) as colors_file:
      color_dict = json.load(colors_file)
      colors = list(color_dict.values())
      color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}

      rendered_types = []
      for event in events.itertuples():
         time = getattr(event, TIME)
         type = getattr(event, TYPE)
         already_rendered = (type in rendered_types) 
         if (not already_rendered): rendered_types.append(type)
         fig.add_vline(x=time, line_width=3, line_dash="dash", line_color=color_map[type], name=type, legendgroup=type, showlegend=(not already_rendered))

def weekly_plot_all(df: pd.DataFrame, save: str = None, height: int = 1000):
   """Graphs weekly plots for all of the patients within the given DataFrame. 
    Also saves these plots as PDF and HTML files if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param save: path of the location where the saved PDF and HTML versions of the plots are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot(s), defaults to 1000
    :type height: int, optional
   """
   for id, data in df.groupby(ID):
      weekly_plot(data, id, height)

def weekly_plot(df: pd.DataFrame, id: str, save: str = None, height: int = 1000, app = False):
   """Graphs a weekly (time-series) plot for the given patient within the given DataFrame. 
    Also saves this plot as a PDF and HTML if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param id: the identification of the patient whose CGM data to plot within the given DataFrame
    :type id: str
    :param save: path of the location where the saved PDF and HTML versions of the plots are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot, defaults to 1000
    :type height: int, optional
    :param app: boolean indicating whether to return the Plotly figure instead of rendering it (used mainly within the web application), defaults to False
    :type app: bool, optional
    :return: None if app is False, otherwise the Plotly figure
    :rtype: 'plotly.graph_objects.Figure' | None
   """
   data = df.loc[id].reset_index().copy()
   data.set_index(TIME, inplace=True)

   weekly_data = data.groupby(pd.Grouper(freq='W'))
   weekly_dfs = [group for _, group in weekly_data]
   
   fig = make_subplots(rows=len(weekly_dfs), cols=1)
   for week_index in range(len(weekly_dfs)):
      week = weekly_dfs[week_index].reset_index()
      fig.add_trace(
         go.Scatter(
               x=week[TIME].apply(lambda dt: dt.isoformat()),
               y=week[GLUCOSE],
               mode='lines+markers',
         ), row=(week_index+1), col=1
      )
      fig.update_yaxes(title_text="Glucose Value (mg/dL)", row=(week_index+1), col=1)
   
   if len(weekly_dfs) > 1:
      offset_before = pd.Timedelta(hours=10)
      offset_after = pd.Timedelta(hours=10)
      first_end = pd.Timestamp(weekly_dfs[1].reset_index()[TIME].dt.date.iloc[0])
      first_start = first_end - pd.Timedelta(weeks=1)
      last_start = pd.Timestamp(weekly_dfs[-1].reset_index()[TIME].dt.date.iloc[0])
      last_end = last_start + pd.Timedelta(weeks=1)
      fig.update_xaxes(range=[first_start - offset_before, first_end + offset_after], row=1, col=1)
      fig.update_xaxes(range=[last_start - offset_before, last_end + offset_after], row=len(weekly_dfs), col=1)
   fig.update_yaxes(range=[min(np.min(data[GLUCOSE]), 60) - 10, max(np.max(data[GLUCOSE]), 180) + 10], 
                   tickfont=dict(size=30), 
                   title=dict(text="Glucose Value<br>(mg/dL)", font=dict(size=35)))
   fig.update_xaxes(tickformat="%B %d, %Y <br> (%a)", tickfont=dict(size=30), title=dict(font=dict(size=35)))

   fig.update_layout(title=dict(text=f"Weekly Plot for {id}", font=dict(size=30)), height=height, showlegend=False)

   if save: 
      path = os.path.join(save, f"{id}_weekly_plot")
      fig.write_image(path + ".pdf", width=1500, height=height)
      fig.write_image(path + ".png", width=1500, height=height)
      fig.write_image(path + ".jpeg", width=1500, height=height)
      fig.write_html(path +".html")

   if app: return fig
   fig.show()

def spaghetti_plot_all(df: pd.DataFrame, chunk_day: bool = False, save: str = None, height: int = 600):
    """Graphs spaghetti plots for all patients within the given DataFrame. 
    Also saves these plots as PDF and HTML files if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param chunk_day: boolean indicating whether to create separate subplots based on whether the data occurred on a weekday or during the weekend
    :type chunk_day: bool, optional
    :param save: path of the location where the saved PDF and HTML versions of the plots are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot(s), defaults to 600
    :type height: int, optional
   """
    for id, data in df.groupby(ID):
        spaghetti_plot(df=data, id=id, chunk_day=chunk_day, save=save, height=height)

def spaghetti_plot(df: pd.DataFrame, id: str, chunk_day: bool = False, save: str = None, height: int = 600, app=False):
    """Graphs a spaghetti plot for the given patient within the given DataFrame. 
    Also saves this plot as both a PDF and HTML file if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param id: the identification of the patient whose CGM data to plot within the given DataFrame
    :type id: str
    :param chunk_day: boolean indicating whether to create separate subplots based on whether the data occurred on a weekday or during the weekend
    :type chunk_day: bool, optional
    :param save: path of the location where the saved PDF and HTML versions of the plots are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot(s), defaults to 600
    :type height: int, optional
    :param app: boolean indicating whether to return the Plotly figure instead of rendering it (used mainly within the web application), defaults to False
    :type app: bool, optional
    :return: None if app is False, otherwise the Plotly figure
    :rtype: 'plotly.graph_objects.Figure' | None
   """
    data = df.loc[id].copy()
    data["Day"] = data[TIME].dt.date
    times = data[TIME] - data[TIME].dt.normalize()
    data["Time"] = (pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times)
    data["Time"] = data["Time"].apply(lambda dt: dt.isoformat())
    data.sort_values(by=[TIME], inplace=True)

    fig = px.line(data, x="Time", y=GLUCOSE, color="Day", title=f"Spaghetti Plot for {id}", height=height, facet_col="Day Chunking" if chunk_day else None)
    fig.update_xaxes(
        tickformat="%H:%M:%S",
        title_text="Time",
        tickfont_size=20,
        title=dict(font=dict(size=35))
    )
    fig.update_yaxes(
        title_text="Glucose Value (mg/dL)",
        tickfont_size=20,
        title=dict(font=dict(size=35)),
        row=1,
        col=1
    )
    fig.update_annotations(font_size=35)
    fig.update_layout(title=dict(font=dict(size=40)), legend=dict(font=dict(size=30)))

    if save: 
      path = os.path.join(save, f"{id}_spaghetti_plot")
      fig.write_image(path + ".pdf", width=1500, height=height)
      fig.write_image(path + ".png", width=1500, height=height)
      fig.write_image(path + ".jpeg", width=1500, height=height)
      fig.write_html(path +".html")

    if app: return fig
    fig.show()

def AGP_plot_all(df: pd.DataFrame, height: int = 600, save: str = None):
    """Graphs AGP-report style plots for all of the patients within the given DataFrame. 
    Also saves these plots as both PDF and HTML files if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param save: path of the location where the saved PDF and HTML versions of the plot are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot, defaults to 600
    :type height: int, optional
   """
    for id, data in df.groupby(ID):
        AGP_plot(data, id, height)


def AGP_plot(df: pd.DataFrame, id: str, save: str = None, height: int = 600, app=False):
    """Graphs AGP-report style plots for the given patient within the given DataFrame. 
    Also saves this plot as both a PDF and HTML file if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to plot
    :type df: 'pandas.DataFrame'
    :param id: the identification of the patient whose CGM data to plot within the given DataFrame
    :type id: str
    :param save: path of the location where the saved PDF and HTML versions of the plot are saved, defaults to None
    :type save: str, optional 
    :param height: the height (in pixels) of the resulting plot, defaults to 600
    :type height: int, optional
    :param app: boolean indicating whether to return the Plotly figure instead of rendering it (used mainly within the web application), defaults to False
    :type app: bool, optional
    :return: None if app is False, otherwise the Plotly figure
    :rtype: 'plotly.graph_objects.Figure' | None
   """
    config.read('config.ini')
    interval = int(config["variables"]["interval"])
    if interval > 5:
         raise Exception("Data needs to have measurement intervals at most 5 minutes long")

    data = df.loc[id].copy()  
    data.reset_index(inplace=True)

    data[[TIME, GLUCOSE, ID]] = pp._resample_data(data[[TIME, GLUCOSE, ID]])
    times = data[TIME] - data[TIME].dt.normalize()
    # need to be in a DateTime format so plots can tell how to scale the x axis labels below
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
    agp_data["Time"] = agp_data["Time"].apply(lambda dt: dt.isoformat())

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="5th", x=agp_data["Time"], y=agp_data["5th"], line=dict(color="#869FCE")))
    fig.add_trace(go.Scatter(name="25th", x=agp_data["Time"], y=agp_data["25th"], fill="tonexty", fillcolor="#C9D4E9", line=dict(color="#97A8CB")))
    fig.add_trace(go.Scatter(name="Median",x=agp_data["Time"], y=agp_data["Median"], fill="tonexty", fillcolor="#97A8CB", line=dict(color="#183260")))
    fig.add_trace(go.Scatter(name="75th", x=agp_data["Time"], y=agp_data["75th"], fill="tonexty", fillcolor="#97A8CB", line=dict(color="#97A8CB")))
    fig.add_trace(go.Scatter(name="95th", x=agp_data["Time"], y=agp_data["95th"], fill="tonexty", fillcolor="#C9D4E9", line=dict(color="#869FCE")))

    fig.add_hline(y=70, line_color="lime")
    fig.add_hline(y=140, line_color="lime")
    fig.add_hline(y=180, line_color="green")
    fig.update_layout(title={"text": f"AGP Plot for {id}", "font": {"size":30}}, height=height, yaxis_range = [35,405], 
                     legend=dict(font=dict(size=40), itemsizing='constant'))
    fig.update_xaxes(tickformat="%H:%M:%S", title_text="Time", tickfont_size=30, title_font_size=40)
    fig.update_yaxes(title_text="Glucose Value<br>(mg/dL)", tickfont_size=30, title_font_size=35)

    if app: return fig
    fig.show()

def AGP_report(df: pd.DataFrame, id: str, path: str = None):
   """Creates an AGP-report for the given patient within the given DataFrame. 
    Also saves this plot as a PDF file if passed a valid path.

    :param df: the DataFrame (following package guidelines) containing the CGM data to report on
    :type df: 'pandas.DataFrame'
    :param id: the identification of the patient whose CGM data to report on
    :type id: str
    :param path: path of the location where the saved PDF version of the plot is saved, defaults to None
    :type path: str, optional 
    :return: the AGP-report in string form if path is False, otherwise None
    :rtype: str | None
   """
   fig = make_subplots(rows = 1, cols = 2, specs=[[{"type": "table"}, {"type": "bar"}]])
   
   patient_data = df.loc[id]
   TIR = {"< 54 mg/dL": percent_time_in_range(patient_data, 0, 53),
                       "54 - 69 mg/dL": percent_time_in_range(patient_data, 54, 69),
                       "70 - 140 mg/dL": percent_time_in_tight_range(patient_data),
                       "141 - 180 mg/dL": percent_time_in_range(patient_data, 141, 180),
                       "181 - 250 mg/dL": percent_time_in_range(patient_data, 181, 250),
                       "> 250 mg/dL": percent_time_in_range(patient_data, 251, 400)}
   
   COLORS = {"< 54 mg/dL": "rgba(151,34,35,255)",
                       "54 - 69 mg/dL": "rgba(229,43,23,255)",
                       "70 - 140 mg/dL": "#00ff00",
                       "141 - 180 mg/dL": "rgba(82,173,79,255)",
                       "181 - 250 mg/dL": "rgba(250,192,3,255)",
                       "> 250 mg/dL": "rgba(241,136,64,255)"}
   
   for key, value in TIR.items():
      fig.add_trace(go.Bar(name=key, x=["TIR"], y=[value], 
                          marker=dict(color=COLORS[key]), 
                          text=[round(value, 2)], 
                          textposition="inside",
                          textfont=dict(size=35)), row=1, col=2)
   fig.update_layout(barmode='stack', height=800, font=dict(size=35),
                    legend=dict(font=dict(size=35)))
   
   fig.update_xaxes(title_text="Time in Range", title_font=dict(size=35), tickfont=dict(size=35), row=1, col=2)
   fig.update_yaxes(title_text="Percentage (%)", title_font=dict(size=35), tickfont=dict(size=35), row=1, col=2)
   
   ave_glucose = mean(patient_data)
   gmi = GMI(patient_data)
   cv = CV(patient_data)

   days = patient_data[TIME].dt.date
   table_body = [["ID:", f"{len(days.unique())} Days:", "Average Glucose (mg/dL):", "Glucose Management Indicator:", "Glucose Variability:"], 
                 [id, f"{days.iloc[0]} to {days.iloc[-1]}", str(round(ave_glucose, 2)), str(round(gmi, 2)), str(round(cv, 2))]]
   fig.add_trace(go.Table(cells=dict(values=table_body,align=['left', 'center'],font=dict(size=35),height=100)), row=1, col=1)
   
   agp = AGP_plot(df, id, app=True)
   weekly = weekly_plot(df, id, height=1000, app=True)
   weekly.update_layout(margin=dict(l=150))
   weekly.update_yaxes(title=dict(text="Glucose Value<br>(mg/dL)", font=dict(size=35)))

   header_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
   agp_html = agp.to_html(full_html=False, include_plotlyjs=False)
   weekly_html = weekly.to_html(full_html=False, include_plotlyjs=False)

   html_template = f"""
   <!DOCTYPE html>
   <html>
   <head>
      <title>AGP Report for {id}</title>
   </head>
   <body>
      <h1 style="font-size: 48px; text-align: center;"> AGP Report: Continuous Glucose Monitoring </h1>
      {header_html}
      {agp_html}
      {weekly_html}
   </body>
   </html>
   """

   if path:
      with open(os.path.join(path, f"{id}_AGP_Report.html"), "w") as f:
         f.write(html_template)

   return html_template