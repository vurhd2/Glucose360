from preprocessing import import_data, resample_data
from events import *
from plots import *
from features import create_features
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from shiny import App, ui, render, reactive
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget

#app = Dash(__name__)

TIME = "Timestamp (YYYY-MM-DDThh:mm:ss)"
GLUCOSE = "Glucose Value (mg/dL)"

#df = import_data("datasets")
#events = get_curated_events(df)
#events[TIME] = pd.to_datetime(events[TIME])

# ------------------- SHINY -----------------------
app_ui = ui.page_fluid(
   ui.navset_pill(
      ui.nav_panel(
         "Features",
         ui.input_file("data_import", "Import Dexcom CGM Data (.zip file)", accept=[".zip", ".csv"], multiple=False),
         ui.card(ui.output_data_frame("features_table"))
      ),
      ui.nav_panel(
         "Plots",
         ui.layout_columns(
            ui.output_ui("patient_plot"),
            ui.input_select("select_plot", "Select Plot:", ["Daily (Time-series)", "Spaghetti", "AGP"]),
            ui.output_ui("plot_settings")
         ),
         ui.card(
            output_widget("plot"),
            ui.output_plot("agp_plot")
         )
      ),
      ui.nav_panel(
         "Events",
         ui.output_ui("patient_event"),
         ui.layout_columns(
            ui.card(ui.input_file("event_import", "Import Events (.zip file)", accept=[".zip", ".csv"], multiple=False)),
            ui.card(ui.output_data_frame("event_metrics_table")),
         ),
         ui.layout_columns(
            ui.card(
               ui.input_switch("show_excursions", "Show Excursions", value=True),
               ui.input_switch("show_episodes", "Show Episodes", value=True),
               ui.input_switch("show_meals", "Show Meals", value=True),
               ui.output_data_frame("events_table")
            ),
            ui.card(output_widget("event_plot"))
         ),
      ),
      id="tab"
   )
)

def server(input, output, session):
   @reactive.Calc
   def df():
      file: list[FileInfo] | None = input.data_import()
      if file is None:
         return pd.DataFrame()
      print(file[0]["datapath"]) # /var/folders/bw/0rhyvszj0tjfkmy61rv7lg_w0000gn/T/fileupload-3sfa72lh/tmpyu6zs96y/0.zip
      return import_data(file[0]["datapath"], name=file[0]["name"].split(".")[0])
   
   @reactive.Calc
   def get_events():
      return pd.concat([get_curated_events(df()), import_events("imported_events", "ElizaSutherland")])

   def daily(data):
      data["Day"] = data[TIME].dt.date
      days = data["Day"].astype(str).unique().tolist()
      fig = make_subplots(rows=len(days), cols=1)
      row = 1
      for day, dataset in data.groupby("Day"):
         fig.add_trace(
            go.Scatter(
                  x=dataset[TIME],
                  y=dataset[GLUCOSE],
                  mode='lines+markers',
                  name=str(day)
            ), row=row, col=1
         )
         row += 1

      if input.daily_events_switch():
         id = input.select_patient_plot()
         events = get_events()
         if isinstance(events, pd.DataFrame):
            event_data = events[events["id"] == id] if events is not None else None
            if event_data is not None:
               event_types = event_data['Type'].unique()
               with open('event_colors.json') as colors_file:
                  color_dict = json.load(colors_file)
                  colors = list(color_dict.values())
                  color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
                  for index, row in event_data.iterrows():
                     r = days.index(str(row[TIME].date())) + 1
                     fig.add_shape(go.layout.Shape(
                        type="line", 
                        yref="y domain",
                        x0=pd.to_datetime(row[TIME]), 
                        y0=0,
                        x1=pd.to_datetime(row[TIME]),
                        y1=1,
                        line=dict(color=color_map[row['Type']], width=3)), row=r, col=1)
                     fig.add_annotation(yref="y domain", x=pd.to_datetime(row[TIME]), y=1, text=row['Type'], row=r, col=1, showarrow=False)
         elif events["id"] == id:
            day = str(events[TIME].dt.date)
            fig.add_vline(x=pd.to_datetime(events[TIME]), line_dash="dash", line_color=color_map[events['Type']])
            fig.add_annotation(yref="y domain", x=pd.to_datetime(events[TIME]), y=1, text=events['Type'], showarrow=False)
         
         #for i in range(1, len(days)+1): fig.update_yaxes(range=[-450, 450], row=i, col=1)
      
      fig.update_layout(height=1500)
      return fig

   @render.ui
   def patient_plot():
      return ui.input_select("select_patient_plot", "Select Patient:", df().index.unique().tolist())
   
   @render.ui
   def patient_event():
      return ui.input_select("select_patient_event", "Select Patient:", df().index.unique().tolist())

   @render.ui
   def plot_settings():
      plot_type = input.select_plot()
      if plot_type == "Daily (Time-series)":
         events_on = input.daily_events_switch() if "daily_events_switch" in input else False
         return ui.input_switch("daily_events_switch", "Show Events", value=events_on)
      elif plot_type == "Spaghetti":
         chunk_day = input.spaghetti_chunk_switch() if "spaghetti_chunk_switch" in input else False
         return ui.input_switch("daily_events_switch", "Chunk Weekend/Weekday", value=chunk_day)

   @render_widget
   def plot():
      plot_type = input.select_plot()
      if plot_type != "AGP":
         def spaghetti(data):
            data["Day"] = data[TIME].dt.date
            times = data[TIME] - data[TIME].dt.normalize()
            data["Time"] = (pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times)
            data.sort_values(by=[TIME], inplace=True)
            return px.line(data, x="Time", y=GLUCOSE, color="Day")
         
         plot_type = input.select_plot()
         if plot_type == "Daily (Time-series)":
            return daily(df().loc[input.select_patient_plot()])
         elif plot_type == "Spaghetti":
            return spaghetti(df().loc[input.select_patient_plot()])
   
   @render.plot
   def agp_plot():
      plot_type = input.select_plot()
      if plot_type == "AGP":
         return AGP_plot(df(), input.select_patient_plot())
   
   @render.data_frame
   def features_table():
      return render.DataGrid(create_features(df()).reset_index(names=["Patient"]))
   
   @render.data_frame
   def events_table():
      events = get_events()
      data = events[events["id"] == input.select_patient_event()].copy()
      data.drop(columns=["id"], inplace=True)
      data[TIME] = data[TIME].astype(str)

      return render.DataGrid(data, row_selection_mode="single")
   
   @render.data_frame
   def event_metrics_table():
      events = get_events()
      event = events[events["id"] == input.select_patient_event()].iloc[list(input.events_table_selected_rows())]
      return render.DataGrid(event_metrics(df(), event.squeeze()))

   @render_widget
   def event_plot():
      id = input.select_patient_event()
      data = df().loc[id]

      events = get_events()
      event = events[events["id"] == id].iloc[list(input.events_table_selected_rows())]
      before = event[TIME].iloc[0] - pd.Timedelta(minutes=event[BEFORE].iloc[0])
      after = event[TIME].iloc[0] + pd.Timedelta(minutes=event[AFTER].iloc[0])

      subplot_figs = []
      data["Day"] = data[TIME].dt.date
      for day, dataset in data.groupby("Day"):
         subplot_figs.append(
            go.Scatter(
                  x=dataset[TIME],
                  y=dataset[GLUCOSE],
                  mode='lines+markers',
                  name=str(day)
            )
         )
      fig = go.Figure(data=subplot_figs, layout=go.Layout(title='Daily (Time-Series) Plot'))

      events = get_events()
      if isinstance(events, pd.DataFrame):
         event_data = events[events["id"] == id] if events is not None else None
         if event_data is not None:
            event_types = event_data['Type'].unique()
            with open('event_colors.json') as colors_file:
               color_dict = json.load(colors_file)
               colors = list(color_dict.values())
               color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
               for index, row in event_data.iterrows():
                  fig.add_vline(x=pd.to_datetime(row[TIME]), line_dash="dash", line_color=color_map[row['Type']])
                  fig.add_annotation(yref="y domain", x=pd.to_datetime(row[TIME]), y=1, text=row['Type'], showarrow=False)

      elif events["id"] == id:
         fig.add_vline(x=pd.to_datetime(events[TIME]), line_dash="dash", line_color=color_map[events['Type']])
         fig.add_annotation(yref="y domain", x=pd.to_datetime(events[TIME]), y=1, text=events['Type'], showarrow=False)

      fig.update_xaxes(type="date", range=[before, after])
      return fig

app = App(app_ui, server, debug=True)