from preprocessing import import_data, resample_data
from events import *
from plots import *
from features import create_features
import pandas as pd
import configparser

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from shiny import App, ui, render, reactive
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget

config = configparser.ConfigParser()
config.read('config.ini')
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
INTERVAL = config['variables'].getint('interval')

BEFORE = "Before"
AFTER = "After"
TYPE = "Type"
DESCRIPTION = "Description"

filtered_events = pd.DataFrame()

app_ui = ui.page_fluid(
   ui.navset_pill(
      ui.nav_panel(
         "Features",
         ui.card(
            ui.input_text("glucose_col", "Name of Glucose Column", "Glucose Value (mg/dL)"),
            ui.input_text("time_col", "Name of Timestamp Column", "Timestamp (YYYY-MM-DDThh:mm:ss)"),
            ui.input_numeric("resample_interval", "Resampling Interval", 5, min=1),
            ui.input_numeric("max_gap", "Maximum Gap for Interpolation", 45),
            ui.input_file("data_import", "Import Dexcom CGM Data (.csv or .zip file)", accept=[".zip", ".csv"], multiple=False),
         ),
         ui.card(ui.output_data_frame("features_table"))
      ),
      ui.nav_panel(
         "Plots",
         ui.layout_columns(
            ui.output_ui("patient_plot"),
            ui.input_select("select_plot", "Select Plot:", ["Daily (Time-Series)", "Weekly (Time-Series)", "Spaghetti", "AGP"]),
            ui.output_ui("plot_settings"),
            ui.output_ui("plot_height")
         ),
         ui.card(
            output_widget("plot"),
         )
      ),
      ui.nav_panel(
         "Events",
         ui.output_ui("patient_event"),
         ui.layout_columns(
            ui.card(
               ui.layout_columns(
                  ui.card(
                     "Bulk Import Events",
                     ui.input_text("event_day_col", "Name of Day Column", "Day"),
                     ui.input_text("event_time_col", "Name of Time Column", "Time"),
                     ui.input_numeric("event_import_before", "# of Minutes Before Timestamp to Include", 60, min=0),
                     ui.input_numeric("event_import_after", "# of Minutes After Timestamp to Include", 60, min=0),
                     ui.input_text("event_type", "Type of Events Being Imported", "Cronometer Meal"),
                     ui.input_file("event_import", "Import Events (.zip file)", accept=[".zip", ".csv"], multiple=False)
                  ),
                  ui.card(
                     "Add Singular Event",
                     ui.input_date("add_event_date", "Date"),
                     ui.input_text("add_event_time", "Time"),
                     ui.input_numeric("add_event_before", "# of Minutes Before Time to Include", 60, min=0),
                     ui.input_numeric("add_event_after", "# of Minutes After Time to Include", 60, min=0),
                     ui.input_text("add_event_type", "Type"),
                     ui.input_text("add_event_description", "Description"),
                     ui.input_action_button("add_event_button", "Add Event")
                  )
               )
            ),
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
   events_ref = reactive.Value(pd.DataFrame())
   filtered_events_ref = reactive.Value(pd.DataFrame())

   @reactive.Calc
   def df():
      file: list[FileInfo] | None = input.data_import()
      if file is None:
         return pd.DataFrame()
      return import_data(path=file[0]["datapath"], name=file[0]["name"].split(".")[0],
                         glucose=input.glucose_col(), time=input.time_col(),
                         interval=input.resample_interval(), max_gap=input.max_gap())
   
   @reactive.Effect
   @reactive.event(input.data_import)
   def get_initial_events():
      events_ref.set(get_curated_events(df()))
      filtered_events_ref.set(events_ref.get())

   def daily(data):
      show_events = input.daily_events_switch()
      id = input.select_patient_plot()
      events = events_ref.get()

      data["Day"] = data[TIME].dt.date
      days = data["Day"].astype(str).unique().tolist()

      fig = make_subplots(rows=len(days), cols=2 if show_events else 1, column_widths=[0.66,0.34] if show_events else None,
                          specs=[[{"type":"scatter"}, {"type":"table"}] for i in range(len(days))] if show_events else None,
                          horizontal_spacing=0.01 if show_events else None)
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

         if show_events:
            day_events = events[(events[TIME].dt.date == day) & (events["id"] == id)].sort_values(TIME)
            table_body = [day_events[TIME].dt.time.astype(str).tolist(), day_events[DESCRIPTION].tolist()]
            if day_events.shape[0] != 0:
               fig.add_trace(
                  go.Table(
                     columnwidth=[10,40],
                     header=dict(
                        values = [["<b>Time</b>"], ["<b>Description</b>"]],
                        font=dict(size=11)
                     ),
                     cells=dict(
                        values=table_body,
                        align=['left', 'left'],
                        font=dict(size=10),
                     )
                  ), row=row, col=2
               )

         row += 1

      if show_events:
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
      
      # standardizing axes
      if len(days) > 1:
         offset_before = pd.Timedelta(hours=1, minutes=26)
         offset_after = pd.Timedelta(hours=1, minutes=23)
         fig.update_xaxes(range=[pd.Timestamp(days[0]) - offset_before, pd.Timestamp(days[1]) + offset_after], row=1, col=1)
         fig.update_xaxes(range=[pd.Timestamp(days[-1]) - offset_before, (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)) + offset_after], row=len(days), col=1)
      fig.update_yaxes(range=[min(np.min(data[GLUCOSE]), 60) - 10, max(np.max(data[GLUCOSE]), 180) + 10])

      fig.update_layout(title=f"Daily Plot for {id}", height=input.plot_height_slider(), showlegend=(not show_events))
      return fig

   @render.ui
   def patient_plot():
      return ui.input_select("select_patient_plot", "Select Patient:", df().index.unique().tolist())
   
   @render.ui
   def patient_event():
      return ui.input_select("select_patient_event", "Select Patient:", df().index.unique().tolist())
   
   @render.ui
   def plot_height():
      plot_type = input.select_plot()
      min = 750
      height = 3000
      max = 4000
      if plot_type == "Weekly (Time-Series)":
         min = 500
         height = 750
         max = 2000
      elif plot_type == "Spaghetti": 
         min = 250
         height = 600
         max = 750
      elif plot_type == "AGP":
         min = 600
         max = 2000 
         height = 1000

      return ui.input_slider("plot_height_slider", "Set Plot Height", min, max, height)

   @render.ui
   def plot_settings():
      plot_type = input.select_plot()
      if plot_type == "Daily (Time-Series)":
         events_on = input.daily_events_switch() if "daily_events_switch" in input else False
         return ui.input_switch("daily_events_switch", "Show Events", value=events_on)
      elif plot_type == "Spaghetti":
         chunk_day = input.spaghetti_chunk_switch() if "spaghetti_chunk_switch" in input else False
         return ui.input_switch("spaghetti_chunk_switch", "Chunk Weekend/Weekday", value=chunk_day)

   @render_widget
   def plot():
      plot_type = input.select_plot()
      if plot_type == "Daily (Time-Series)":
         return daily(df().loc[input.select_patient_plot()])
      elif plot_type == "Weekly (Time-Series)":
         return weekly_plot(df(), input.select_patient_plot(), input.plot_height_slider(), app=True)
      elif plot_type == "Spaghetti":
         return spaghetti_plot(df(), input.select_patient_plot(), input.spaghetti_chunk_switch(), input.plot_height_slider(), app=True)
      else:
         return AGP_plot(df(), input.select_patient_plot(), input.plot_height_slider(), app=True)
   
   @render.data_frame
   def features_table():
      return render.DataGrid(create_features(df()).reset_index(names=["Patient"]))
   
   @reactive.Effect
   @reactive.event(input.event_import)
   def bulk_import_events():
      file: list[FileInfo] | None = input.event_import()
      if file is not None:
         events_ref.set(pd.concat([events_ref.get(), import_events(path=file[0]["datapath"], name=file[0]["name"].split(".")[0],
                               id=input.select_patient_event(), day_col=input.event_day_col(),
                               time_col=input.event_time_col(), before=input.event_import_before(),
                               after=input.event_import_after(), type=input.event_type())]).reset_index(drop=True))

   @render.data_frame
   def events_table():
      show_episodes = input.show_episodes()
      show_excursions = input.show_excursions()
      show_meals = input.show_meals()

      events = events_ref.get()
      data = events[events["id"] == input.select_patient_event()].copy()
      data[TIME] = data[TIME].astype(str)

      def filter(s):
         if not show_episodes and 'episode' in s.lower(): return False
         elif not show_excursions and 'excursion' in s.lower(): return False
         elif not show_meals and 'meal' in s.lower(): return False
         return True

      filtered_events_ref.set(data[data[TYPE].apply(filter)])
      return render.DataGrid(filtered_events_ref.get().drop(columns=["id"]), row_selection_mode="single")
   
   @reactive.Effect
   @reactive.event(input.add_event_button)
   def add_event():
      added_event = pd.DataFrame.from_records({
         "id": input.select_patient_event(),
         TIME: str(input.add_event_date()) + " " + input.add_event_time(),
         BEFORE: input.add_event_before(),
         AFTER: input.add_event_after(),
         TYPE: input.add_event_type(),
         DESCRIPTION: input.add_event_description()
      }, index=[0])
      added_event[TIME] = pd.to_datetime(added_event[TIME])
      events_ref.set(pd.concat([events_ref.get(), added_event]).reset_index(drop=True))
   
   @render.data_frame
   def event_metrics_table():
      filtered_events = filtered_events_ref.get()
      event = filtered_events[filtered_events["id"] == input.select_patient_event()].iloc[list(input.events_table_selected_rows())]
      return render.DataGrid(event_metrics(df(), event.squeeze()))

   @render_widget
   def event_plot():
      id = input.select_patient_event()
      data = df().loc[id]

      filtered_events = filtered_events_ref.get()
      event = filtered_events[filtered_events["id"] == id].iloc[list(input.events_table_selected_rows())]
      event[TIME] = pd.to_datetime(event[TIME])
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

      if isinstance(filtered_events, pd.DataFrame):
         event_data = filtered_events[filtered_events["id"] == id] if filtered_events is not None else None
         if event_data is not None:
            event_types = event_data['Type'].unique()
            with open('event_colors.json') as colors_file:
               color_dict = json.load(colors_file)
               colors = list(color_dict.values())
               color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
               for index, row in event_data.iterrows():
                  fig.add_vline(x=pd.to_datetime(row[TIME]), line_dash="dash", line_color=color_map[row['Type']])
                  fig.add_annotation(yref="y domain", x=pd.to_datetime(row[TIME]), y=1, text=row['Type'], showarrow=False)

      elif filtered_events["id"] == id:
         fig.add_vline(x=pd.to_datetime(filtered_events[TIME]), line_dash="dash", line_color=color_map[filtered_events['Type']])
         fig.add_annotation(yref="y domain", x=pd.to_datetime(filtered_events[TIME]), y=1, text=filtered_events['Type'], showarrow=False)

      fig.update_xaxes(type="date", range=[before, after])
      return fig

app = App(app_ui, server, debug=False)