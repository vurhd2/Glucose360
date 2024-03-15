from preprocessing import import_data
from events import *
from plots import *
from features import create_features
import pandas as pd
import configparser

import plotly.graph_objects as go

from shiny import App, ui, render, reactive
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget

config = configparser.ConfigParser()
config.read('config.ini')
ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
BEFORE = config['variables']['before']
AFTER = config['variables']['after']
TYPE = config['variables']['type']
DESCRIPTION = config['variables']['description']

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
         ui.card(
            ui.output_data_frame("features_table"),
            ui.download_button("download_features", "Download Features as .csv file")
         )
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
            ui.card(output_widget("display_event_plot"))
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

   @render.ui
   def patient_plot():
      return ui.input_select("select_patient_plot", "Select Patient:", df().index.unique().tolist())
   
   @render.ui
   def patient_event():
      return ui.input_select("select_patient_event", "Select Patient:", df().index.unique().tolist())
   
   @render.ui
   def plot_height():
      plot_type = input.select_plot()
      lower = 500
      height = 750
      upper = 2000
      if plot_type == "Daily (Time-Series)":
         lower = 750
         height = 3000 
         upper = 4000
         if input.daily_events_switch() and (events_ref.get().shape[0] != 0): 
            dates = events_ref.get()[TIME].dt.date
            data = df()
            height = max(height, (dates.value_counts().iloc[0] * 60 * data[data[ID] == input.select_patient_plot()][TIME].dt.date.unique().size))
            upper = height
      elif plot_type == "Spaghetti": 
         lower = 250
         height = 600
         upper = 750
      elif plot_type == "AGP":
         lower = 600
         upper = 2000 
         height = 1000

      return ui.input_slider("plot_height_slider", "Set Plot Height", lower, upper, height)

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
         daily_events = events_ref.get() if input.daily_events_switch() else None
         return daily_plot(df(), input.select_patient_plot(), input.plot_height_slider(), daily_events, app=True)
      elif plot_type == "Weekly (Time-Series)":
         return weekly_plot(df(), input.select_patient_plot(), input.plot_height_slider(), app=True)
      elif plot_type == "Spaghetti":
         return spaghetti_plot(df(), input.select_patient_plot(), input.spaghetti_chunk_switch(), input.plot_height_slider(), app=True)
      else:
         return AGP_plot(df(), input.select_patient_plot(), input.plot_height_slider(), app=True)
   
   @render.data_frame
   def features_table():
      if df().shape[0] == 0: raise Exception("Please upload your CGM data above.")
      return render.DataGrid(create_features(df()).reset_index(names=["Patient"]))
   
   @render.download(filename="features.csv")
   def download_features():
      if df().shape[0] == 0: raise Exception("Please upload your CGM data above.")
      yield create_features(df()).reset_index(names=["Patient"]).to_csv()
   
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
      data = events[events[ID] == input.select_patient_event()].copy()
      data[TIME] = data[TIME].astype(str)

      def filter(s):
         if not show_episodes and 'episode' in s.lower(): return False
         elif not show_excursions and 'excursion' in s.lower(): return False
         elif not show_meals and 'meal' in s.lower(): return False
         return True

      filtered_events_ref.set(data[data[TYPE].apply(filter)])
      return render.DataGrid(filtered_events_ref.get().drop(columns=[ID]), row_selection_mode="single")
   
   @reactive.Effect
   @reactive.event(input.add_event_button)
   def add_event():
      added_event = pd.DataFrame.from_records({
         ID: input.select_patient_event(),
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
      if not input.events_table_selected_rows(): raise Exception("Select an event from the table to display relevant metrics.")
      filtered_events = filtered_events_ref.get()
      event = filtered_events[filtered_events[ID] == input.select_patient_event()].iloc[list(input.events_table_selected_rows())]
      return render.DataGrid(event_metrics(df(), event.squeeze()))

   @render_widget
   def display_event_plot():
      if not input.events_table_selected_rows(): 
         empty = go.Figure(go.Scatter(x=pd.Series(), y=pd.Series(), mode="markers"))
         empty.update_layout(title="Select an event from the table to display event plot.")
         return empty
      filtered_events = filtered_events_ref.get()
      id = input.select_patient_event()
      event = filtered_events[filtered_events[ID] == id].iloc[list(input.events_table_selected_rows())].squeeze()
      return event_plot(df(), id, event, events_ref.get(), app=True)

app = App(app_ui, server, debug=False)