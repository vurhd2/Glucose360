from preprocessing import import_data, segment_data
from events import *
from plots import *
from features import create_features
import pandas as pd
import configparser

import plotly.graph_objects as go

from shiny import App, ui, render, reactive
from shiny.types import FileInfo
from shiny.ui import tags
from shinywidgets import output_widget, render_widget

import zipfile
import io

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
         ui.layout_columns(
            ui.card(
               ui.input_text("id_template", "Template for ID Retrieval"),
               ui.input_text("glucose_col", "Name of Glucose Column", "Glucose Value (mg/dL)"),
               ui.input_text("time_col", "Name of Timestamp Column", "Timestamp (YYYY-MM-DDThh:mm:ss)"),
               ui.input_numeric("resample_interval", "Resampling Interval", 5, min=1),
               ui.input_numeric("max_gap", "Maximum Gap for Interpolation", 45),
               ui.input_file("data_import", "Import Dexcom CGM Data (.csv or .zip file)", accept=[".zip", ".csv"], multiple=False),
            ),
            ui.card(ui.input_file("split_data", "Split Data", accept=[".csv"], multiple=False))
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
            ui.output_data_frame("daily_filters"),
            ui.output_ui("plot_height"),
            ui.download_button("download_plot", "Download Currently Displayed Plot")
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
               ui.card(
                  ui.output_ui("ui_event_row"),
                  ui.input_select("edit_event_col", "Column to Edit", [TIME, BEFORE, AFTER, TYPE, DESCRIPTION]),
                  ui.input_text("edit_event_text", "Value to Replace With"),
                  ui.input_action_button("edit_event", "Edit Event")
               ),
               ui.card(
                  ui.output_ui("ui_event_select"),
                  tags.div(
                     tags.div(ui.output_data_frame("event_filters"), style="height: 100%"),
                     ui.output_data_frame("events_table"),
                     style="display: flex;"
                  ),
                  ui.download_button("download_events", "Download Events in Table as .csv file")
               )
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
   daily_events_ref = reactive.Value(pd.DataFrame())
   fig: go.Figure = None

   def notify(message):
      ui.notification_show(message, close_button=True) 

   @reactive.Calc
   def df():
      data_file: list[FileInfo] | None = input.data_import()
      if data_file is None:
         return pd.DataFrame()
      data = import_data(path=data_file[0]["datapath"], name=data_file[0]["name"].split(".")[0],
                         id_template=input.id_template(),
                         glucose=input.glucose_col(), time=input.time_col(),
                         interval=input.resample_interval(), max_gap=input.max_gap(), output=notify)
      split_file: list[FileInfo] | None = input.split_data()
      if split_file is None:
         return data
      return segment_data(split_file[0]["datapath"], data)
   
   @reactive.Effect
   @reactive.event(input.data_import)
   def get_initial_events():
      events_ref.set(get_curated_events(df()).reset_index(drop=True))
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
      patient = input.select_patient_plot()
      lower = 500
      height = 750
      upper = 2000
      if plot_type == "Daily (Time-Series)":
         lower = 750
         height = 3000 
         upper = 4000
         if input.daily_filters_selected_rows() and (events_ref.get().shape[0] != 0): 
            events = daily_events_ref.get()
            dates = pd.to_datetime(events[TIME]).dt.date
            data = df()
            height = max(3000, (dates.value_counts().iloc[0] * 65 * data.loc[patient][TIME].dt.date.unique().size))
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
      if plot_type == "Spaghetti":
         chunk_day = input.spaghetti_chunk_switch() if "spaghetti_chunk_switch" in input else False
         return ui.input_switch("spaghetti_chunk_switch", "Chunk Weekend/Weekday", value=chunk_day)
   
   @reactive.Effect
   @reactive.event(input.daily_filters_selected_rows)
   def filter_daily_events():
      daily_events_ref.set(None)

      if input.daily_filters_selected_rows():
         daily_events = events_ref.get()
         daily_events = daily_events[daily_events[ID] == input.select_patient_plot()]
         daily_events[TIME] = pd.to_datetime(daily_events[TIME])

         filtered_types = pd.Series(daily_events[TYPE].unique()).iloc[list(input.daily_filters_selected_rows())]
         daily_events_ref.set(daily_events[daily_events[TYPE].isin(filtered_types)])

   @render.data_frame
   def daily_filters():
      events = events_ref.get()
      events = events[events[ID] == input.select_patient_plot()]
      table = pd.DataFrame(events.drop_duplicates(subset=[TYPE])[TYPE])
      return render.DataGrid(table, row_selection_mode="multiple")

   @render_widget
   def plot():
      global fig
      plot_type = input.select_plot()
      if plot_type == "Daily (Time-Series)":
         fig = daily_plot(df(), input.select_patient_plot(), input.plot_height_slider(), daily_events_ref.get(), app=True)
      elif plot_type == "Weekly (Time-Series)":
         fig = weekly_plot(df(), input.select_patient_plot(), input.plot_height_slider(), app=True)
      elif plot_type == "Spaghetti":
         fig = spaghetti_plot(df(), input.select_patient_plot(), input.spaghetti_chunk_switch(), input.plot_height_slider(), app=True)
      else:
         fig = AGP_plot(df(), input.select_patient_plot(), input.plot_height_slider(), app=True)
      
      return fig
   
   @render.download(filename="plot.zip")
   def download_plot():
      global fig
      encoded_pdf = fig.to_image(format="pdf", width=1500, height=input.plot_height_slider())
      encoded_html = bytes(fig.to_html(), 'utf-8')
      
      html_stream = io.BytesIO(encoded_html)
      pdf_stream = io.BytesIO(encoded_pdf)

      # Create a byte stream to hold the zip file
      zip_stream = io.BytesIO()

      # Create a zip file and write the HTML and PDF files to it
      with zipfile.ZipFile(zip_stream, 'w', zipfile.ZIP_DEFLATED) as zip_file:
         # Write the HTML file
         zip_file.writestr('plot.html', html_stream.getvalue())
         # Write the PDF file
         zip_file.writestr('plot.pdf', pdf_stream.getvalue())

      # Get the byte-encoded zip file
      zip_stream.seek(0)  # Rewind the stream to the beginning
      yield zip_stream.getvalue()

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
         added_events = pd.concat([events_ref.get(), import_events(path=file[0]["datapath"], name=file[0]["name"].split(".")[0],
                               id=input.select_patient_event(), day_col=input.event_day_col(),
                               time_col=input.event_time_col(), before=input.event_import_before(),
                               after=input.event_import_after(), type=input.event_type())])
         added_events[TIME] = pd.to_datetime(added_events[TIME])
         added_events = added_events.reset_index(drop=True).sort_values(by=[TIME])
         events_ref.set(added_events)
         filtered_events_ref.set(added_events)

   @render.data_frame
   def event_filters():
      events = events_ref.get()
      events = events[events[ID] == input.select_patient_event()]
      table = pd.DataFrame(events.drop_duplicates(subset=[TYPE])[TYPE])
      return render.DataGrid(table, row_selection_mode="multiple")

   @reactive.Effect
   @reactive.event(input.select_patient_event, input.event_filters_selected_rows, input.edit_event)
   def filter_events():
      events = events_ref.get()
      events = events[(events[ID] == input.select_patient_event())]

      filtered_events = events.copy()
      if input.event_filters_selected_rows():
         filtered_types = pd.Series(events[TYPE].unique()).iloc[list(input.event_filters_selected_rows())]
         filtered_events = events[events[TYPE].isin(filtered_types)].copy()
      filtered_events_ref.set(filtered_events)

   @render.data_frame
   def events_table():
      filtered_events = filtered_events_ref.get()
      filtered_events[TIME] = filtered_events[TIME].astype(str)
      table = filtered_events.drop(columns=[ID]).reset_index(drop=True)

      return render.DataGrid(table, row_selection_mode="multiple")
   
   @render.ui
   def ui_event_row():
      return ui.input_numeric("edit_event_row", "Row to Edit", 1, min=1, max=filtered_events_ref.get().shape[0])
   
   @reactive.Effect
   @reactive.event(input.edit_event)
   def edit_event():
      filtered_events = filtered_events_ref.get()
      filtered_events[input.edit_event_col()].iloc[input.edit_event_row()-1] = input.edit_event_text()
      filtered_events_ref.set(filtered_events)

      events = events_ref.get()
      events.loc[filtered_events.index.tolist()] = filtered_events
      events_ref.set(events)

   @render.download(filename="events.csv")
   def download_events():
      filtered_events = filtered_events_ref.get()
      downloaded_events = filtered_events[filtered_events[ID] == input.select_patient_event()].reset_index(drop=True)
      yield downloaded_events.loc[list(input.events_table_selected_rows())].to_csv()
   
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
   
   @render.ui
   def ui_event_select():
      return ui.input_numeric("select_event", "Event to Display", 1, min=1, max=filtered_events_ref.get().shape[0])

   @render.data_frame
   def event_metrics_table():
      if not input.select_event(): raise Exception("Select an event from the table to display relevant metrics.")
      filtered_events = filtered_events_ref.get()
      event = filtered_events[filtered_events[ID] == input.select_patient_event()].iloc[input.select_event()-1]
      return render.DataGrid(event_metrics(df(), event.squeeze()))

   @render_widget
   def display_event_plot():
      if not input.select_event(): 
         empty = go.Figure(go.Scatter(x=pd.Series(), y=pd.Series(), mode="markers"))
         empty.update_layout(title="Select an event from the table to display event plot.")
         return empty
      
      filtered_events = filtered_events_ref.get()
      plot_events = filtered_events
      if not input.event_filters_selected_rows():
         provided_event_types = ["hyper level 0 episode", "hyper level 1 episode", "hyper level 2 episode", "hypo level 1 episode", "hypo level 2 episode", "hyper excursion", "hypo excursion"]
         plot_events = plot_events[~plot_events[TYPE].isin(provided_event_types)]
      id = input.select_patient_event()
      event = filtered_events[filtered_events[ID] == id].iloc[input.select_event()-1].squeeze()
      return event_plot(df(), id, event, plot_events, app=True)

app = App(app_ui, server, debug=False)