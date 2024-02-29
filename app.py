from preprocessing import import_directory, resample_data
from events import *
from plots import *
from features import create_features
import pandas as pd
import seaborn as sns

from dash import Dash, html, dcc, dash_table, callback, Output, Input
import plotly.express as px
import plotly.graph_objs as go

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget

#app = Dash(__name__)

TIME = "Timestamp (YYYY-MM-DDThh:mm:ss)"
GLUCOSE = "Glucose Value (mg/dL)"

df = import_directory("datasets")
#features = create_features(df).T.reset_index(names=['Metric'])

# ------------------- SHINY -----------------------
app_ui = ui.page_fluid(
   ui.navset_pill(
      ui.nav_panel(
         "Features",
         ui.input_file("file", "Import Dexcom CGM Data (.zip file)", accept=[".zip"], multiple=False),
         ui.output_data_frame("features_table")
      ),
      ui.nav_panel(
         "Plots",
         ui.input_select("select_patient_plot", "Select Patient:", df.index.unique().tolist()),
         ui.input_select("select_plot", "Select Plot:", ["Daily (Time-series)", "Spaghetti", "AGP"]),
         output_widget("plot"),
         ui.output_text_verbatim("plot_hover_x"),
         ui.output_text_verbatim("plot_hover_y")
      ),
      ui.nav_panel("Events"),
      id="tab"
   )
)

def server(input, output, session):
   @render_widget
   def plot():
      def daily(data):
         #fig = px.line(data, x=TIME, y=GLUCOSE, title=f"Daily (Time-Series) for {id}")
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
         #return {'data': subplot_figs, 'layout': go.Layout(title='Dynamic Subplots')}
         #layout = go.Layout(title='Daily (Time-Series)', xaxis=dict(title='Time'), yaxis=dict(title='Glucose'))
         return go.Figure(data=subplot_figs, layout=go.Layout(title='Daily (Time-Series) Plot'))
      
      def spaghetti(data):
         data["Day"] = data[TIME].dt.date
         times = data[TIME] - data[TIME].dt.normalize()
         data["Time"] = (pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times)
         data.sort_values(by=[TIME], inplace=True)
         return px.line(data, x="Time", y=GLUCOSE, color="Day")
      
      plot_type = input.select_plot()
      if plot_type == "Daily (Time-series)":
         return daily(df.loc[input.select_patient_plot()])
      elif plot_type == "Spaghetti":
         return spaghetti(df.loc[input.select_patient_plot()])
      else:
         return AGP_plot(df, input.select_patient_plot())

   @render.text
   def plot_hover_x():
      return f"Timestamp: {input.plot_hover()['x']}"
   
   @render.text
   def plot_hover_y():
      return f"Glucose Value: {input.plot_hover()['y']}"
   
   @render.data_frame
   def features_table():
      return render.DataTable(create_features(df).reset_index(names=["Patient"]))

app = App(app_ui, server, debug=True)

"""
with ui.navset_pill(id="tab"):
   with ui.nav_panel("Features"):
      ui.div
      "hub"

   with ui.nav_panel("Plots"):
      ui.input_select("select_patient_plot", "Select Patient:", df.index.unique().tolist())
      ui.input_select("select_plot", "Select Plot:", ["Daily (Time-series)", "Spaghetti", "AGP"])

      @render.plot
      def plot():
         plot_type = input.select_plot()
         plot = sns.relplot(data=df.loc[input.select_patient_plot()],kind="line",x=TIME, y=GLUCOSE)

   with ui.nav_panel("Events"):
      "hub"

@render.plot
def plot():
   plot_type = input.select_plot()
   daily_plot(df, input.select_patient_plot())
"""

"""
# ------------------------ DASH ------------------------
app.layout = html.Div([
   dcc.Tabs([
      dcc.Tab(label='Features', children=[
         dash_table.DataTable(
            data=features.to_dict('records'),
            columns=[{"name": i, "id": i} for i in features.columns]
         )
      ]),
      dcc.Tab(label='Plots', children=[
         dcc.Dropdown(df.index.unique().tolist(), id='plot-id-dropdown'),
         dcc.Dropdown(['Daily (Time-series)', 'Spaghetti', 'AGP'], 'Daily (Time-series)', id='plot-dropdown'),
         html.Div(id='plot')
      ]),
      dcc.Tab(label='Events', children=[

      ])
   ])
])

@callback (Output("plot", "children"),
           Input("plot-id-dropdown", "value"),
           Input("plot-dropdown", "value"))
def render_plot(id, plot_type):
   data = df.loc[id].copy()

   #fig = px.line(data, x=TIME, y=GLUCOSE, title=f"Daily (Time-Series) for {id}")
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

   fig = {'data': subplot_figs, 'layout': go.Layout(title='Dynamic Subplots')}

   if plot_type == "Spaghetti":
      data["Day"] = data[TIME].dt.date
      times = data[TIME] - data[TIME].dt.normalize()
      data["Time"] = (pd.to_datetime(["1/1/1970" for i in range(data[TIME].size)]) + times)
      data.sort_values(by=[TIME], inplace=True)
      fig = px.line(data, x="Time", y=GLUCOSE, color="Day")
   elif plot_type == "AGP":
      data.reset_index(inplace=True)
      data[[TIME, GLUCOSE]] = resample_data(data[[TIME, GLUCOSE]])
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

      fig = px.line(agp_data, x="Time", y=GLUCOSE, color="Metric")

   return dcc.Graph(figure=fig)


if __name__ == "__main__":
   app.run(debug=True)
"""