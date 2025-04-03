import io
import zipfile
import pandas as pd
import numpy as np
import configparser
import os

from shiny import App, ui, render, reactive
from shiny.types import FileInfo
from shiny.ui import tags, update_navs
from shinywidgets import output_widget, render_widget


from glucose360.preprocessing import (
    import_data, segment_data
)
from glucose360.events import (
    get_curated_events, import_events, retrieve_event_data,
    event_summary, AUC, iAUC, baseline, peak, nadir, delta,
    event_metrics, create_event_features
)
from glucose360.features import (
    create_features
)
from glucose360.plots import (
    daily_plot, weekly_plot, spaghetti_plot, AGP_plot, AGP_report
)

# ---------------------------------------------------------------------
# Read config (e.g., for ID, TIME, GLUCOSE variable names, etc.)
# ---------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, "config.ini")
config = configparser.ConfigParser()
config.read(config_path)

ID = config['variables']['id']
GLUCOSE = config['variables']['glucose']
TIME = config['variables']['time']
BEFORE = config['variables']['before']
AFTER = config['variables']['after']
TYPE = config['variables']['type']
DESCRIPTION = config['variables']['description']

# ---------------------------------------------------------------------
# Shiny UI
# ---------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.navset_pill(
        # 1) TAB: Import CGM Data
        ui.nav_panel(
            "Import Data",
            ui.layout_columns(
                ui.card(
                    # Replaced the checkbox with a radio button:
                    ui.input_radio_buttons(
                        "data_source_choice",
                        "How would you like to load data?",
                        choices={
                            "example": "Use Example Data",
                            "upload": "Upload Your Own Data"
                        },
                        selected="upload"  # default to "upload", or "example" if you prefer
                    ),
                    # These two UI outputs are now conditional on the radio selection
                    ui.output_ui("sensor_select"),        
                    ui.output_ui("upload_data_button"),   
                    ui.card(
                        "Advanced Parameters",
                        ui.input_numeric("resample_interval", "Resampling Interval", 5, min=1),
                        ui.input_numeric("max_gap", "Maximum Gap for Interpolation", 45),
                        ui.output_ui("advanced_custom_data_options"),
                    ),
                ),
                ui.card(
                    ui.input_file("split_data", "Split Data", accept=[".csv"], multiple=False),
                ),
            ),
            ui.markdown(
                "**⚠️ Important Privacy Notice**: The hosted web application is provided for demonstration purposes only. "
                "**Please DO NOT upload any Protected Health Information (PHI)** or personally identifiable medical data to this "
                "public instance. We do not store any data on our servers — all processing is done in-memory and data is "
                "immediately discarded after your session ends. For processing sensitive health data, we strongly recommend "
                "installing and running the package locally in your secure environment."
            ),
            ui.hr(),
            ui.markdown(
                "If you're **unsure about the correct data format**, please see our sample data."
            ),
            ui.input_action_button("go_sample_data_tab", "View Sample Data Format Examples"),
        ),

        # 2) TAB: Import Events
        ui.nav_panel(
            "Import Events",
            ui.output_ui("patient_import_event"),  # The patient to attach the events to
            ui.layout_columns(
                ui.card(
                    "Bulk Import Events",
                    ui.input_text("event_day_col", "Name of Day Column", "Day"),
                    ui.input_text("event_time_col", "Name of Time Column", "Time"),
                    ui.input_numeric("event_import_before", "# of Minutes Before Timestamp to Include", 60, min=0),
                    ui.input_numeric("event_import_after", "# of Minutes After Timestamp to Include", 60, min=0),
                    ui.input_text("event_type", "Type of Events Being Imported", "Cronometer Meal"),
                    ui.input_file("event_import", "Import Events (.zip or .csv file)", accept=[".zip", ".csv"], multiple=False),
                ),
                ui.card(
                    "Add Singular Event",
                    ui.input_date("add_event_date", "Date"),
                    ui.input_text("add_event_time", "Time"),
                    ui.input_numeric("add_event_before", "# of Minutes Before Time to Include", 60, min=0),
                    ui.input_numeric("add_event_after", "# of Minutes After Time to Include", 60, min=0),
                    ui.input_text("add_event_type", "Type", "Custom Event"),
                    ui.input_text("add_event_description", "Description"),
                    ui.input_action_button("add_event_button", "Add Event"),
                ),
            ),
        ),

        # 3) TAB: Features
        ui.nav_panel(
            "Features",
            ui.card(
                ui.output_data_frame("features_table"),
                ui.download_button("download_features", "Download Features as .csv file")
            ),
            ui.card(
                ui.output_data_frame("event_features_table"),
                ui.download_button("download_event_features", "Download Event Features as .csv file")
            ),
        ),

        # 4) TAB: Plots
        ui.nav_panel(
            "Plots",
            ui.layout_columns(
                ui.output_ui("patient_plot"),  # user picks the patient to plot
                ui.input_select("select_plot", "Select Plot:", 
                    ["Daily (Time-Series)", "Weekly (Time-Series)", "Spaghetti", "AGP"]
                ),
                ui.output_ui("plot_settings"),    # e.g. an optional chunk-day switch for spaghetti
                ui.output_data_frame("daily_filters"),
                ui.output_ui("plot_height"),
                ui.download_button("download_plot", "Download Currently Displayed Plot"),
                ui.download_button("download_AGP_report", "Download AGP Report for Patient"),
            ),
            ui.card(
                output_widget("plot"),
            ),
        ),

        # 5) TAB: Events (view/edit)
        ui.nav_panel(
            "Events",
            ui.output_ui("patient_event"),
            ui.layout_columns(
                ui.card(
                    ui.output_ui("ui_event_row"),
                    ui.input_select("edit_event_col", "Column to Edit", [TIME, BEFORE, AFTER, TYPE, DESCRIPTION]),
                    ui.input_text("edit_event_text", "Value to Replace With"),
                    ui.input_action_button("edit_event", "Edit Event"),
                ),
                ui.card(ui.output_data_frame("event_metrics_table")),
            ),
            ui.layout_columns(
                ui.card(
                    ui.output_ui("ui_event_select"),
                    tags.div(
                        tags.div(ui.output_data_frame("event_filters"), style="height: 100%"),
                        ui.output_data_frame("events_table"),
                        style="display: flex;"
                    ),
                    ui.download_button("download_events", "Download Events in Table as .csv file"),
                ),
                ui.card(output_widget("display_event_plot")),
            ),
        ),

        # 6) TAB: Sample Data Formats
        ui.nav_panel(
            "Sample Data Formats",
            ui.h2("Examples of Common CGM Data Formats"),
            ui.card(
                ui.h3("Dexcom CSV Example"),
                tags.img(src="dexcom_sample.png", style="max-width:100%; height:auto;")
            ),
            
            ui.card(
                ui.h3("Custom Columns CSV Example"),
                tags.img(src="custom_columns_sample.png", style="max-width:25%; height:auto;")
            ),
            value="sample_data_formats"
        ),
        id="tab",
    )
)

# ---------------------------------------------------------------------
# Shiny Server
# ---------------------------------------------------------------------
def server(input, output, session):
    # Keep references to data & events in reactive.Value
    events_ref = reactive.Value(pd.DataFrame())
    filtered_events_ref = reactive.Value(pd.DataFrame())
    daily_events_ref = reactive.Value(None)

    fig = None  # We'll store the active plot figure here

    def notify(message):
        ui.notification_show(message, close_button=True)

    # -----------------------------------------------------------------
    # 1) Reactive: df() loads or returns the CGM data
    #    Re-written to handle the radio button logic
    # -----------------------------------------------------------------
    @reactive.Calc
    def df():
        """
        This reactive is responsible for loading CGM data, optionally from an example file.
        Then it optionally segments the data if a 'split_data' CSV is uploaded.
        Finally it auto-generates curated events (episodes/excursions) to populate events_ref.
        """
        data_source = input.data_source_choice()  # "example" or "upload"
        
        # If user chooses "example"
        if data_source == "example":
            # Example data path from your repository:
            path = "trial_ids.zip"  # adjust as needed
            name = None
            sensor = "dexcom"  # or whatever default you want
            data = import_data(
                path=path, 
                name=name,
                sensor=sensor,
                interval=input.resample_interval(), 
                max_gap=input.max_gap(),
                output=notify
            )

        else:
            # User chooses "upload"
            data_file: list[FileInfo] | None = input.data_import()
            if not data_file:
                return pd.DataFrame()  # no file yet
            path = data_file[0]["datapath"]
            name = data_file[0]["name"].split(".")[0]
            sensor = input.sensor()
            
            # By default, None unless the user typed custom columns
            glucose_col = input.glucose_col() if input.glucose_col() else None
            time_col = input.time_col() if input.time_col() else None
            id_template = input.id_template() if input.id_template() else None

            data = import_data(
                path=path, 
                name=name,
                sensor=sensor,
                id_template=id_template,
                glucose=glucose_col,
                time=time_col,
                interval=input.resample_interval(), 
                max_gap=input.max_gap(),
                output=notify
            )

        # If user uploaded a CSV to split data
        split_file: list[FileInfo] | None = input.split_data()
        if split_file is not None:
            data = segment_data(split_file[0]["datapath"], data)

        # Generate curated events automatically (hypo/hyper episodes, excursions)
        auto_events = get_curated_events(data).reset_index(drop=True)
        events_ref.set(auto_events)
        filtered_events_ref.set(auto_events)

        return data

    # -----------------------------------------------------------------
    # 2) Bulk Import of Additional Events
    # -----------------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.event_import)
    def bulk_import_events():
        file: list[FileInfo] | None = input.event_import()
        if file:
            # Import new events
            new_events = import_events(
                path=file[0]["datapath"],
                id=input.select_patient_import_event(),
                day_col=input.event_day_col(),
                time_col=input.event_time_col(),
                before=input.event_import_before(),
                after=input.event_import_after(),
                type=input.event_type()
            )
            # Combine with existing events in events_ref
            combined = pd.concat([events_ref.get(), new_events]).reset_index(drop=True)
            combined[TIME] = pd.to_datetime(combined[TIME])
            combined.sort_values(by=[TIME], inplace=True)
            events_ref.set(combined)
            filtered_events_ref.set(combined)

    # -----------------------------------------------------------------
    # 3) Add Single Event
    # -----------------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.add_event_button)
    def add_single_event():
        single_event = pd.DataFrame.from_records({
            ID: input.select_patient_import_event(),
            TIME: str(input.add_event_date()) + " " + input.add_event_time(),
            BEFORE: input.add_event_before(),
            AFTER: input.add_event_after(),
            TYPE: input.add_event_type(),
            DESCRIPTION: input.add_event_description()
        }, index=[0])

        single_event[TIME] = pd.to_datetime(single_event[TIME])

        updated = pd.concat([events_ref.get(), single_event]).reset_index(drop=True)
        events_ref.set(updated)
        filtered_events_ref.set(updated)

    # -----------------------------------------------------------------
    # 4) Features Tab
    # -----------------------------------------------------------------
    @render.data_frame
    def features_table():
        """
        Show computed CGM features for each patient in the dataset.
        """
        if df().empty:
            raise Exception("No CGM data loaded. Please upload CGM data above.")
        feats = create_features(df())
        return render.DataGrid(feats.reset_index(names=["Patient"]))

    @render.download(filename="features.csv")
    def download_features():
        if df().empty:
            raise Exception("No CGM data loaded.")
        yield create_features(df()).reset_index(names=["Patient"]).to_csv(index=False)

    @render.data_frame
    def event_features_table():
        """
        Show computed event-based features for each patient in the dataset.
        """
        if df().empty:
            raise Exception("No CGM data loaded. Please upload CGM data above.")
        feats = create_event_features(df(), events_ref.get())
        return render.DataGrid(feats.reset_index(names=["Patient"]))

    @render.download(filename="event_features.csv")
    def download_event_features():
        if df().empty:
            raise Exception("No CGM data loaded.")
        yield create_event_features(df(), events_ref.get()).reset_index(names=["Patient"]).to_csv(index=False)

    # -----------------------------------------------------------------
    # 5) Plots Tab
    # -----------------------------------------------------------------

    @render.ui
    def patient_plot():
        """
        Dropdown to select which patient's data to plot (unique IDs from df())
        """
        data = df()
        if data.empty:
            return None
        return ui.input_select(
            "select_patient_plot",
            "Select Patient:",
            data.index.unique().tolist()
        )

    @render.ui
    def sensor_select():
        """
        Only appear if data_source_choice == 'upload'
        """
        if input.data_source_choice() == "upload":
            return ui.input_select(
                "sensor", 
                "Type of CGM Device:", 
                {
                    "dexcom": "Dexcom",
                    "freestyle libre 2": "FreeStyle Libre 2 or 3",
                    "freestyle libre pro": "FreeStyle Libre Pro",
                    "columns": "Custom CSV Format (Adhering to glucose360 Guidelines)",
                }
            )
        return None

    @render.ui
    def upload_data_button():
        """
        Only appear if data_source_choice == 'upload'
        """
        if input.data_source_choice() == "upload":
            return ui.input_file(
                "data_import", 
                "Import CGM Data (.csv or .zip file)", 
                accept=[".zip", ".csv"], 
                multiple=False
            )
        return None

    @render.ui
    def advanced_custom_data_options():
        """
        Additional text fields for custom columns if data_source_choice == 'upload'.
        """
        if input.data_source_choice() == "upload":
            return ui.TagList(
                ui.input_text("glucose_col", "Name of Glucose Column"),
                ui.input_text("time_col", "Name of Timestamp Column"),
                ui.input_text("id_template", "Template for ID Retrieval (Regex or Format)"),
            )
        return None
    
    @reactive.Effect
    @reactive.event(input.go_sample_data_tab)
    def go_to_sample_data_tab():
        # Switch nav panels to "Sample Data Formats"
        update_navs("tab", selected="sample_data_formats")

    @render.ui
    def patient_import_event():
        """
        In 'Import Events' tab, pick which patient to attach these events to.
        """
        data = df()
        if data.empty:
            return None
        return ui.input_select(
            "select_patient_import_event", 
            "Select Patient to Add Events For:", 
            data.index.unique().tolist()
        )

    @render.ui
    def plot_settings():
        """
        Example: if user chooses 'Spaghetti' plot, we can show a 'chunk by weekday/weekend' switch.
        """
        plot_type = input.select_plot()
        if plot_type == "Spaghetti":
            # Provide the chunk-day switch
            return ui.input_switch("spaghetti_chunk_switch", "Chunk Weekend/Weekday", value=False)
        return None

    @render.data_frame
    def daily_filters():
        """
        If user selects a daily plot, show a data frame listing event types that they can highlight or filter.
        We'll store selected rows in daily_events_ref for annotation.
        """
        events = events_ref.get()
        if events.empty:
            return None

        # Show only types for the chosen patient
        patient = input.select_patient_plot()
        if not patient:
            return None

        subset = events[events[ID] == patient].drop_duplicates(subset=[TYPE])
        if subset.empty:
            return None

        table = pd.DataFrame(subset[TYPE])
        return render.DataGrid(table, row_selection_mode="multiple")

    @reactive.Effect
    @reactive.event(input.daily_filters_selected_rows)
    def filter_daily_events():
        """
        Creates a subset of events for the selected rows in daily_filters table.
        That subset is passed to daily_plot for annotation lines/tables.
        """
        daily_events_ref.set(None)
        events = events_ref.get()
        if events.empty:
            return

        if not input.select_patient_plot():
            return

        patient = input.select_patient_plot()
        patient_events = events[events[ID] == patient].copy()

        selected_rows = input.daily_filters_selected_rows()
        if selected_rows:
            # Convert row indexes to event types
            unique_types = patient_events[TYPE].unique()
            type_series = pd.Series(unique_types)
            selected_types = type_series.iloc[list(selected_rows)]
            # Keep only those events
            final = patient_events[patient_events[TYPE].isin(selected_types)]
            daily_events_ref.set(final)

    @render.ui
    def plot_height():
        """
        Let user define the vertical size of the plot (slider).
        We can adjust default min/max based on the chosen plot type.
        """
        plot_type = input.select_plot()
        default_val = 750
        min_val = 500
        max_val = 2000

        if plot_type == "Daily (Time-Series)":
            default_val = 3000
            min_val = 750
            max_val = 4000
        elif plot_type == "Spaghetti":
            default_val = 600
            min_val = 250
            max_val = 750
        elif plot_type == "AGP":
            default_val = 1000
            min_val = 600
            max_val = 2000
        elif plot_type == "Weekly (Time-Series)":
            default_val = 1000
            min_val = 500
            max_val = 2000

        return ui.input_slider("plot_height_slider", "Set Plot Height", min_val, max_val, default_val)

    @render_widget
    def plot():
        """
        Renders the final Plotly figure chosen by user: daily, weekly, spaghetti, or AGP.
        """
        nonlocal fig

        data = df()
        if data.empty:
            # Return an empty figure if no data
            import plotly.graph_objects as go
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No CGM data loaded.")
            return empty_fig

        patient = input.select_patient_plot()
        if not patient:
            # Return empty if no patient selected
            import plotly.graph_objects as go
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Select a patient to display.")
            return empty_fig

        plot_type = input.select_plot()
        height = input.plot_height_slider()

        # We can slice the data for the chosen patient
        # (Your code references multi-index data, so adapt if needed)
        patient_data = data.loc[patient]

        # For daily plots, we might pass in daily_events_ref to show vertical lines/tables
        highlight_events = None
        if plot_type == "Daily (Time-Series)":
            highlight_events = daily_events_ref.get()

        # Depending on plot_type, call your custom plots
        if plot_type == "Daily (Time-Series)":
            fig = daily_plot(df=data, id=patient, height=height, events=highlight_events, app=True)
        elif plot_type == "Weekly (Time-Series)":
            fig = weekly_plot(df=data, id=patient, height=height, app=True)
        elif plot_type == "Spaghetti":
            chunk_day = input.spaghetti_chunk_switch() if "spaghetti_chunk_switch" in input else False
            fig = spaghetti_plot(df=data, id=patient, chunk_day=chunk_day, height=height, app=True)
        else:  # "AGP"
            fig = AGP_plot(df=data, id=patient, height=height, app=True)

        return fig

    @render.download(filename=lambda: f"{input.select_patient_plot()}_{input.select_plot()}.zip")
    def download_plot():
        """
        Exports the figure as PDF + HTML in a single ZIP.
        """
        import plotly.io as pio

        nonlocal fig
        if fig is None:
            yield b""
            return

        # Create PDF and HTML
        encoded_pdf = fig.to_image(format="pdf", width=1500, height=input.plot_height_slider())
        encoded_html = fig.to_html().encode('utf-8')

        pdf_stream = io.BytesIO(encoded_pdf)
        html_stream = io.BytesIO(encoded_html)

        zip_stream = io.BytesIO()
        with zipfile.ZipFile(zip_stream, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("plot.pdf", pdf_stream.getvalue())
            zf.writestr("plot.html", html_stream.getvalue())

        zip_stream.seek(0)
        yield zip_stream.getvalue()

    @render.download(filename=lambda: f"{input.select_patient_plot()}_AGP_Report.html")
    def download_AGP_report():
        """
        If user chooses to generate an AGP_report for the selected patient.
        """
        patient = input.select_patient_plot()
        if not patient:
            yield b"No patient selected."
            return

        # Generate the HTML via AGP_report
        report_html = AGP_report(df(), patient)
        yield report_html.encode("utf-8")

    # -----------------------------------------------------------------
    # 6) Events Tab (view/edit)
    # -----------------------------------------------------------------

    @render.ui
    def patient_event():
        """
        Dropdown to pick which patient's events to manage
        """
        data = df()
        if data.empty:
            return None
        return ui.input_select("select_patient_event", "Select Patient:", data.index.unique().tolist())

    @render.data_frame
    def event_filters():
        """
        Let user filter event types via row selections in a minimal table
        """
        events = events_ref.get()
        if events.empty:
            return None
        if not input.select_patient_event():
            return None

        subset = events[events[ID] == input.select_patient_event()].drop_duplicates(subset=[TYPE])
        if subset.empty:
            return None

        table = pd.DataFrame(subset[TYPE])
        return render.DataGrid(table, row_selection_mode="multiple")

    @reactive.Effect
    @reactive.event(input.select_patient_event, input.event_filters_selected_rows, input.edit_event)
    def filter_events():
        """
        Filter events based on the chosen patient & the row selections of event_filters.
        Keep them in filtered_events_ref, which is displayed in events_table.
        """
        all_events = events_ref.get()
        if all_events.empty:
            return

        patient = input.select_patient_event()
        if not patient:
            filtered_events_ref.set(pd.DataFrame())
            return

        subset = all_events[all_events[ID] == patient].copy()
        # if user picked any event types from the filter table
        selected_rows = input.event_filters_selected_rows()
        if selected_rows:
            unique_types = subset[TYPE].unique()
            chosen_type_series = pd.Series(unique_types)
            selected_types = chosen_type_series.iloc[list(selected_rows)]
            subset = subset[subset[TYPE].isin(selected_types)]

        filtered_events_ref.set(subset)

    @render.data_frame
    def events_table():
        """
        Shows the filtered events in a DataGrid, with row selection enabled for e.g. editing or exporting.
        """
        events = filtered_events_ref.get().copy()
        if events.empty:
            return None

        # Convert TIME to string so it looks nice in the grid
        events[TIME] = events[TIME].astype(str)
        return render.DataGrid(events.drop(columns=[ID]).reset_index(drop=True), row_selection_mode="multiple")

    @render.ui
    def ui_event_row():
        """
        UI numeric input to let user pick which row to edit in the filtered events table
        """
        events = filtered_events_ref.get()
        if events.empty:
            return None
        return ui.input_numeric(
            "edit_event_row", 
            "Row to Edit", 
            value=1, 
            min=1, 
            max=events.shape[0]
        )

    @reactive.Effect
    @reactive.event(input.edit_event)
    def edit_event():
        """
        Let user modify some cell in the filtered events table for the chosen row.
        They specify the column from a dropdown, new text, and row index in 'ui_event_row'.
        """
        events = filtered_events_ref.get()
        if events.empty:
            return

        row_idx = input.edit_event_row() - 1
        if row_idx < 0 or row_idx >= events.shape[0]:
            return

        col = input.edit_event_col()
        val = input.edit_event_text()

        # If we're editing TIME, parse as datetime
        if col == TIME:
            val = pd.to_datetime(val)
        elif col == BEFORE or col == AFTER:
            val = int(val)

        events.iloc[row_idx, events.columns.get_loc(col)] = val

        # Reassign to both references
        filtered_events_ref.set(events)
        # Also update events_ref (the global event store)
        full = events_ref.get()
        # Because 'events' here is a subset: replace those indexes
        full.loc[events.index, col] = events[col]
        events_ref.set(full)

    @render.download(filename="events.csv")
    def download_events():
        """
        Export the selected rows in the 'events_table' as a CSV.
        """
        events = filtered_events_ref.get()
        sel = input.events_table_selected_rows()
        if events.empty or not sel:
            yield "No events selected."
            return

        out = events.iloc[list(sel)].reset_index(drop=True)
        yield out.to_csv(index=False)

    @render.ui
    def ui_event_select():
        """
        Let user pick which event row they want to show metrics/plot for
        """
        events = filtered_events_ref.get()
        if events.empty:
            return None
        return ui.input_numeric("select_event", "Event to Display", 1, min=1, max=events.shape[0])

    @render.data_frame
    def event_metrics_table():
        """
        Calculate some basic metrics for the single selected event.
        """
        events = filtered_events_ref.get()
        if events.empty:
            return None
        row_idx = input.select_event()
        if not row_idx or row_idx < 1 or row_idx > events.shape[0]:
            return None

        # Grab that single event
        single_event = events.iloc[row_idx - 1]
        # Generate metrics
        metrics_df = event_metrics(df(), single_event)
        return render.DataGrid(metrics_df)

    @render_widget
    def display_event_plot():
        """
        Render an event-specific CGM plot for the single selected event.
        """
        import plotly.graph_objects as go

        events = filtered_events_ref.get()
        if events.empty:
            # Return empty figure
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No events available.")
            return empty_fig

        row_idx = input.select_event()
        if not row_idx or row_idx < 1 or row_idx > events.shape[0]:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Select an event from the table.")
            return empty_fig

        # For demonstration, we can show lines for all events of the same patient
        single_event = events.iloc[row_idx - 1]
        # Build the event-specific plot
        patient = single_event[ID]

        # If you want to hide the automatically curated episodes in that plot, you can filter them out:
        # e.g. keep only the "user-defined" ones or whichever logic you prefer
        # For now, let's just show everything from 'events':
        all_events_same_patient = events_ref.get()
        all_events_same_patient = all_events_same_patient[all_events_same_patient[ID] == patient].copy()

        # Use your event_plot function
        return event_plot(df(), patient, single_event, all_events_same_patient, app=True)

# ---------------------------------------------------------------------
# Create the Shiny app
# ---------------------------------------------------------------------
app = App(app_ui, server, debug=False, static_assets=os.path.join(os.path.dirname(__file__), "www"))
