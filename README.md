# Glucose360

A Python package that provides **Continuous Glucose Monitoring (CGM)** data preprocessing, feature extraction, event detection, and plotting utilities. Whether you're analyzing CGM data for research, clinical insights, or personal projects, this library aims to simplify your workflow.

---

## Table of Contents

- [Overview](#overview)
- [Hosted Web UI](#hosted-web-ui)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Using the Local Web Application](#using-the-local-web-application)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

This package simplifies the handling of CGM (Continuous Glucose Monitoring) data by providing:

1. **Preprocessing**: Easily clean and structure raw CGM readings.  
2. **Feature Extraction**: Derive statistical, temporal, or clinically relevant metrics from your CGM data.  
3. **Event Detection**: Automatically identify significant patterns, such as hypo- or hyperglycemic events or create features around events such as meals.
4. **Plotting**: Generate visualizations.

---

## Hosted Web UI

If you prefer not to install anything locally, explore our hosted web application here:  
[**Web App »**](https://vurhd2.shinyapps.io/glucose360/)

> **⚠️ Important Privacy Notice**: The hosted web application is provided for demonstration purposes only. Please DO NOT upload any Protected Health Information (PHI) or personally identifiable medical data to this public instance. We do not store any data on our servers - all processing is done in-memory and data is immediately discarded after your session ends. For processing sensitive health data, we strongly recommend installing and running the package locally in your secure environment.

Use it to upload CGM files, process them on the fly, and visualize key metrics immediately.

---

## Installation

### Via pip

```bash
pip install glucose360
```

Visit the [PyPI package page](https://pypi.org/project/glucose360) for more details.

This is the simplest method to install the package and will automatically handle all dependencies.

### From Source

1. **Clone or Download**  
   ```bash
   git clone https://github.com/vurhd2/Glucose360.git
   ```
   Or download the repository as a ZIP and unzip it.

2. **Navigate**  
   ```bash
   cd Glucose360
   ```

3. **Install Dependencies**  
   - General library dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - If you also want to run the local web application:
     ```bash
     pip install -r app_requirements.txt
     ```

That's it! You're ready to use the library.

---

## Getting Started

Here's how to start using the package:

```python
import pandas as pd
from preprocessing import *
from features import *
from events import *
from plots import *

# Import and preprocess your data
df = import_data("datasets/")  # Supports .csv, directory of .csvs, or .zip

# Calculate CGM metrics and statistics
metrics = create_features(df)
print("Key Stats:", metrics)

# Get CGM events (e.g., hypo/hyperglycemic episodes)
events = get_curated_events(df)
print("Detected Events:", events)

# Generate visualizations
daily_plot(df)  # Time series plot
AGP_plot(df)  # Ambulatory Glucose Profile
```

For a more detailed walkthrough, see [guide.ipynb](./examples/guide.ipynb).

---

## Using the Local Web Application

1. **Navigate to `app.py`**:  
   ```bash
   cd path/to/Glucose360/app
   ```
2. **Launch the Shiny App**:  
   ```bash
   shiny run
   ```

3. **Interact in the Browser**:  
   Access the provided URL (usually `http://127.0.0.1:xxxx`) to upload your own data, process it, and visualize results in real time.

4. **Feature Overview**:  
   See [web_app_walkthrough.mp4](./web_app_walkthrough.mp4) for a guided tour.

---

## Documentation

Complete documentation is available on ReadTheDocs:  
[**ReadTheDocs Link**](https://glucose360.readthedocs.io/en/latest/)

---

## License

This project is licensed under the [GNU General Public License v2.0](./LICENSE). Feel free to use, modify, and distribute this software under the terms of this
license.

**MAGE algorithm**: Glucose360 reuses iglu's unique implementation of MAGE with only a R->python modification. If you use Glucose360's implementation of MAGE, we ask that you additionally cite the following papers:

Fernandes N, Nguyen N, Chun E, Punjabi N and Gaynanova I (2022) Open-Source Algorithm to Calculate Mean Amplitude of Glycemic Excursions Using Short and Long Moving Averages. Journal of Diabetes Science and Technology, Vol. 16, No. 2, 576-577.

Broll S, Urbanek J, Buchanan D, Chun E, Muschelli J, Punjabi N and Gaynanova I (2021). Interpreting blood glucose data with R package iglu. PLoS One, Vol. 16, No. 4, e0248560.

Chun E, Fernandes JN and Gaynanova I (2024) An Update on the iglu Software for Interpreting Continuous Glucose Monitoring Data. Diabetes Technology and Therapeutics, Vol. 26, No. 12, 939-950.

---

Other excellent CGM data analysis packages can be found in the following languages:

R: [cgmanalysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0216851)

R: [GLU](https://pmc.ncbi.nlm.nih.gov/articles/PMC7394960/)

R: [iglu](https://irinagain.github.io/iglu/)

Matlab: [AGATA](https://github.com/gcappon/agata)

**Thanks for using Glucose360!**
