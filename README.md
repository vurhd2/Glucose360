# Python-CGM-Package

**Installation**:
1. Download the contents of the repository into a directory (either by using `git clone` or downloading all the files through a .zip file)
2. Navigate to this directory in a terminal
3. Install necessary languages and libraries:
    1. Ensure some version of Python 3 is located on your machine (does not need to be the latest version)
    2. run `pip install -r requirements.txt` in the terminal
    3. run `pip install -r app_requirements.txt` in the terminal **solely if** you wish to use the web application functionality too
4. That's it for installation!

**Getting Started**:
- Using the package directly:
    - in the file(s) you want to use the package in, import the four package modules: preprocessing.py, features.py, events.py, and plots.py
    - see [guide.ipynb](./guide.ipynb) for a brief notebook guide on all the functions that the package currently provides, as well as how to use them properly
- Using the web application:
    - Run `shiny run` in the terminal of the directory that houses `app.py`
    - See [![web_app_walkthrough.mp4](./web_app_walkthrough.mp4)](./web_app_walkthrough.mp4) for a more detailed walkthrough of the application's features

 Thanks for using the package!