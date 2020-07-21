# XploreML
Machine Learning Tools with Best Practices and Frameworks (Python 3.8)

## Setup and execution pipeline

Install virtual environment package

`pip install virtualenv`

Create the virtual environment

`virtualenv XploreML`

Acivate the virtual environment

Mac OS/Linux: `source XploreML/bin/activate`

Windows: `XploreML\Scripts\activate`

Install requirement packages

`pip install -r requirements`

Create .env file and set environment variables

`PYTHON_WARNINGS="ignore"`

Select the main task and set parameters on json config file for selected main scripts applicable for your ML pipeline 

```python SCRIPT_MAIN -f CONFIG_FILE.json```

Each routine has a command example to run with a simple example database. The config parameters are explained
in ```src/lib/data_schemas``` folder. Each json object has a python class data schema with comments. 

View results on ML tracking portal

Start server: `mlflow ui --port PORTNUMBER`

Access website: `http://localhost:PORTNUMBER`


## Pipeline Overview and Project Tools 

Step problems for Machine Learning projects. Each problem has a main script + json config file to work with.

### Step 1 - Data collection

_In progress._ 

### Step 2 - Data visualization

#### data2view: Visualization data from numerical, categorical and textual sources

Run script to visualize data in webview application.

``` streamlit run --server.port 80  src/main/data2dataset/data2view_main.py -- -f config/data2view_main_template.json```

Features:
- Load data criteria
- Variables selection
- Streamlit application with visualization filters

### Step 3 - Pre processing raw data 

#### data2dataprep: Pre processing filters for numerical, categorical and textual variables

Features:
- Missing data processing
- Duplicate samples processing
- Outliers detection
- 
    
### Step 4 - Building dataset

#### dataprep2dataset: Prepare dataset for model building.  

Features:
- Split training and test subsets

### Step 5 - Encoding problems

#### Char2Encode
_In progress._

#### Word2Encode
_In progress._

#### Doc2Encode
_In progress._

#### Auto2Encode
_In progress._
    
### Step 6 - Feature selection problems
_In progress._

### Step 7 - Model building

### Unsupervised problems

#### Cluster2Class
_In progress._

### Supervised problems

#### Static2Value: Problems involving static input feature vector to estimate the most likelihood value estimation

Features:
- Scaling numerical variables
- Encoding categorical variables
- Linear regression model building


#### Static2Class: Problems involving static input feature vector to estimate the most likelihood class
_In progress._ 
#### Img2Class: Problems involving input images to estimate the most likelihood class
_In progress._
#### Seq2Class: Problems involving input sequences of feature vectors to estimate the most likelihood class
_In progress._  
#### Seq2Value: Problems involving input sequences of feature vectors to estimate the most likelihood value estimation
_In progress._ 
#### Static2Forecast: Problems involving static input feature vector to estimate the most likelihood forecast estimation from time series
_In progress._
#### Seq2Forecast: Problems involving input sequences of feature vectors to estimate the most likelihood forecast estimation from time series
_In progress._

### Step 8 - Model deploy
_In progress._

## Repository Organization

- docs: links for tutorials and reciples used 
- guides: instructions for working with this project 
- lab: experimental scripts for discovery and initial testing
- src: source code with libraries, labs and main scripts
- outputs: folder for outputs produced by scripts experiments 
- static: static files for experiments 
- test: test template scripts   

# Technical references

## Frameworks Applied

### ML Lifecycle Management

- [Mlflow](https://mlflow.org/)
- [Streamlit](https://www.streamlit.io/)

### ML Modelling

- [Scikit-learn](https://scikit-learn.org/)
- [Tensorflow](https://www.tensorflow.org/) 

### Math 

- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

### Visualization

- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

## Courses 
- [Machine Learning A-Z: Hands-On Python & R in Data Science](https://www.udemy.com/course/machinelearning/)

## Articles

# Changelog
All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

- Added for new features.
- Changed for changes in existing functionality.
- Deprecated for soon-to-be removed features.
- Removed for now removed features.
- Fixed for any bug fixes.
- Security in case of vulnerabilities.

## [Unreleased]

### Added
- Linear regression for static2value taks
- Result tracking for data dataprep2dataset with mlflow
- Split training and test datasets for model building (dataprep2dataset)
- Essencial preprocessing tasks for numerical, categorical and textual data (data2dataprep)
- Simple application for data visualization with streamlit webserver (data2view)
- Script management: logging, working folder, time spent
- Project conception by [@danieldominguete](https://github.com/danieldominguete).

