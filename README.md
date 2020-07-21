# XploreML
Machine Learning Tools with Best Practices and Frameworks

Version for Python 3.8

## Pipeline Overview 

Step problems for Machine Learning projects. Each problem has a Main Script Template to work with.

### Data collection 

### Data visualization

#### data2view: Visualization data from numerical, categorical and textual sources

Features:
- dddd

### Pre processing raw data 

#### data2dataprep: Pre processing filters for numerical, categorical and textual variables

Features:
- dddd
    
### Building dataset

#### dataprep2dataset: Prepare dataset for model building.  

Features:
- Split training and test subsets
    
### Feature selection problems

### Encoding problems

- Char2Encode
- Word2Encode
- Doc2Encode
- Auto2Encode

### Unsupervised problems

- Cluster2Class

### Supervised problems

#### Static2Value: Problems involving static input feature vector to estimate the most likelihood value estimation

Features:
- Scaling numerical variables
- Encoding categorical variables
- Linear regression model building


- Static2Class: Problems involving static input feature vector to estimate the most likelihood class 
- Img2Class: Problems involving input images to estimate the most likelihood class
- Seq2Class: Problems involving input sequences of feature vectors to estimate the most likelihood class  
- Seq2Value: Problems involving input sequences of feature vectors to estimate the most likelihood value estimation 
- Static2Forecast: Problems involving static input feature vector to estimate the most likelihood forecast estimation from time series
- Seq2Forecast: Problems involving input sequences of feature vectors to estimate the most likelihood forecast estimation from time series

## Setup and execution pipeline

- Create virtual environment
- Install packages

```pip install -r requirements```

- Set environment variables

PYTHON_WARNINGS="ignore"

- Configuration parameters and execution

Set parameters on json config file for selected main scripts applicable for your ML pipeline 

```python SCRIPT_MAIN -f CONFIG_FILE.json```

## ML General Pipeline and Main Scripts

- Collect raw data 
    - Scrape (web2data)
- Build dataset (data2dataset)
    - View raw data (data2view)
    - Preprocessing raw data (data2dataprep)

- Build model
- Analyse model
- Deploy model

### Collect raw data  

### View Raw Data

Run script to visualize data in webview application and analyse quality of data.

``` streamlit run --server.port 80  src/main/data2dataset/all2view_main.py -- -f config/all2view_main_template.json```


### Build dataset

### Build model

### Analyse model

View results on MLOps tracking portal

Terminal command: $ mlflow ui

Start server: $ mlflow ui --port PORTNUMBER
Access website: http://localhost:PORTNUMBER 

### Deploy model


## Project Organization

- docs: links for tutorials and reciples used 
- guides: instructions for working with this project 
- lab: experimental scripts for discovery and initial testing
- src: source code with libraries, labs and main scripts
- outputs: folder for outputs produced by scripts experiments 
- static: static files for experiments 
- test: test template scripts   

## Frameworks Applied

### ML Lifecycle Management

- Mlflow (https://mlflow.org/)
- Streamlit (https://www.streamlit.io/)

### ML Modelling

- Scikit-learn (https://scikit-learn.org/)
- Tensorflow (https://www.tensorflow.org/) 

### Math 

- Numpy (https://numpy.org/)
- Pandas (https://pandas.pydata.org/)

### Visualization

- Matplotlib (https://matplotlib.org/)
- Seaborn (https://seaborn.pydata.org/)

## Technical references

- [Machine Learning A-Z: Hands-On Python & R in Data Science](https://www.udemy.com/course/machinelearning/)

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

