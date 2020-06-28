# XploreML
Machine Learning Tools with Best Practices and Frameworks

## Overview 

Problems to handle in this project. Each problem has a Main Script Template to work with.

### Data collection problems

-  Web2Txt

### Pre processing data problems

- Static2Dataset: Pre processing static data values for dataset building.
- Series2Dataset: Pre processing data values series for dataset building. 
- Txt2Dataset: Pre processing text data for dataset building.

### Feature selection problems

- Data2Components: 

### Encoding problems

- Char2Encode
- Word2Encode
- Doc2Encode
- Auto2Encode

### Unsupervised problems

- Cluster2Class

### Supervised problems

- Static2Class: Problems involving static input feature vector to estimate the most likelihood class 
- Img2Class: Problems involving input images to estimate the most likelihood class
- Seq2Class: Problems involving input sequences of feature vectors to estimate the most likelihood class 
- Static2Value: Problems involving static input feature vector to estimate the most likelihood value estimation 
- Seq2Value: Problems involving input sequences of feature vectors to estimate the most likelihood value estimation 
- Static2Forecast: Problems involving static input feature vector to estimate the most likelihood forecast estimation from time series
- Seq2Forecast: Problems involving input sequences of feature vectors to estimate the most likelihood forecast estimation from time series

## Setup and execution pipeline

- Create virtual environment
- Install packages

```pip install -r requirements```

- Set environment variables

- Configuration parameters and execution

Set parameters on json config file for selected main scripts applicable for your ML pipeline 

```python SCRIPT_MAIN -f CONFIG_FILE.json```

## ML General Pipeline and Main Scripts

- Collect raw data
- Analyse raw data
- Build dataset
- Build model
- Analyse model
- Deploy model

### Collect raw data  

### Analyse raw data

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

- Mlflow

### ML Modelling

- Scikit-learn (https://scikit-learn.org/)
- Tensorflow (https://www.tensorflow.org/) 

### Math 

- Numpy (https://numpy.org/)
- Pandas (https://pandas.pydata.org/)

### Visualization

- Matplotlib (https://matplotlib.org/)
- Seaborn (https://seaborn.pydata.org/)

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
- Project conception by [@danieldominguete](https://github.com/danieldominguete).

