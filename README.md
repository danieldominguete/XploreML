# XploreML
Machine Learning Tools with Best Practices and Frameworks

## Overview 

Problems to handle in this project. Each problem has a Main Script Template to work with.

### Data collection problems

### Pre processing data problems

### Feature selection problems

### Encoding problems

- Char2Encode
- Word2Encode
- Auto2Encode

### Unsupervised problems

### Supervised problems

- Static2Class: Problems involving static input feature vector to estimate the most likelihood class 
- Img2Class: Problems involving input images to estimate the most likelihood class
- Seq2Class: Problems involving input sequences of feature vectors to estimate the most likelihood class 
- Static2Regress: Problems involving static input feature vector to estimate the most likelihood value estimation 
- Img2Regress: Problems involving input images to estimate the most likelihood value estimation
- Seq2Regress: Problems involving input sequences of feature vectors to estimate the most likelihood value estimation 
- Static2Forecast: Problems involving static input feature vector to estimate the most likelihood forecast estimation from time series
- Seq2Forecast: Problems involving input sequences of feature vectors to estimate the most likelihood forecast estimation from time series

### Reinforcement problems


## Setup and execution pipeline

- Create virtual environment
- Install packages

```pip install -r requirements```

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
- lib: XploreML official library
- main: main template scripts
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

