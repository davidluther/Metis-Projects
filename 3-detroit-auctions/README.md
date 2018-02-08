# Metis Project 3 — McNulty

## Overview

While browsing through the open data portal of my home state's largest city, Detroit, I found a table of all closed auctions in the [Detroit Land Bank Authority](http://www.buildingdetroit.org/) auction program, which puts vacant and/or blighted properties up for auction in an effort to stabilize Detroit neighborhoods. This table listed the buyer type as either a Homeowner or Investor. This seemed ripe for a classification effort, so I set out to see if I could build a model to predict which type the buyer would be. I used data included in the [Auctions Closed](https://data.detroitmi.gov/Property-Parcels/DLBA-Auctions-Closed/tgwk-njih) table, as well as that from a [record of building permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf) issued. Algorithms employed included Logistic Regression, Support Vector Machine, Naive Bayes, and Random Forest.

## Folder Contents

### Code

Contains all Jupyter notebooks, Python scripts, and SQL scripts used throughout the project.

* **0-mcnulty-exploration**: initial exploratory measures and early viz
* **05-mcnulty-cleaning**: automation of initial cleaning and formatting steps
* **06-mcnulty-permits-feature-eng**: exploring relationships between auctions table and building permits table to in an effort to figure out if permits or their characteristcs could be used as features, also some feature engineering efforts with original features of the closed auctions table
* **07-mcnulty-class-development**: construction of two classes to be used in the feature engineering efforts
* **10-mcnulty-feature-engineering-final**: automation of feature engineering
* **20-mcnulty-viz**: all pre-modeling visualizations
* **30-mcnulty-model-dev**: exploratory modeling, grid search cross-validation
* **31-model-testing**: testing of three final models on reserved test set
* **mcnultymod.py**: module of all major functions and classes used throughout the project
* **mcnulty.sql**: several SQL scripts used for table creation and queries

### Img

Image files saved for presentation and/or blog.

### Presentation

PDF of final presentation with appendix slides.
