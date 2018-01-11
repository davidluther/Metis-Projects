# Proj-02-Luther

---

## Overview
Being an aviation enthusiast of many years, I chose to look at a common complaint of many air travelers: delay. Would there be a way to predict, with any accuracy, the degree to which one might be late on a particular flight? For the scope of this project, I focused on arrivals at one airport (ORD) from one origin (LGA) operated by one airline (American). I collected available arrival information by scraping data from FlightAware.com. It was my hypothesis that some correlation might be found between lateness (actual arrival minus scheduled arrival) and both day of week and time of day. I used these features and others to build a linear regression model in an attempt to predict the lateness of flights on this route.

## Folder Contents

### Code

Contains all Jupyter notebooks used throughout the project. 
* **0-Luther-Prelim:** initial exploration of scraping requirements
* **1-Luther-Scraping:** more refined code and process for scraping
* **2-Luther-Cleaning-Exploration:** cleaning function, exploration of data, assorted visualizations, cleaning draft
* **3-Luther-Regression:** steps to building final regression model, test on validation set

### Viz

Data visualizations for presentation and/or blog.
