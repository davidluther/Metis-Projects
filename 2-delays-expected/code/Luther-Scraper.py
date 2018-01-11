import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pickle
from datetime import datetime, timedelta
import time
import re
import sys

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver


def make_flight_url(date_str, flight_num, route_num):
    """
    Takes date string, flight number, and route number, returns 
    URL to search for a particular flight/route/day. 
    ---
    IN: date string YYYYMMDD (str), 
        flight_num (str), 
        route_num (str)
    OUT: search URL (str)
    """

    # base URL constructors
    base_url1 = 'http://flightaware.com/live/flight/'
    base_url2 = '/history/'
    base_url3 = '/KLGA/KORD'
    
    # merge vars with URL bases
    search_url = (base_url1 + flight_num + base_url2 + 
                  date_str + '/' + route_num + base_url3)

    return search_url


def scrape_flight_soup(soup, flight_num, search_date):
    """
    Scrapes pertinient information off single flight page, returns record for 
    that flight (one record), returns None if no record for that day.
    ---
    IN: BS4 object of webpage, search date (datetime obj)
    OUT: four flight arrival times (list)
    """
    
    date_str = datetime.strftime(search_date, "%Y%m%d")
    
    # is there a flight that day?
    names = []
    for meta in soup.find_all('meta'):
        names.append(meta.get('name'))
    if not 'airline' in names:
        return 'No Flight'
    
    # was the flight canceled?
    if 'cancelled' in soup.find(class_="flightPageSummary").text.lower():
        return 'Canceled'
    
    # if flight arrived
    try:
        details = soup.find(class_="flightPageDetails")
        details_sub = details.find(attrs={
            "data-template": "live/flight/details"})
        spans = list(details_sub.find_all('span'))
        arrival_times = []
        fptd_divs = details_sub.find_all(class_="flightPageTimeData")

        # pulls from the four relevant indices of fptd_divs
        for i in [9,11,12,14]:
            time_str = fptd_divs[i].text.strip().split(' ')[0]
            arrival_times.append(time_str)

        arr_conv = map(lambda x: datetime.strptime(x, "%I:%M%p").time(), 
                       arrival_times)
        arrival_times = list(map(lambda x: datetime.combine(search_date, x), 
                       arr_conv))
        return arrival_times

    except Exception as e:
        print(f"*** {flight_num}, {date_str}: ERROR: {e}")
        return None

def scrape_fn(days, flight_num, route_num, df=None):
    """
    Goes through a series of steps to gather data for a given flight number and    route over a given length of time. Appends each record to a dataframe 
    (provided or generated).
    ---
    IN: days, number of days to scrape, starting yesterday (int)
        flight_num, flight number as searched on FlightAware (str)
        route_num, route number as searched on FlightAware (str)
        df, pandas dataframe
    OUT: pandas dataframe
    """
    
    # makes df if none passed
    if df is None:
        df = pd.DataFrame(columns=['airline',
                                   'f_num',
                                   'origin',
                                   'dest',
                                   'date',
                                   'land_act',
                                   'arr_act',
                                   'land_sch',
                                   'arr_sch'])

    # starts Selenium and sets timeout preferences
    driver = webdriver.Chrome(chromedriver)
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(5)
    
    today = datetime.now().date()
    no_flight_count = 0
    
    # loop to search each date
    for d in range(days):
        time.sleep(np.random.uniform(1.0,2.0))
        search_date = today - timedelta(days=d+1)
        date_str = datetime.strftime(search_date, "%Y%m%d")
        record_a = ['American', flight_num, 'LGA', 'ORD', search_date]
        flight_url = make_flight_url(date_str, flight_num, route_num)
        
        try:
            driver.get(flight_url)
            flight_soup = BeautifulSoup(driver.page_source, 'html.parser')
            record_b = scrape_flight_soup(flight_soup, flight_num, search_date)
        except Exception as e:
            print(f"*** {flight_num}, {date_str}: ERROR: {e}")
            record_b = None
        
        if record_b == None:
            continue
        
        elif record_b == 'Canceled':
            no_flight_count = 0
            print(f"{flight_num}, {date_str}: canceled")
        
        elif record_b == 'No Flight':
            no_flight_count += 1
            print(f"{flight_num}, {date_str}: no flight")
            if no_flight_count == 7:
                print(f"{flight_num}: 7 consecutive days of no flights as of {date_str}!")
                break
        else:
            no_flight_count = 0
            record = record_a + record_b
            print(f"{flight_num}, {date_str}: flight data recorded")
            df.loc[len(df)] = record    
    
    # pickle the current round
    timestamp = datetime.strftime(datetime.now(), "%m%d_%H%M%S")
    filepath = f'../pickles/{fn}_{timestamp}.pkl' 
    with open(filepath, 'wb') as picklefile:
        pickle.dump(df, picklefile)
    
    driver.close()
    
    return df
    

def multiple_flights(days, flight_list):
    """
    Finds all flights in a list of flight number/route number tuples
    over however many days provided and returns data in a concatenated
    dataframe.
    ---
    IN: number of days to search (int)
        list of flight number/route numbers (string tuples in list)
    OUT: dataframe with all flight info (pandas df)
    """
    
    flight_df = pd.DataFrame(columns=['airline',
                                   'f_num',
                                   'origin',
                                   'dest',
                                   'date',
                                   'land_act',
                                   'arr_act',
                                   'land_sch',
                                   'arr_sch'])
    
    for fn, rn in flight_list:
        print(f"*** {fn} ***")
        flight_df = scrape_fn(days, fn, rn, df=flight_df)

    return flight_df
