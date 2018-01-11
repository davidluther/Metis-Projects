# Modeling Arrival Delay at [Airport]

## Objective (dynamic, updated daily)
Look at all the American flights from LGA to ORD for one year (BTS), train a model with delay (Arr<sub>actual</sub> - Arr<sub>sched</sub>) as target and day of week as the single feature. Test against as much FlightAware data as I can pull down in a week.

## Potential Data Sources
* [FlightAware](https://flightaware.com/)
* [Flightradar24](https://www.flightradar24.com/data/)
* [FlightStats](https://www.flightstats.com/go/Home/home.do)
* [Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
* [FlightView](https://www.flightview.com/)
* [OpenFlights](https://openflights.org/data.html)

#### FlightAware
*Use to scrape test data*  
Only caches records for one week, but offers historical data for purchase. Airport arrival tables only show actual arrival time, not scheduled, but the actual flight page shows both. 

#### Flightradar24
Very little in the way of historical, at least at first glance!

#### FlightStats
Focused more on individual flights than airports. No direct access to historical data without professional subscription ($25/mo), but gives some dubious stats on delays. Offers 30 day trial account for APIs though.

#### BTS
*Download CSV as traning data*  
Lots and lots available here, but it's rather impenetrable at first glance, and all .aspx. TVC keeps on coming up blank, but there are some decent records for DTW and other bigger ones. Try:
* https://www.transtats.bts.gov/ONTIME/Index.aspx – No TVC! But yes to DTW. Would have to use Selenium and then Soup.
* https://www.bts.gov/topics/airlines-and-airports/quick-links-popular-air-carrier-statistics – lots and lots of things
* https://www.transtats.bts.gov/databases.asp?Mode_ID=1&Mode_Desc=Aviation&Subject_ID2=0 – perhaps the richest mine from the above
* https://www.faa.gov/airports/airport_safety/airportdata_5010/ – not sure how useful this is for this project, but might be helpful down the road

#### FlightView
TVC taps their feed for site stats. Check back later, got a 500 Internal Server Error on first try.

#### OpenFlights
Not sure what exaclty this is...

## How to Scrape Flight Aware Data
* Build list of flight numbers by AAL from LGA to ORD by searching ORD arrivals and adding flight number to a set everytime LGA is contained in the origin field
* Flight number website has a link to history of this form: http://flightaware.com/live/flight/AAL321/history/
* Appending a number to the end of it will display up to that number of records, and will not break if the number is very large (e.g. AAL321 displays somewhere around 160 records, and they will all display if 2000 is added to the end of the link). *This provides data for the last four months.*
* Clicking on the date provides a page for that individual flight: http://flightaware.com/live/flight/AAL321/history/20170928/0130Z/KLGA/KORD
* One could easily just fill in the date here, and while the flight number is not unique to that route, the 0130Z is. This page provides all sorts of stats, including sched/actual arrival time, average price per ticket, etc.
* **BUT WAIT...** Even if you reach the last link on the history page, it can be hacked by changing the date in the URL. Not sure what the limit is on this one.
