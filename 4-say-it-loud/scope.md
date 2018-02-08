# Lyrical Themes from 60s to 70s

## Overview

Lyrics in popular music during the 60s were largely inert, with plenty of songs about idealized love, dance fads of the day, or outright nonsense (think "Wooly Bully," the number one hit of 1965). With the Civil Rights Movement gaining momentum in the mid-60s, bringing with it a palpable possibility of change, themes of love and hope worked their way throughout as well. 1965 saw the assassination of Malcolm X and the Watts race riots During the same year, regular deployment of troops began in Vietnam, known as the "start" of the war for the general population. (selective service?) The streets of Detroit erupted into riots in the spring of 1967. In April 1968, MLK was assasinated, and then Bobby Kennedy, one month later. The forward momentum of the Civil Rights Movement slowed to a crawl, and a general haze of disillusionment overtook any sense of hope that the world was becoming a better place. In soul music, lyrics wholly reflected this turn, with themes of isolation and entrenchment, of distrust, of taking care of your own because nobody else will.

I'll be collecting lyrics from 1965-1975, minimum, and hopefully from as early as 1960, if not before, and then using NLP and dimension reduction (perhaps more?) to tease out themes present over time. If possible, I'd like to see how this ties in with genre as well.

## Project Requirements

As listed on Fletcher intro page...

### Data
* TYPE: text data
* ACQUISITION: API, scraping, etc.
* STORAGE: MongoDB

### Skills & Tools
* Flask — *hopefully/maybe*
* MongoDB — *yes*
* NLP — *yes*
* Unsupervised Learning — *yes*
* Dimensionality Reduction — *yes*
* Topic Modeling — *maybe?*
* Recommender Systems — *not sure if or how*

## Wish List/Order Of Attack

MVP: must happen  
II: second phase  
III: third phase  
MOON: wouldn't it be cool if...

### Data
* Store lyrics in MongoDB (MVP)
* Clean lyrics (MVP)
* Acquire missing lyrics (II)
* Acquire lyrics for 1960-1965 (III)

### Analysis
* NLP on phrases for sentiment (MVP)
* NLP on words for similarity (MVP)
* Dimensionality reduction to distill concepts (MVP/II)
* Unsupervised learning to find possible clusters (MVP/II)

### Viz
* Concepts (II)
* Themes (clusters) (II)
* Timeline of themes with major cultural events (II)

### Other
* Scrape weekly Hot 100 from PDFs of Billboard (MOON)
* Build scraper to collect Hot 100, #1 R&B & Country from Wikipedia (II/III)
* Build scraper to collect lyrics from Lyrics A to Z and other web site as necessary (II/III)
* Build Flask app to do ___ (II/III)
