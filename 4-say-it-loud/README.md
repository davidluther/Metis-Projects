# Themes of Social Awareness in Lyrics, 1965-1975

Lyrics in popular music during the early 60s were largely inert, with plenty of songs about idealized love, dance fads of the day, or outright nonsense (think "Wooly Bully"). 1965 saw the assassination of Malcolm X. During the same year, regular deployment of troops began in Vietnam, known as the "start" of the war for the general population. Race riots plagued inner cities throughout the latter half of the decade. In April 1968, MLK was assasinated, and then Bobby Kennedy, one month later. The forward momentum of the Civil Rights Movement slowed to a crawl, and a general haze of disillusionment overtook any sense of hope that the world was becoming a better place. In soul music, lyrics wholly reflected this turn, with themes of isolation and entrenchment, of distrust, of taking care of your own because nobody else will.

Using [a dataset of the Billboard Year-End Hot 100 songs from 1965-1975](https://github.com/walkerkq/musiclyrics), I applied NLP, topic modeling, and clustering to see if I could tease out these emergent themes. I also used a basic recommender system to see what songs it would suggest as similar to songs known to exemplify these themes. Lyrics and assorted song info were stored in MongoDB for access via PyMongo throughout.

## Folder Contents

### Code

* **10-mongo-basic-training.ipynb**: PyMongo training steps and creation of lyrics collection
* **11-DB-formatting.ipynb**: cleaning and formatting of lyrics collection
* **20-pipeline-dev.ipynb**: first draft of entire pipeline, from cleaning of lyrics through clustering in topic space
* **21-preprocessing.ipynb**: preprocessing of lyrics, everything prior to vectorization, using functions of project module
* **22-vec-and-beyond.ipynb**: streamlined process for choosing vectorizer and topic modeling method, looking at similar songs by cosine similarity, and testing clustering algorithms
* **fletchmod.py**: module containing all necessary functions and classes for use throughout the project

### Img

Images generated for use in presentation. (At this point, there is only a sample dendrogram generated from the final topic-space vector array.)

### Pres

PDF of final presentation.
