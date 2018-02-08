# module for all major functions/classes used in Fletcher

### NECESSARY IMPORTS

# import pandas as pd
import numpy as np
import json
from itertools import chain
from pymongo import MongoClient
from bson.objectid import ObjectId
import re
import string
import pickle
from IPython import display
# import pprint
import matplotlib.pyplot as plt
import seaborn as sns

import enchant
from spacy.en import STOP_WORDS
import spacy
from nltk.metrics.distance import edit_distance
from nltk.stem import (PorterStemmer,
                       LancasterStemmer,
                       SnowballStemmer,
                       RegexpStemmer as REStemmer,
                       WordNetLemmatizer
                      )

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, Normalizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


### OBJECTS/VARS NEEDED IN GLOBAL SPACE

client = MongoClient()
p4_db = client.proj4
lyrics_coll = p4_db.lyrics

eng_dict = enchant.Dict("en_US")
nlp = spacy.load('en')

# for setting random state
rs = 23


### LOADING DATA FROM DATABASE

def lyrics_from_db(coll, 
                   lyric_field,
                   sample_size='all', 
                   min_year=1965, 
                   max_year=1975):
    """
    Returns raw lyrics and corresponding BSON IDs for songs within the min 
    year/max year range (inclusive). Returns all songs in range, or a sample of
    given size.
    ---
    IN
    coll: MongoDB lyrics collection
    lyric_field: exact name of lyric field to load (str)
    sample_size: number of songs in sample (int), or 'all' if all in range 
        (str)
    min_year: lower bound for samples (int)
    max_year: upper bound for samples (int)
    OUT
    lyrics: list of raw lyrics (list)
    song_ids: list of BSON IDs corresponding with lyrics list (list)
    """
    
    if sample_size == 'all':
        docs = coll.find({'Year': {"$gte": min_year, "$lte": max_year},
                          lyric_field: {"$exists": True}})
        
    else:
        assert type(sample_size) == int, "size must be an integer if not 'all'"
        docs = coll.aggregate([{"$match": 
                                {"$and": [{'Year': {"$gte": min_year, 
                                                    "$lte": max_year}},
                                          {lyric_field: {"$exists": True}}
                                         ]}},
                               {"$sample": {"size": sample_size}}
                              ])
    
    lyrics = []
    song_ids = []
    for doc in docs:
        lyrics.append(doc[lyric_field])
        song_ids.append(doc['_id'])
    
    return lyrics, song_ids


def field_by_id(song_id, 
                field,
                min_year=1965, 
                max_year=1975):
    """
    Returns field from lyrics collection given document ID and field name.
    ---
    IN
    coll: MongoDB lyrics collection
    song_id: BSON ID for song document (str)
    field: name of field to retrieve (str)
    min_year: lower bound for samples (int)
    max_year: upper bound for samples (int)
    """

    song = lyrics_coll.find_one({"_id": ObjectId(song_id)})
    return song[field] 


def load_clean_store(coll, 
                     sample_size='all', 
                     min_year=1965, 
                     max_year=1975,
                     return_copy=True):
    """
    Loads raw lyrics from DB, cleans lyrics, and stores clean versions in 
    respective DB documents. Returns lists of cleaned lyrics and song IDs if 
    desired.
    ---
    IN
    coll: MongoDB lyrics collection
    sample_size: number of songs in sample (int), or 'all' if all in range 
        (str)
    min_year: lower bound for samples (int)
    max_year: upper bound for samples (int)
    return_copy: return lists of lyrics and corresponding BSON IDs if true 
        (bool)
    OUT
    clean_lyrics: list of cleaned lyrics (list)
    song_ids: list of BSON IDs corresponding with lyrics list (list)
    fails: list of BSON IDs for documents to which cleaned lyrics could not be
        added (list)
    """

    raw_lyrics, song_ids = lyrics_from_db(coll,
                                          lyric_field='Lyrics',
                                          sample_size=sample_size,
                                          min_year=min_year,
                                          max_year=max_year
                                         )

    assert len(raw_lyrics) == len(song_ids), "unequal numbers of lyrics & IDs"

    fails = []
    clean_lyrics = []
    for song_id, lyric in zip(song_ids[:], raw_lyrics):
        clean_lyric = clean_it_up(lyric)
        result = coll.update_one({"_id": song_id},
                                 {"$set": {"Lyrics_clean": clean_lyric}})
        if result.modified_count == 0:
            print(f"{song_id}: failed to add cleaned lyrics")
            fails.append(song_id)
            song_ids.remove(song_id)
        else:
            clean_lyrics.append(clean_lyric)

    if return_copy:
        return clean_lyrics, song_ids, fails
    else:
        return fails
            

def pull_and_clean_lyrics(coll, size):
    """
    Aggregate the lyrics_from_db() and clean_it_up() functions.

    *** As of now, cannot specify year range in this one, defaults of
    lyrics_from_db() are used. ***
    ---
    IN
    coll: MongoDB collection of songs
    size: number of songs in sample, or 'all' for all (int or str)
    OUT
    lyrics_clean: list of cleaned lyrics (list of strs)
    song_ids: corresponding list of BSON IDs for each song (list)
    """
    
    raw_lyrics, song_ids = lyrics_from_db(coll, 'Lyrics', sample_size=size)
    
    clean_lyrics = []
    for lyric in raw_lyrics:
        clean_lyrics.append(clean_it_up(lyric))
        
    return clean_lyrics, song_ids


### CLEANING & SPELL CHECKING

def clean_it_up(words, spell_check=True):
    """
    Cleaning operations necessary to clean document and prepare for
    tokenization.
    ---
    IN
    words: string to be cleaned (str)
    OUT
    words: cleaned string (str)
    """
    
    # remove any 'word' beginning with a number
    words = re.sub(r'\b\d+\S*\b', '', words)
    
    # remove swaths of whitespace, strip at beginning and end
    words = re.sub(r'\s+', ' ', words).strip()

    # spell check if option selected
    if spell_check:
        words = check_word_string(words)

    # remove all punctuation
    # LATER: include option to keep certain punctuation, e.g. hyphens
    trans = str.maketrans('', '', string.punctuation)
    words = words.translate(trans)
    
    return words


def spell_checker(word, min_ed=2, keep_fails=False):
    """
    Runs several spell-checking operations to clean up issues in the text.
    
    *** Must define eng_dict and edit_distance in the global space! ***
    ---
    IN
    word: word to be spell-checked (str)
    min_ed: minimum edit distance to replace word (int)
    keep_fails: if True, keep words even if none of the replacement methods
        have worked; if False, delete them (bool)
    OUT
    word_checked: list of checked words (list of strs) 
    """

    exceptions = ['im', 'ive', 'aint', 'dont', 'youre']

    if not word:
        return None

    # no further processes need to run if it's a recognizable word
    if eng_dict.check(word.lower()) or word in exceptions:
        return word

    # otherwise...
    try:
        alts = eng_dict.suggest(word.lower())
        ed = edit_distance(word, alts[0])

        # try with proper gerund form (e.g. 'runnin' > 'running')
        if word[-2:] == 'in':
            return spell_checker(word + 'g')
        # can word be edited in min edit distance?
        elif ed <= min_ed:
            return alts[0].lower()
        # try with leading character removed
        elif ed == min_ed + 1 and len(word) > 1:
            return spell_checker(word[1:])
        elif keep_fails:
            return word
        else:
            return None
        
    except IndexError:
        print(f"IndexError while trying to check '{word}'")
        return None

    
def check_word_list(words, min_ed=2, keep_fails=False):
    """
    Feeds word list to spell checker, reassembles, returns.
    
    *** Must import chain from collections! ***
    ---
    IN
    words: words (strs) to be spell-checked (list)
    keep_fails: if True, keep words even if none of the replacement methods
        have worked; if False, delete them (bool)
    OUT
    checked: list of checked/corrected words (list)
    """
    
    checked = []
    
    for word in words:
        new = spell_checker(word, min_ed=min_ed, keep_fails=keep_fails)
        if new:
            checked = list(chain(checked, new.split(' ')))

    return checked


def check_word_string(doc, min_ed=2, keep_fails=False):
    """
    Wrapper for check_word_list(), takes and returns a string of 
    component words.
    
    *** Must import chain from collections! ***
    ---
    IN
    doc: words to be spell-checked (str)
    keep_fails: if True, keep words even if none of the replacement methods
        have worked; if False, delete them (bool)
    OUT
    string of checked/corrected words (str)
    """

    checked = check_word_list(doc.split(' '), 
                              min_ed=min_ed,
                              keep_fails=keep_fails)
    
    return ' '.join(checked)
    
    
def split_word(word, min_len=3):
    """
    Attempts to find a meaningful split of a falsely compounded word.
    ---
    IN
    word: word to split (str)
    min_len: minimum length of first word (int)
    OUT
    word: new string, either original word or successfully-split word (str)
    """
    pass


### PIPELINE FROM VECTORIZATION TO CLUSTERS

def spacy_tokenizer(text,
                    lemmatize=False,
                    stemmer=None,
                    # stemmer=PorterStemmer(),
                    max_wl=2,
                    stopwords=STOP_WORDS, 
                    punctuations=''):
    """
    Basic tokenizer based on Spacy doc object.
    
    *** Must spawn nlp object from spacy in global space! ***
    ---
    IN
    text: string of text to tokenize (str)
    lemmatize: to lemmatize or not to lemmatize (bool)
    stemmer: stemmer object of choice or None if no stemming wanted
    max_wl: maximum word length (int)
    stopwords: stopwords to omit from final list of tokens (set, list)
    punctuations: punctuation to omit from final tokens (set, list)
    OUT
    tokens: final list of word tokens
    """
    
    add_to_stopwords = ['gonna', 
                        'wanna',
                        'whews',
                        'dint',
                        'said',
                        'ooh',
                        'ill',
                        'ive',
                        'vie',
                        'hey',
                        'huh',
                        'gon',
                        'got',
                        'yeah',
                        'whoa',
                        'instrumental', 
                        'interlude', 
                        'miscellaneous']
    for word in add_to_stopwords:
        STOP_WORDS.add(word)

    tokens = nlp(text)

    if lemmatize:
        tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" 
                                             else tok.lower_ 
                                             for tok in tokens]
    # or simply make list of words
    else: 
        tokens = [tok.lower_ for tok in tokens]

    # remove if a stopword or punctuation
    tokens = [tok for tok in tokens if 
              (tok not in stopwords and tok not in punctuations)]
    # catch one- and two-letter words that stopwords didn't get
    tokens = [tok for tok in tokens if len(tok) > 2]
    # stem remaining words
    if stemmer:
        tokens = [stemmer.stem(tok) for tok in tokens]
    
    return tokens


def return_vec_data(text, vectorizer):
    """
    IF a Pandas dataframe is needed...
    Vectorizes a list of text strings with the vectorizer of choice,
    returns the sparse matrix created by the vectorizer, a pandas
    dataframe, and the fit vectorizer object.
    ---
    IN
    text: list of text strings (list)
    vectorizer: pre-spawned vectorizer object
    """
    
    vec_data = vectorizer.fit_transform(text)
    vec_df = (pd.DataFrame(vec_data.toarray(), 
                       columns=vectorizer.get_feature_names()))
    
    return vec_data, vec_df, vectorizer


def display_topics(topic_model, 
                   feature_names, 
                   no_top_words, 
                   topic_names=None):
    """
    Prints given number of words for each topic, and topic names if provided.
    ---
    IN
    topic_model: fit topic model object (TruncatedSVD, NMF, LDA)
    feature_names: word names from vectoriezer object (vec.get_feature_names())
    no_top_words: number of words to display for each topic (int)
    OUT
    text output, topics and top words for each
    """

    for ind, topic in enumerate(topic_model.components_):
        if not topic_names or not topic_names[ind]:
            print("\nTopic ", ind)
        else:
            print("\nTopic: '",topic_names[ind],"'")
        print(", ".join([feature_names[i]
                         for i in topic.argsort()[:-no_top_words - 1:-1]]))


def find_topics(docs, vectorizer, topic_model, verbose=True, n_words=10):
    """
    Takes text, vectorizer object, and topic model object, fits all, and returns
    fit objects and respective data. Also prints topics and words in each one if
    specified.
    ---
    IN
    docs: collection of text strings (list)
    vectorizer: pre-spawned vectorizer object
    topic_model: pre-spawned topic model object
    verbose: True to print topics
    n_words: number of words per topic
    OUT
    vectorizer: fit vectorizer object
    vec_data: vectorizer data (np array)
    topic_model: fit topic model object
    topic_data: topic model data (np array)
    """

    vec_data = vectorizer.fit_transform(docs)
    topic_data = topic_model.fit_transform(vec_data)
    
    if verbose:
        display_topics(topic_model, vectorizer.get_feature_names(), n_words)

    return vectorizer, vec_data, topic_model, topic_data


def draw_dendro(data, l_method='ward', t_mode='mlab', ct=0.7, img_name=None):
    """
    Draws a dendrogram with given data to assist in cluster identification/
    selection.
    ---
    IN
    data: array of data to be clusetered (np array, list, etc.)
    link_method: method for calculating linkage for each new cluster, can be
        single, complete, average, weighted, centroid, median, or ward (str)
    trunc_mode: truncation mode, if any, can be None, mlab, lastp, or level 
        (str)
    img_name: name of output file without extension (str) or leave as None if 
        no need to save the image 
    OUT
    No return
    """

    # plt.clf()
    Z = linkage(data, method=l_method)
    plt.figure(figsize=(16,10), dpi=200)
    dendrogram(Z, truncate_mode=t_mode, color_threshold=ct*max(Z[:,2]))
    if img_name:        
        fpath = "../img/" + img_name + ".png"
        plt.savefig(fpath, dpi=200, bbox_inches = 'tight')
    plt.show()


class Tester:
    """
    Parent class for model tester classes. Provides the following methods:
    loop(): loop the self.go() method to text parameters
    save_myself(): save current configuration of object as a .pkl file (takes
        filename from object parameters and filename prefix, if provided)
    """

    def __init__(self, fn_prefix):
        self.fn_prefix = fn_prefix
        self.current_params = []

    def loop(self):
        """
        Loops the self.go() function to test parameters.
        """

        while True:
            self.go()
            print("\nGo again? (y/n)")
            if input("> ").lower() == 'y':
                display.clear_output()
            else:
                break

    def save_myself(self):
        """
        Saves current object as a .pkl file.
        """

        fname = self.fn_prefix + '_'.join(list(map(str, self.current_params)))
        fname = fname.replace('.','')
        with open('../data/' + fname + '.pkl', 'wb') as pf:
            pickle.dump(self, pf)

    def go(self):
        pass
        

class TrySomeStuff(Tester):
    """
    Class used for rapid testing of vectorizer and topic models, given user
    input. Fit models, parameters, and output data are saved/accessible.
    """

    def __init__(self, docs, n_words=10, norm=True, fn_prefix=''):
        Tester.__init__(self, fn_prefix)
        self.docs = docs
        self.n_words = n_words
        self.norm = norm
        self.vec = None
        self.tm = None
        self.nm = None
        self.ct = 0.7

    def draw_dendro(self, l_method='ward', img_name=None):
        """
        Draws dendrogram with module function.
        """

        draw_dendro(self.topic_data, 
                    l_method=l_method, 
                    ct=self.ct, 
                    img_name=img_name
                   )

    def show_topics(self):
        """
        Prints topics for current model selections using module function.
        """

        print("Number of words per topic (default 10):")
        usr_in = input("> ")
        if usr_in: self.n_words = int(usr_in)
        display_topics(self.tm, self.vec.get_feature_names(), self.n_words)
         
    def print_models(self):
        """
        Prints vectorizer and topic models to show types and parameters.
        """

        print("\nVECTORIZER:\n")
        print(self.vec)
        print("\nTOPIC MODEL:\n")
        print(self.tm)
    
    def go(self):
        """
        Run all the things.
        """

        self.current_params = []

        # choose type and params for vectorizer
        print("-- VECTORIZER --\n")
        print("Choose your vectorizer type, CV (1) / TFIDF (2):")
        usr_in = input("> ")
        if usr_in: self.vec_type = usr_in.lower()
        if self.vec_type == '1':
            self.current_params.append('cv')
        if self.vec_type == '2': 
            self.current_params.append('tfidf')
    
        print("Max features (~7000 words post-tokenizer):")
        usr_in = input("> ")
        if usr_in: self.max_feat = int(usr_in)
        self.current_params.append(self.max_feat)

        print("Max document frequency (0-1):")
        usr_in = input("> ")
        if usr_in: self.max_df = float(usr_in)
        self.current_params.append(self.max_df)
    
        print("Max n-gram length:")
        usr_in = input("> ")
        if usr_in: self.max_ngram = int(usr_in)
        self.current_params.append(self.max_ngram)
    
        if self.vec_type == '1':
            print("Binary, True (T) / False (F):")
            usr_in = input("> ")
            if usr_in and usr_in.lower() == 't': 
                self.binary = True
            if usr_in and usr_in.lower() == 'f':
                self.binary = False
            if self.binary == True:
                self.current_params.append('bin')
    
        # choose type and params for topic model
        print("\n-- TOPIC MODEL --\n")
        # add LDA later 
        print("Choose your topic model, LSA (1) / NMF (2):")
        usr_in = input("> ")
        if usr_in: self.tm_type = usr_in.lower()
        if self.tm_type == '1':
            self.current_params.append('lsa')
        if self.tm_type == '2':
            self.current_params.append('nmf')
    
        print("Number of components:")
        usr_in = input("> ")
        if usr_in: self.n_comps = int(usr_in)
        self.current_params.append(self.n_comps)
    
        # define vectorizer based on input
        print("\n-- SPAWNING MODELS --")
        if self.vec_type == '1':
            self.vec = CountVectorizer(tokenizer=spacy_tokenizer, 
                                       ngram_range=(1,self.max_ngram),
                                       max_features=self.max_feat,
                                       binary=self.binary,
                                       # min_df=0.02
                                       max_df=self.max_df
                                      ) 
        elif self.vec_type == '2':
            self.vec = TfidfVectorizer(tokenizer=spacy_tokenizer,
                                       ngram_range=(1,self.max_ngram),
                                       max_features=self.max_feat,
                                       # min_df=0.02
                                       max_df=self.max_df
                                      ) 
        else:
            print("Vectorizer type invalid!")
            self.go()
    
        # define topic model based on input
        if self.tm_type == '1':
            self.tm = TruncatedSVD(n_components=self.n_comps, random_state=rs)
        elif self.tm_type == '2':
            self.tm = NMF(n_components=self.n_comps, random_state=rs)
        else:
            print("Topic model invalid!")
            self.go()
    
        # prints models to confirm choices
        self.print_models()
        print("\nHit enter to continue or X to start over:")
        if input("> ").lower() == 'x':
            self.go()

        # fit vectorizer
        self.vec_data = self.vec.fit_transform(self.docs)
        # normalize
        if self.norm == True:
            self.nm = Normalizer()
            self.vd_norm = self.nm.fit_transform(self.vec_data)
        # fit topic model
        self.topic_data = self.tm.fit_transform(self.vd_norm)
        
        # show topics
        print("\n-- TOPICS --")
        print("\nDisplay topics? (y/n)")
        if input("> ").lower() == 'y': 
            self.show_topics()

        # print dendrogram
        print("\n-- DENDROGRAM --")
        print("\nDraw dendrogram? (y/n)")
        if input("> ").lower() == 'y': 
            print("Enter color threshold (default 0.7):")
            usr_in = input("> ")
            if usr_in: self.ct = float(usr_in)
            self.draw_dendro()

        print("\nPickle current object? (y/n)")
        if input("> ").lower() == 'y':
            self.save_myself()
        

class TrySomeClusters(Tester):
    """
    Class used to rapidly test different clustering algorithms and associated
    parameters, which are provided by user input.
    ---
    INIT PARAMS
    topic_data: topic-space vector array (np array)
    song_ids: list of song IDs corresponding with rows of topic data (np array)
    fn_prefix: prefix to use for naming .pkl files, optional (str)

    METHODS
    go(): set up algorithm and parameters, check silhouette score, songs in
        clusters
    sample_cluster_text(): samples lyrics of a given number of songs from each
        cluster, given clusters from current model
    agg_cluster_range(): test a range of cluster numbers and show silhouette
        score for each one
    """
    
    def __init__(self, topic_data, song_ids, fn_prefix=''):
        Tester.__init__(self, fn_prefix)
        self.X = topic_data
        self.y = None
        self.song_ids = song_ids
        self.model = None
        self.sil_score = None
        self.eps = 0.5
        self.min_samp = 5
        
    def go(self):
        """
        Set up parameters for one pass with one clustering algorithm. Displays 
        silhouette score, size of each cluster, and sample songs from each 
        cluster, querying from MongoDB.
        """
        
        self.current_params = []
        
        # choose type of clustering algo and params
        print("\n-- CLUSTERING PARAMETERS --\n")
        print("Algorithm, Agg (1) / DBSCAN (2):")
        usr_in = input("> ")
        if usr_in: self.algo_type = usr_in
        if self.algo_type == '1':
            self.current_params.append('agg')
        elif self.algo_type == '2':
            self.current_params.append('dbs')
        else:
            print("Invalid input")
            self.go()
        
        if self.algo_type == '1':
            print("Number of clusters:")
            usr_in = input("> ")
            if usr_in: self.n_clust = int(usr_in)
            self.current_params.append(self.n_clust)

            result = self.set_link_method()
            if result:
                print("Invalid input")
                self.go()

        if self.algo_type == '2':
            print("Epsilon (default 0.5):")
            usr_in = input("> ")
            if usr_in: self.eps = float(usr_in)
            self.current_params.append(self.eps)
            
            print("Min samples (default 5):")
            usr_in = input("> ")
            if usr_in: self.min_samp = int(usr_in)
            self.current_params.append(self.min_samp)

        # spawning and fit/predict
        print("\n-- FIT AND PREDICT --\n")
        
        if self.algo_type == '1':
            self.agg()

        if self.algo_type == '2':
            self.model = DBSCAN(eps=self.eps, min_samples=self.min_samp)
            self.y = self.model.fit_predict(self.X)

        print(self.model)

        # calcluate and print silhouette score
        self.sil_score = silhouette_score(self.X, self.y)
        print("\nSilhouette score:", self.sil_score)

        # print number of points in each cluster
        print("\nMembers per Cluster:")
        for i, num in enumerate(np.bincount(self.y)):
            print(f"*   Cluster {i}: {num}")

        # print sample text from each cluster
        print("\nShow text samples from each cluster? (y/n):")
        if input("> ").lower() == 'y':
            self.sample_cluster_text()

        print("\nPickle current object? (y/n)")
        if input("> ").lower() == 'y':
            self.save_myself()

    def set_link_method(self):
        """
        User prompt for linkage method.
        """
        print("Linkage method, ward (1), complete (2), average (3):")
        usr_in = input("> ")
        if usr_in: self.link_type = usr_in
        if self.link_type == '1':
            self.link = 'ward'
            self.current_params.append('ward')
        elif self.link_type == '2':
            self.link = 'complete'
            self.current_params.append('comp')
        elif self.link_type == '3':
            self.link = 'average'
            self.current_params.append('avg')
        else:
            return 'error'

    def sample_cluster_text(self, n_songs=5, text_detail=500):
        """
        Displays n_songs from each cluster (song title and lyrics).
        ---
        IN
        n_songs: number of songs to display per cluster (int)
        text_detail: character length of lyric excerpt (int)
        OUT
        None
        """

        assert_msg = "Song IDs and cluster labels of unequal length" 
        assert len(self.song_ids) == len(self.y), assert_msg
 
        for i in range(len(np.bincount(self.y))):
            cluster_ids = []
            print(f"\nCluster {i}:")
            for cluster, song_id in zip(self.y, self.song_ids):
                if cluster == i:
                    cluster_ids.append(song_id)
            sample_size = min(len(cluster_ids), n_songs)
            sample_ids = np.random.choice(cluster_ids, sample_size, False)
            for song_id in sample_ids:
                song = lyrics_coll.find_one({'_id': ObjectId(song_id)})
                print(f"\nTitle: {song['Song'].title()}")
                print(song['Lyrics_clean'][:text_detail])

    def agg(self, n_clust=None, link=None):
        """
        Fits/predicts model using agglomerative clustering, params can 
        be provided at function call or determined in a previous function 
        (default to the latter).
        ---
        IN
        n_clust: number of clusters (int)
        link: linkage method (str)
        OUT
        None
        """

        if not n_clust:
            n_clust = self.n_clust
        if not link:
            link = self.link
 
        self.model = AgglomerativeClustering(n_clusters=n_clust,
                                             linkage=link)
        self.y = self.model.fit_predict(self.X)

    def agg_cluster_range(self, c_min=2, c_max=None, link=None):
        """
        Fits/predicts model using agglomerative clustering over a range of 
        cluster numbers, which can be provided as kwargs, or entered by prompt, 
        then calculates total silhouette score for each number of clusters. 
        Linkage method defaults to whatever the last value was, or can be 
        specified as kwarg. Object will retain maximum number of clusters as 
        the number of clusters.
        ---
        IN
        c_min: minimum number of clusters (int)
        c_max: maximum number of clusters (int)
        link: linkage method (str)
        OUT
        None
        """

        if not c_max:
            print("Minimum number of clusters:")
            c_min = int(input("> "))
            print("Maximum number of clusters:")
            c_max = int(input("> "))

        if not link:
            result = self.set_link_method()
            if result:
                print("Invalid input")
                self.go()

        print("Clusts\tSilhouette Score:")    
        for n in range(c_min, c_max+1):
            self.agg(n_clust=n)
            self.sil_score = silhouette_score(self.X, self.y)
            print(f"{n}\t{self.sil_score}")


def get_similar(song_id, song_ids, topic_vectors, n_sim=10):
    """
    Returns top similar songs and associated data (distances, indices) given the
    BSON ID of a song in the lyrics collection.
    ---
    IN
    song_id: BSON ID of song for which to find similar songs (str)
    song_ids: list of all song IDs considered for similairity (list)
    topic_vectors: topic space array from which to caluclate similarity
        (np array)
    n_sim: number of similar songs (int)
    OUT
    sim_songs: song title, artist tuples of top similar songs (list)
    dists: distances of ranked similar songs as calculated by NearestNeighbors
        (np array)
    indices: indices of ranked similar songs in topic space array (np array)
    """
    
    assert len(song_ids) == len(topic_vectors), "Lists of unequal length"
    
    ix = song_ids.index(ObjectId(song_id))
    song_vec = topic_vectors[ix]
    nn = NearestNeighbors(n_neighbors=n_sim+1, 
                          metric='cosine', 
                          algorithm='brute'
                         )
    nn.fit(topic_vectors)
    nn_data = nn.kneighbors(song_vec.reshape(-1,1).T)
    dists = nn_data[0][0][1:]
    indices = nn_data[1][0][1:]
    
    sim_songs = []
    for nn_ix in indices:
        title = field_by_id(song_ids[nn_ix], 'Song')
        artist = field_by_id(song_ids[nn_ix], 'Artist')
        sim_songs.append((title, artist))
    
    return sim_songs, dists, indices


def display_sim_songs(song_id, sim_songs, dists):
    """
    Prints top similar songs and their corresponding cosine similarities given 
    a list of those songs and a corresponding list of distances. Cosine 
    similarity is found by subtracting the distance from 1, as per sklearn's 
    convention for returning distance from NearestNeighbors when 'cosine' is 
    selected as metric.
    ---
    IN
    song_id: BSON ID of song for which similarities have been calculated (str)
    sim_songs: list of similar songs generated by get_similar() function (list)
    dists: array of distances generated by get_similar() function (np array)
    OUT
    None
    """

    title = field_by_id(song_id, 'Song')
    artist = field_by_id(song_id, 'Artist')
    print(f"\nSimilar to {title.title()} by {artist.title()}:")
    for song, dist in zip(sim_songs, dists):
        print(f"{song[0].title()} - {song[1].title()} ({round(1 - dist, 3)})")
