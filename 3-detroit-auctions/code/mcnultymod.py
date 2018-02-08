# module for all the heavy-lifting functions needed in McNulty
import os
import pandas as pd
import numpy as np
import pickle
import warnings

from sqlalchemy import create_engine
from geopy.distance import vincenty

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import (train_test_split, 
                                     cross_validate,
                                     cross_val_score, 
                                     cross_val_predict,
                                     learning_curve,
                                     GridSearchCV,
                                     RandomizedSearchCV
                                    )


class WeightByProximity(object):
    """
    Includes functions to assign a weight to a particular location based on its
    proximity to a list of other locations based on latitude/longitude. Can be 
    set to return a binary weight if any other locations fall within a certain
    threshold, or the count of other locations that fall within a certain
    threshold. The fit() method must be run before using the prox_weight()
    method.
    """
    
    def __init__(self, lat_col='latitude', lng_col='longitude'):
        self.lat_col_ = lat_col
        self.lng_col_ = lng_col
        self.fit_ = False

        
    def fit(self,
            fit_df, 
            match_column,
            lat_filter=0.015, 
            lng_filter=0.02):
        """
        Loads table containing latitude/longitude information to be used to
        provide proximity weights for other table(s).  
        ---
        IN
        fit_df: dataframe containing latitude and longitude columns to be used 
            in calculation of weights (df)
        match_column: set to the name of a column appearing in both dfs to 
            filter based on matching value. COMING SOON: leave as None if no 
            filter needed, ignore matching lat/lng pairs option. (str)
        lat_filter: latitude delta that should be close to 1 mi, or slightly 
            larger -- used to create a sub-table for precise calculations.
            Preset works for a latitude of ~43. (float)
        lng_filter: longitude delta close to 1 mi, or slightly larger (float)
        """

        if self.fit_ == True:
            print("ERROR: Object already fit!")
            return None
        
        self.fit_ = True
        self.match_col_ = match_column
        self.lat_filter_ = lat_filter
        self.lng_filter_ = lng_filter
        self.match_df_ = (fit_df
         .filter([self.match_col_, 'latitude', 'longitude']))
        
        
    def prox_weight(self, 
                    current, 
                    threshold, 
                    mode='binary', 
                    coarse_filter=True):
        """
        Calculates weight of a location based on proximity to locations provided
        in the fit dataframe. 
        ---
        IN
        current: row of dataframe for which to calculate proximity weight
        threshold: length of radius (in miles) for comparison (float)
        mode: 'binary' or 'cumulative' -- if 'binary', will return 0 if no 
            location matches found within threshold, and 1 if at least one. 
            If 'cumulative', will return the number of matches found within 
            threshold. (str)
        coarse_filter: cuts down on computation time by filtering lat/lng values
            outside of a certain perimeter from the current lat/lng.
            (bool)
        OUT
        weight: proximity weight for location (df)
        """
        
        if self.fit_ == False:
            print("ERROR: You haven't fit the object, do that first.")
            return None
        
        weight = 0
        lat_in = current[self.lat_col_]
        lng_in = current[self.lng_col_]
        
        if coarse_filter == True:
            lat_lim = threshold * self.lat_filter_
            lng_lim = threshold * self.lng_filter_
            match_df = (self.match_df_[
                (self.match_df_[self.lat_col_] > (lat_in - lat_lim)) &
                (self.match_df_[self.lat_col_] < (lat_in + lat_lim)) &
                (self.match_df_[self.lng_col_] > (lng_in - lng_lim)) &
                (self.match_df_[self.lng_col_] < (lng_in + lng_lim))
                ])
        else:
            match_df = self.match_df_

        for _, row in match_df.iterrows():
            lat = row[self.lat_col_]
            lng = row[self.lng_col_]

            if current[self.match_col_] == row[self.match_col_]:
                continue
            else:
                dist = vincenty((lat, lng), (lat_in, lng_in)).miles

            if dist < threshold and mode == 'binary':
                weight = 1
                break
            elif dist < threshold and mode == 'cumulative':
                weight += 1
            else:
                continue

        return weight


class BinNeighborhoods(object):
    """
    Bins neighborhoods according to density of investors, with bin bounds 
    provided during fit.
    """
    
    def __init__(self, nh_col='neighborhood'):
        self.nh_col_ = nh_col
        self.bin_bounds_ = None
        self.fit_ = False


    def _nhood_investor_ratios(self, fit_df):
        """
        Calculates the ratio of investors to all buyers per neighborhood, 
        appends as a new column to neighborhoods dataframe.
        ---
        IN
        fit_df: main auction dataframe
        OUT
        nhood_df: updated with investor ratio column
        """

        if self.fit_ == False:
            print("ERROR: You haven't fit the object, do that first!")
            return None
        
        inv_ratios = []
        
        for _, row in self.nhoods_.iterrows():
            nhood = row[self.nh_col_]
            temp = fit_df[fit_df[self.nh_col_] == nhood]
            inv_ratio = (temp[temp.purchasertype == 'Investor'].shape[0] / 
                         temp.shape[0])
            inv_ratios.append(inv_ratio)
            
        self.nhoods_['inv_ratio'] = inv_ratios

        
    def _bin_neighborhoods(self):
        """
        Assigns each neighborhood a bin number based on ratio of investors.
        ---
        IN
        df: neighborhoods dataframe with investor ratios (df)
        binlist: lower bounds for each bin
        OUT
        df: neighborhoods dataframe with investor ratio bins column
        """
        
        if self.fit_ == False:
            print("ERROR: You haven't fit the object, do that first!")
            return None
        
        self.nhoods_['n_bin'] = 0

        for n, val in enumerate(self.bin_bounds_):
            self.nhoods_.loc[self.nhoods_.inv_ratio > val, 'n_bin'] = n
        
    def fit(self, fit_df, lower_bin_bounds=[0,0.1,0.25,0.33,0.45,0.75]):
        """
        Docstring coming soon!
        """
        
        if self.fit_ == True:
            print("ERROR: Object already fit!")
            return None
        
        self.fit_ = True
        self.bin_bounds_ = lower_bin_bounds 
        
        self.nhoods_ = (fit_df[self.nh_col_]
         .value_counts()
         .reset_index()
         .rename(columns={'index': self.nh_col_, self.nh_col_: 'n_count'})
                   )
        
        self._nhood_investor_ratios(fit_df)
        self._bin_neighborhoods()

        
    def merge_bins(self, df, how_merge='left'):
        """
        Docstring coming soon!
        """
        
        if self.fit_ == False:
            print("You haven't fit the object, do that first!")
            return None            
        
        df = pd.merge(df, 
                      self.nhoods_.filter([self.nh_col_, 'n_bin']), 
                      how=how_merge, 
                      on=self.nh_col_)

        # if test set neighborhood is not in train set, set n_bin to 3
        # which is where the bin in which the average investor ratio falls
        df.n_bin.fillna(3, inplace=True)

        return df


def connect_to_sql(aws_ip):
    """
    Connects to PostgreSQL on AWS instance.
    ---
    IN
    aws_ip: current IP address of AWS instance (str)
    OUT
    engine: engine object
    """

    engine_name = ("postgresql://" + 
                   str(os.environ['psqlUN']) + ":" + 
                   str(os.environ['psqlPW']) + "@" +
                   aws_ip +
                   ":5432/detroit")
    engine = create_engine(engine_name)
    
    return engine


def fetch_data(table_name, aws_ip):
    """
    Grabs requested table from PostgreSQL on AWS instance and returns as a 
    dataframe.
    ---
    IN
    table_name: exact name of table in PSQL (str)
    aws_ip: current IP address of AWS instance (str)
    OUT
    df: pandas dataframe
    """

    cnx = connect_to_sql(aws_ip)

    df = pd.read_sql_query('''SELECT * FROM ''' + table_name, cnx)

    return df


def tts(df, target_col, test_size=0.2, random_seed=23):
    """
    Quick and dirty train-test-split to label training and test groups in main
    df.
    ---
    IN
    df: dataframe to be split
    target_col: name of target column (str)
    test_size: set test group size, number between 0-1 (float)
    random_state: set random state (int)
    OUT
    df_train: training dataframe
    df_test: training dataframe
    """

    ### ADD A STRATIFY OPTION HERE

    np.random.seed(random_seed)
    train_test = np.random.choice(['train','test'], 
                                  size=df.shape[0], 
                                  p=[1-test_size, test_size],
                                 )

    df['tts'] = train_test
    df_train = df[df.tts == 'train']
    df_test = df[df.tts == 'test']

    return df_train, df_test


def convert_cash(price):
    """
    Converts price string to float.
    ---
    IN
    price: dollar amount (str)
    OUT
    price_float: price (float)
    """

    if price == None:
        return None

    trans = str.maketrans('','',',$')
    price_float = float(price.translate(trans))

    return price_float


def split_and_clean(df, ignore_warnings=False):
    """
    Performs train-test-split, then cleaning and formatting options on both as 
    developed from the training set and explained in the 0-McNulty-Exploration 
    notebook.
    ---
    IN
    df: dataframe to be split and cleaned
    OUT
    df_train, df_test: cleaned and formatted train/test dataframes
    """
    
    df.dropna(inplace=True)

    df_train, df_test = train_test_split(df, 
                                         test_size=0.2, 
                                         random_state=23, 
                                         stratify=df.purchasertype
                                        )

    for df_sub in [df_train, df_test]:
        df_sub = clean_and_format(df_sub)

    return df_train, df_test


def check_ratios(df_train, df_test):
    """
    Checks ratios of Investors and Homebuyers in test and train sets, prints
    values for each.
    ---
    IN
    df_train: training dataframe
    df_test: testing dataframe
    OUT
    None
    """

    for which_set, df in zip(["TRAIN", "TEST"], [df_train, df_test]):
        inv = df.purchasertype.value_counts()[1] / df.shape[0]
        hb = df.purchasertype.value_counts()[0] / df.shape[0]
        print(f"\n{which_set}")
        print("Investor ratio:", inv)
        print("Homebuyer ratio:", hb)

    
def clean_and_format(df):
    """
    Basic repeated cleaning and formatting operations on dataframe, to be 
    repeated for test and train.
    """

    df.price = df.loc[:, 'price'].apply(lambda p: convert_cash(p))
    
    # drop salestatus, buyerstatus, program, location
    # order with target as first column
    df = df.filter(['purchasertype', 
                    'parcelid', 
                    'address', 
                    'price', 
                    'closingdate', 
                    'councildistrict', 
                    'neighborhood',
                    'latitude',
                    'longitude'
                    ])
    
    return df


def add_permit_counts(df, permits):
    """
    Adds a column to auctions dataframe indicating the number of building
    permits issued during the relevant timeframe for the corresponding parcel.
    """

    # make temp permits df matching with dates in auctions df
    permits_temp = pd.merge(df.filter(['parcelid', 'closingdate']), 
                                      permits, 
                                      how='inner', 
                                      on='parcelid'
                                     ) 
    
    # take out any permits that were issued more than 30 days before
    # auction closing date
    permits_temp = (permits_temp[
                    (permits_temp.closingdate - 
                    permits_temp.dateissued < 
                    timedelta(days=30))]
                   )

    # create df of permit counts by parcel ID
    permit_counts = permits_temp.parcelid.value_counts()
    permit_counts.name = 'num_permits'
    permit_counts = pd.DataFrame(permit_counts).reset_index()
    permit_counts.head()

    # merge permit counts into auctions df
    df = pd.merge(auctions, permit_counts, how='left', on='parcelid')
    df.fillna(0, inplace=True)

    return df


def rand_investor_ratios(df, max_size=100, samples=200):
    """
    Calculates investors/total buyers ratio for a given number of random 
    samples ranging in size from 1 to the given maximum.
    ---
    IN
    df: auctions dataframe
    max_size: maximum sample size (int)
    samples: number of samples (int)
    OUT
    sample_sizes: list of sample sizes (list of ints)
    investor_ratios: list of corresponding investor ratios (list of ints)
    (both in a tuple)
    """
    
    sample_sizes = []
    inv_ratios = []
    
    for i in range(samples):
        size = np.random.randint(1,max_size+1)
        temp = df.sample(size)
        inv_ratio = (temp[temp.purchasertype == 'Investor']
                     .shape[0] / temp.shape[0])
        sample_sizes.append(size)
        inv_ratios.append(inv_ratio)

    return sample_sizes, inv_ratios


def rand_investor_ratios2(df, samples, random_seed=23):
    """
    Same as above, but uses geometric distribution to generate sample sizes.
    ---
    IN
    df: auctions dataframe
    samples: number of samples (int)
    random_seed: random seed for numpy generator (int)
    OUT
    sample_sizes: list of sample sizes (list of ints)
    investor_ratios: list of corresponding investor ratios (list of ints)
    (both in a tuple)
    """
    
    np.random.seed = random_seed

    #sample_sizes = np.random.geometric(p=0.05, size=samples)
    sample_floats = np.random.gamma(0.4,20,113)
    sample_sizes = [int(i)+1 for i in sample_floats]
    inv_ratios = []
    
    for size in sample_sizes:
        temp = df.sample(size)
        inv_ratio = (temp[temp.purchasertype == 'Investor']
                     .shape[0] / temp.shape[0])
        inv_ratios.append(inv_ratio)

    return sample_sizes, inv_ratios


def make_list_if_not(thing):
    """
    Checks to see if an item is a list, and if not, turns it into one.
    ---
    IN
    thing: a thing of any datatype
    OUT
    thing as a list (list)
    """

    if type(thing) != list:
        thing_list = []
        thing_list.append(thing) 
        return thing_list
    else:
        return thing


def feature_eng(dfs_in, binary_thresholds=[0.1], cumulative_thresholds=[0.5]):
    """
    Performs certain transforms on features of training and test sets based on 
    characteristics of training set.
    ---
    IN
    dfs_in: single training set, or list of training and test set. IF THE
        LATTER, TRAINING SET MUST BE FIRST! (dfs)
    binary_thresholds: list of 
    OUT
    
    """

    dfs = make_list_if_not(dfs_in)
#    if type(dfs_in) != list:
#        temp = dfs_in
#        dfs_in = []
#        dfs_in.append(temp)

    # add binary column for Investor/Homebuyer
    # and bin neighborhoods by investor ratio
    bn = BinNeighborhoods()
    bn.fit(dfs[0])
    for n, df in enumerate(dfs[:]):
        df['ptype_num'] = 0
        df.loc[df.purchasertype == 'Investor', 'ptype_num'] = 1
        print("Inv/HB column binary")
        df = bn.merge_bins(df)
        print("Neighborhoods binned")
        dfs[n] = df

    # add bin columns for proximity to other auction properties
    pw = WeightByProximity()
    pw.fit(dfs[0], match_column='parcelid')
    for n, df in enumerate(dfs[:]):
        for i, thresh in enumerate(binary_thresholds):
            df['thr_' + str(i)] = (df.apply(
                lambda row: pw.prox_weight(row, thresh),
                axis=1))
        print("First proximity columns added")
        for i, thresh in enumerate(cumulative_thresholds):
            df['tcount_' + str(n)] = (df
             .apply(lambda row: pw.prox_weight(
                        row, 
                        thresh, 
                        mode='cumulative'), 
                    axis=1))
        print("Second proximity columns added")
        dfs[n] = df

    # bin by price 
    for n, df in enumerate(dfs[:]):
        df['p_bin'] = 0
        df.loc[df.price > 5000, 'p_bin'] = 1
        df.loc[df.price > 20000, 'p_bin'] = 2 
        print("Binned by price")

    print("Done!")

    if len(dfs) == 1:
        return dfs[0]
    else:
        return tuple(dfs)


def prep_X(cat_cols, Xs_in):
    """
    Runs feature array through pre-modeling preparation, including setting
    type of columns as category, turning them into dummy columns, and scaling
    numerical data.
    ---
    IN
    cat_cols: list of columns to for which type will be set to 
    Xs_in: list of X arrays to prep, X_train must be first if with X_test
    OUT
    X_train_std, X_test_std: features sets standardized for modeling (np arrays)
    """

    Xs = make_list_if_not(Xs_in)

    Xs_dums = []
    Xs_scaled = []
    
    for X in Xs:
        for col in cat_cols:
            X[col] = X[col].astype('category')
        X = pd.get_dummies(X, drop_first=True)
        Xs_dums.append(X)
        
    # build scaler on X_train, scale X_train and X_test with it
    ss = StandardScaler()
    ss.fit(Xs_dums[0])
    
    for X in Xs_dums:
        X = ss.transform(X) 
        Xs_scaled.append(X)
    
    if len(Xs_scaled) == 1:
        return Xs_scaled[0]
    else:
        return tuple(Xs_scaled)
    # return tuple(Xs_scaled)


def print_knn_scores(knn_scores):
    """
    Prints KNN scores in table format. No return.
    ---
    IN
    knn_scores: scores as dict output of try_some_ks()
    """

    print("K-value\tAcc.\tPre.\tRec.\tF1")
    
    for item in knn_scores.items():
        s = str(item[0])
        for score in item[1]:
            s += ('\t' + str(score))
        print(s)


def print_scores(model_name, score_list):
    """
    Prints out four scores in a format pleasant to the eye.
    ---
    IN
    model_name: name of model tested (str)
    score_list: list of scores generated by model-tester function (list
    """
    
    score_names = ['Accuracy:  ',
                   'Precision: ',
                   'Recall:    ',
                   'F1:        '
                  ]
    
    print(f"\n{model_name}")
    for name, score in zip(score_names, score_list):
        print("*   ", name, round(score, 4))
 

def print_scores2(score_dict, 
                  model_name, 
                  score_types, 
                  train_scores=True, 
                  round_val=4
                 ):
    """
    Prints average scores in easily-legible table format from dictionary 
    provided by sklearn cross_validation.
    ---
    IN
    score_dict: scores as output by sklearn cross_validation (dict)
    model_name: name of scored model (str)
    score_types: list of scores to be calculated (list of strs)
    train_scors: show scores for training set (bool)
    round_val: round to this number of digits (int)
    OUT
    None
    """
    
    print(f"\n{model_name}")
    if train_scores == True:
        groups = ['train', 'test']
        tabs = (round_val+2) // 8 + 1
        print("\t\tTRAIN" + "\t" * tabs + "TEST")
    else:
        groups = ['test']
    
    for score_type in score_types:
        tabs = (11 - len(score_type)) // 8
        s = f"*   {score_type.capitalize()}:" + "\t" * tabs
        scores = []
        for group in groups:
            score = score_dict[group + '_' + score_type].mean()
            scores.append(score)
        for score in scores:
            s += ('\t' + str(round(score, round_val)))
        print(s)


def the_dummy(y):
    """
    Predicts the desired outcome in all cases, returns F1 score calculated i
    against training y.
    IN
    y: targets (df, series, array)
    OUT
    scores: F1, maybe others (floats)
    """
    
    y_hat = np.array([1] * len(y))
    
    acc = metrics.accuracy_score(y, y_hat)
    pre = metrics.precision_score(y, y_hat)
    rec = metrics.recall_score(y, y_hat)
    f1 = metrics.f1_score(y, y_hat)
    
    return acc, pre, rec, f1


def try_some_ks(X, y, max_k=20):
    """
    Uses KNN to predict purchaser type with k values from one up to the given
    max. Returns k values v. four scores.
    ---
    IN
    X: features (df, series, array)
    y: target (df, series, array)
    max_k: largest k value to test
    OUT
    results: accuracy for each k value (dict)
    """
    
    scores_by_k = {}

    for k in range(1,max_k+1):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores_by_k[k] = run_cvs(knn, X, y)

    return scores_by_k


def run_cvs(model, X, y, folds=3, score_types=None):
    """
    Docstring coming soon!
    """
    
    scores = []
    if score_types == None:
        score_types = ['accuracy',
                       'precision',
                       'recall',
                       'f1',
                      ]
    
    for scorer in score_types:
        score = cross_val_score(model, X, y, scoring=scorer, cv=folds).mean()
        scores.append(round(score,4))
        
    return(scores)


def crossval(X, y,
             model_list,
             model_names,
             folds=5,
             score_types=None,
             print_scores=True,
             train_scores=True
            ):
    """
    Performs cross validation on a list of models, returns their scores and 
    prints scores if desired.
    """

    model_scores = {}
    if score_types == None:
        score_types = ['accuracy',
                       'precision',
                       'recall',
                       'f1'
                      ]

    for name, clf in zip(model_names, model_list):
        scores = cross_validate(clf, 
                                X, 
                                y, 
                                scoring=score_types, 
                                cv=folds, 
                                return_train_score=train_scores,
                                n_jobs=-1
                               )
        model_scores[name] = scores

    if print_scores == True:
        for name in model_names:
            print_scores2(model_scores[name],
                          name,
                          score_types,
                          train_scores=train_scores,
                         )

    return model_scores


def tune_params(estimator,
                param_grid,
                which='grid',
                folds=5,
                output='all' 
               ):
    pass


def rf_grid(X,
            y,
            num_estimators=[10],
            max_depths=None,
            rs=23,
            class_weight='balanced',
            ignore_warnings=True,
            verbose=True
            ):
    """
    Home-baked grid search for Random Forest based on OOB score.
    ---
    IN
    X: features
    y: target
    num_estimators: list of number of estimators (trees) to be tried
    max_depths: list of 
    """

    max_depths = make_list_if_not(max_depths)

    # must import warnings
    if ignore_warnings == True:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    if type(max_depths) == int or max_depths == None:
        temp = max_depths
        max_depths = []
        max_depths.append(temp)

    if verbose == True:
        print("Number of Trees:", num_estimators)
        print("Max Depths:", max_depths)
        print("\nRandom Forest Scores")
        print("TREES\tMAX D\tTRAIN\tOOB\tDIFF")
    
    scores = {}
    max_oob = 0
    min_diff = 1
    underfit = False

    ### I think this breaks if max_depths=None -- FIX!!

    for d in max_depths:
        if d == None:
            d_str = 'None'
        else:
            d_str = str(d)
        d_full = 'maxd_' + d_str
        scores[d_full] = {}
        
        for n in num_estimators:    
            n_str = 'trees_' + str(n)
            scores[d_full][n_str] = []

            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                oob_score=True,
                n_jobs=-1,
                random_state=rs,
                class_weight=class_weight
            )
            rf.fit(X,y)
            y_hat = rf.predict(X)
            
            acc_train = metrics.accuracy_score(y_hat, y)
            acc_oob = rf.oob_score_
            diff = acc_train - acc_oob
            max_oob = max(max_oob, acc_oob)
            min_diff = min(min_diff, diff)        

            if diff == min_diff:
                min_settings = [d_full, n_str]
            
            t = '\t'
            s = str(n) + t + d_str
            for score in [acc_train, acc_oob, diff]:
                scores[d_full][n_str].append(score)
                s += (t + str(round(score, 4)))
                      
            if verbose == True:
                print(s) 

    md_r = round(min_diff, 4)
    if scores[min_settings[0]][min_settings[1]][0] < max_oob:
        underfit = True

    print(f"\nMaximum OOB: {round(max_oob)}")
    print(f"Minimum difference between TRAIN and OOB of {md_r} found with:")
    print(f"*   {min_settings[0]} and {min_settings[1]}")
    if underfit == True:
        print("TRAIN SCORE LESS THAN MAX OOB SCORE, MODEL LIKELY UNDERFIT")
                  
    return scores
