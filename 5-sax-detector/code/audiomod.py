# METIS PROJECT 5 - KOJAK
#
# Module containing all pre-PyTorch functions/classes

import numpy as np
import pandas as pd
import librosa
from librosa import display
from pymongo import MongoClient
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from glob import glob
import os
import re
from datetime import datetime
 

# GLOBAL VARIABLES AND OBJECTS

client = MongoClient(
    "mongodb://{}:{}@{}/kojak".format(
        os.environ['mdbUN'],
        os.environ['mdbPW'],
        os.environ['mdbIP']
    )
)
kdb = client.kojak


# AUDIO MANIPULATION

def chunk_song(fpath_in, 
               dir_out='../audio/wav_chunked',
               chunk_len=5, 
               sr=22050, 
               fileid_min='auto',
               log=True):
    """
    Splits song into as many complete chunks of specified length
    as can be created from length of input file.
    ---
    IN
    fpath_in: path of file to be loaded and chunked (str)
    chunk_len: duration of chunk in seconds (int)
    sr: sample rate (int)
    fileid_min: either number of first file id for song (int) or 'auto' to 
        detect last number used in the directory (str)
    log: True to log chunk and song name in MongoDB and printout, else False
    NO OUT
    """
    
    if fileid_min == 'auto':
        try:
            fileid_min = int(os.listdir(dir_out)[-1][:6]) + 1
        except ValueError:
            fileid_min = 0
        except IndexError:
            fileid_min = 0
        except Exception:
            print("Unexpected file conditions encountered.")
            sys.exit(1)
    else:
        fail = "*** ERROR: fileid_min should be of type int if not 'auto'" 
        assert type(fileid_min) is int, fail 
    
    ssr = None
    fileid = fileid_min
    chunk_samples = chunk_len * sr
    
    # loads song of any format
    try:
        y, ssr = librosa.load(fpath_in, sr=sr)
    except:
        print("*** ERROR: could not load file:", fpath_in)
    
    # figures out how many chunks in song and splits into that many + 1
    if ssr:
        try:
            n_chunks = (y.shape[0] / sr) // chunk_len
            split_points = [chunk_samples * n for n in range(1,int(n_chunks)+1)]
            y_split = np.split(y, split_points)
            # print("Chunking", fpath_in)
    
            # saves all chunks of correct length as .wav files
            for chunk in y_split:
                if chunk.shape[0] == chunk_samples:
                    fileid_str = str(fileid).rjust(6,'0') 
                    fpath_out = os.path.join(dir_out, (fileid_str + '.wav'))
                    librosa.output.write_wav(fpath_out, chunk, sr)
                    if log:
                        song_name = song_name_extractor(fpath_in)
                        log_chunk(fileid_str, song_name)
                    fileid += 1
        except:
            print("*** ERROR: could not chunk file:", fpath_in)


def song_name_extractor(file_link):
    """
    Pulls out song name from file path, strips disc/track numbers and file 
    extension.
    ---
    IN
    file_link: os path to audio file (str)
    OUT
    sname: song name (str)
    """

    # first pattern takes everything between last / and .ext
    p1 = re.compile(r"/([^/]+)\.\w{3}")
    # next takes everything after track/disc number and whitespace
    p2 = re.compile(r"[\d-]*\s(.+)")

    # testing both cases
    step1 = p1.search(file_link)
    if step1:
        sname = step1.group(1)
    else:
        sname = file_link

    step2 = p2.match(sname)
    if step2:
        sname = step2.group(1)

    return sname


def log_chunk(fileid_str, song_name, verbose=True):
    """
    Logs chunk in a in MongoDB collection on EC2.
    ---
    IN
    fileid: six-digit fileid as a string (str)
    fpath_in: absolute or relative os filepath (str)
    verbose: prints chunk id, song name as comma-separated pair if True (bool)
    NO OUT
    """

    if verbose:
        print("{}, {}".format(fileid_str, song_name))
    
    kdb.test_songs.insert_one({"chunk_id": fileid_str,
                               "song_name": song_name,
                              })


def chunk_queue(dir_in="../audio/chunk_queue",
                dir_out="../audio/wav_chunked",
                chunk_len=5,
                sr=22050,
                log=True
               ):
    """
    Feeds each song in queue directory to the chunk_song() function.
    ---
    IN
    dir_in: path of audio queue directory, absolute or relative (str)
    dir_out: path of output directory, absolute or relative (str)
    chunk_len: duration of chunk in seconds (int)
    sr: sample rate (int)
    fileid_min: either number of first file id for song (int) or 'auto' to 
        detect last number used in the directory (str)
    log: True to log chunk and song name in MongoDB and printout, else False
    NO OUT
    """
    
    for root, dirs, files in os.walk(dir_in):
        for fname in files:
            if not re.match(r'^\.', fname):
                rel_fpath = os.path.join(root, fname)
                chunk_song(rel_fpath, chunk_len=chunk_len, sr=sr, log=log)


def wav_to_mp3_batch(dir_in,
                     dir_out="../audio/mp3_chunked",
                     bitrate=96
                    ):
    """
    Converts all .wav files in a directory to .mp3 with bitrate specified.
    Checks destination directory to see if file has been converted already.
    ---
    IN
    dir_in: directory path in which .wav files reside (str)
    dir_out: directory path in which .mp3 files will be saved (str)
    bitrate: bitrate (int or str)
    NO OUT
    """

    existing = set()
    bitrate = str(bitrate)
    
    for mp3_fpath in glob(dir_out + "/*.mp3"):
        f_id = os.path.splitext(os.path.basename(mp3_fpath))[0]
        existing.add(f_id)
        
    for wav_fpath in glob(dir_in + "/*.wav"):
        f_id = os.path.splitext(os.path.basename(wav_fpath))[0]
        if f_id not in existing:
            command = "lame -b{} {}/{}.wav {}/{}.mp3".format(bitrate, 
                                                             dir_in, 
                                                             f_id, 
                                                             dir_out, 
                                                             f_id)
            result = os.system(command)    
            if result != 0:
                print("*** ERROR: {} not converted".format(fb_id))


def db_status():
    """
    Prints labeled status of samples in Mongo DB, adds a status record to a
    separate status DB.
    """
    
    db = kdb.test_songs
    
    # pull last record from status DB for comparison
    last = kdb.status.find_one({"last": True})
    
    labels = [
        ("Total samples\t", 'total'),
        ("Labeled samples\t", 'labeled'),
        ("Skipped samples\t", 'skipped'),
        ("Vocals, foreground", 'vox_fg'),
        ("Vocals, background", 'vox_bg'),
        ("Saxophone, foreground", 'sax_fg'),
        ("Saxophone, background", 'sax_bg'),
        ("Piano, foreground", 'pno_fg'),
        ("Piano, background", 'pno_bg')
    ]

    # creating dict of db figures
    figs = {}
    
    figs['total'] = db.count()
    figs['labeled'] = db.find({"labeled": True}).count()
    figs['skipped'] = db.find({"skipped": True}).count()
    figs['vox_fg'] = db.find({"vocals": 2}).count()
    figs['vox_bg'] = db.find({"vocals": 1}).count()
    figs['sax_fg'] = db.find({"sax": 2}).count()
    figs['sax_bg'] = db.find({"sax": 1}).count()
    figs['pno_fg'] = db.find({"piano": 2}).count()
    figs['pno_bg'] = db.find({"piano": 1}).count()
    
    percent = {}
    for k, v in figs.items():
        percent[k] = round(100 * v/figs['labeled'], 1)
    percent['total'] = 'N/A'

    print("\nSAMPLE DATABASE STATUS")
    print("Category\t\tCount\tDelta\t% Lab'd")
    print("-" * 48)
    for pair in labels:
        current_val = figs[pair[1]] 
        delta = current_val - last[pair[1]]
        print("{}\t{}\t{}\t{}"
            .format(pair[0],
                    str(current_val).rjust(5),
                    str(delta).rjust(5),
                    str(percent[pair[1]]).rjust(5))
             )
    print("-" * 48, '\n')

    # change 'last' field of previous status entry
    update_result = kdb.status.update_one({"last": True},
                                          {"$set": {"last": False}}
                                         )
    if update_result.modified_count != 1:
        print("\n*** Error altering previous status record in DB")
    
    # add 'timestamp', 'last', and 'auto' fields to current record
    figs['timestamp'] = datetime.now()
    figs['last'] = True
    figs['auto'] = False
    # and add to DB
    add_result = kdb.status.insert_one(figs)
    if not add_result:
        print("\n*** Error adding current status record to DB")


### DATAGROUP MANAGEMENT

def create_datagroup_in_db(group_name, pos_label, n_per_label='auto'):
    """
    Creates a datagroup in db by randomly sampling an equal number from records
    where the positive label has a value of 2 (foreground) and records where the
    positive label has a value of 2 (none).
    ---
    IN
    group_name: name of datagroup (str)
    pos_label: field name to use as positive label (str)
    n_per_label: number of samples per label, will use the number of positive 
        samples if 'auto' (str, int)
    NO OUT
    """

    assert_msg = "Invalid input for n_per_label"
    assert n_per_label == 'auto' or type(n_per_label) == int, assert_msg

    pos_ids = np.array([])
    neg_ids = np.array([])

    if n_per_label == 'auto':
        n_per_label = kdb.test_songs.find({pos_label: 2}).count()

    label = 1
    for val, arr in zip([2,0], [pos_ids, neg_ids]):
        chunks = kdb.test_songs.aggregate([
            {"$match": {pos_label: val}},
            {"$sample": {"size": n_per_label}}
        ]) 
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            result = kdb.test_songs.update_one(
                {"chunk_id": chunk_id},
                {"$set": {group_name: label}}
            )
            if result.modified_count != 1:
                print("*** Error on DB insertion, {}".format(chunk_id))
                break
        label -= 1

    for label in [1,0]:
        members = kdb.test_songs.find(
            {group_name: label}    
        ).count()
        print("Label {}: {}".format(label, members))


def pull_datagroup_from_db(group_name, df=True):
    """
    Pulls datagroup from Mongo DB, returns a list of chunk_id, label tuples.
    ---
    IN
    group_name: group name as named in create_datagroup_in_db() function (str)
    df: if True, return a pandas df (bool)
    OUT
    datagroup: list of tuples (list)
    """

    datagroup = []
    
    for item in kdb.test_songs.find({group_name: {"$exists": True}}):
        datagroup.append((item['chunk_id'], item[group_name]))

    if df:
        dg_trans = list(zip(*datagroup))
        datagroup = pd.DataFrame({
            'chunk_id': dg_trans[0],
            'actual': dg_trans[1]
        })

    return datagroup.filter(['chunk_id', 'actual'])
    

def tts(df, train_size=0.8):
    """ 
    Create train and test dataframes with provided datagroup dataframe.
    ---
    IN
    df: datagroup df as created by pull_datagroup_from_db() function (df)
    train_size: size of training set, test set will be 1-train_size (float)
    OUT
    train_df, test_df
    """
     
    assert_msg = "train_size must be between 0 and 1"
    assert train_size < 1 and train_size > 0, assert_msg

    records = df.shape[0]
    # 0 for train, 1 for test
    tt_assigns = np.random.choice(2, records, p=[train_size, 1-train_size])

    assert_msg = "tt_assigns array of incorrect length"
    assert len(tt_assigns) == df.shape[0], assert_msg

    df['tt'] = tt_assigns
    tt_dfs = []
    for i in range(2):
        tt_dfs.append(df[df.tt == i]
            .filter(['chunk_id', 'actual'])
            .reset_index(drop=True)
        )
    
    # train first, test second
    return tt_dfs[0], tt_dfs[1]


def assign_cv_groups(df, folds=4):
    """
    Append a column of cross validation groups to datagroup df.
    ---
    IN
    df: datagroup df as created by pull_datagroup_from_db() function (df)
    folds: number of CV folds (int)
    OUT
    df: datagroup with CV groups column (df)
    """

    probs = [1/folds for i in range(folds)]
    records = df.shape[0]
    cv_groups = np.random.choice(folds, records, p=probs)
    df['cv'] = cv_groups

    return df


def tts_full(
        round_name,
        train_size=0.8,
        n_labels=2,
        n_per_label='auto',
        pos_label='sax'
       ):
    """
    *** OBSOLETE, USE create_datagroup_in_db(), pull_datagroup_from_db(), and
    tts() instead ***

    Creates dataset labels in MongoDB under provided round name.
    First pass only deals with two labels; future versions will accommodate more
    as necessary.
    ---
    IN
    round_name: name of the training/test round. e.g. 'round_1' (str)
    n_labeles: number of labeles (int)
    train_size: ratio of train set to whole (float, 0-1)
    n_per_label: number of samples per label; if 'auto', will make this the
        number of the minimum available labels (int)
    pos_label: MongoDB field name of positive label (str) (this will be a list 
        in future versions)
    ### could have a 'labels' input that expected a list of the search terms for        each label, e.g. [{"sax": 2}, {"sax": 0}], run with eval in DB query.
    NO OUT
    """

    pos_ids = np.array([])
    neg_ids = np.array([])

    if n_per_label == 'auto':
        n_per_label = kdb.test_songs.find({pos_label: 2}).count()

    # find cutoff index value that would split arrays into appropriately-sized
    # groups for train/test labeling, if this would be faster than generating
    # a train/test selection on each insertion
    # cutoff_ix = int(train_size * n_per_label)

    # pull IDs, label each record accordingly
    label = n_labels - 1
    for val, arr in zip([2,0], [pos_ids, neg_ids]):
        chunks = kdb.test_songs.aggregate([
            {"$match": {pos_label: val}},
            {"$sample": {"size": n_per_label}}
        ]) 
        for chunk in chunks:
            arr = np.append(arr, chunk["chunk_id"])
        # shuffle array if needed
        # np.random.shuffle(arr)
        # insert field with label number and train/test indicator
        for chunk_id in arr:
            tt = np.random.choice(
                ['train','test'], 
                p=[train_size, 1-train_size]
            )
            result = kdb.test_songs.update_one(
                {"chunk_id": chunk_id},
                {"$set": {round_name: (label, tt)}}
            )
            if result.modified_count != 1:
                print("*** error on db insertion, {}".format(chunk_id))
                break
        label -= 1
    
    # print validation statments
    for label in [1,0]:
        for group in ['train','test']:
            members = kdb.test_songs.find(
                {round_name: (label, group)}    
            ).count()
            print("Label {}, {}: {}".format(label, group, members))
    

### SPECTROGRAM GENERATION

def audio_loader(
        chunk_id,
        dir_in="../audio/wav_chunked",
        sample_rate=22050,
        duration=5.0
    ):
    """
    Loads audio file as 2D mono matrix, outputs file and sample rate.
    ---
    I/O coming soon
    """

    wav_fpath = os.path.join(dir_in, chunk_id + '.wav')
    y, sr = librosa.load(
        wav_fpath,
        sr=sample_rate,
        duration=5.0
    )

    return y, sr


def make_spectro(
        audio_ndarray,
        sample_rate,
        hl=256,
        n_fft=1024,
        n_mels=512,
        normalize=False,
        db_scale=True
    ):        
    """
    Makes spectrogram for sample with provided chunk ID and returns.
    ---
    IN
    chunk_id: six-digit ID, e.g. 000123 (str)
    dir_in: path to input directory containing audio files (str)
    sample_rate: number of samples per second (int)
    hl: hop length, number of samples between FFT calculations (int)
    n_fft: number of samples per FFT calculation, usually 4x hl (int)
    n_mels: number of mel bands, frequency groups for FFT calculations (int)
    normalize: normalize the sample before performing FFT if True (bool)
    db_scale: will convert from power to DB if True (bool)
    OUT
    ms: mel-band spectrogram (np array)
    """

    
    if normalize:
        ### INSERT NORMALIZATION CODE HERE
        print("Normalization option coming soon.")

    # make spectrogram array on mel scale
    ms = librosa.feature.melspectrogram(
        y=audio_ndarray,
        sr=sample_rate,
        hop_length=hl,
        n_mels=n_mels
    )

    if db_scale:
        # setting ref=np.max automatically normalizes
        # this is where the if normalize function could come in
        ms = librosa.power_to_db(ms, ref=np.max)

    return ms


def batch_spectros(
        dir_in="../audio/wav_chunked",
        dir_out="../specs/mel",
        files='labeled',
        sample_rate=22050,
        hl=256,
        n_fft=1024,
        n_mels=512,
        normalize=False
    ):
    """
    Make spectrograms out of all audio files in given directory for which 
    spectrograms do not exist in out directory.
    ---
    IN
    dir_in: path to input directory containing audio files (str)
    dir_out: path to directory in which to save spectrogram files (str)  
    files: 'all' or 'labeled'; labeled will only make spectrograms for samples
        flagged as labeled in MongoDB (str)
    sample_rate: number of samples per second (int)
    hl: hop length, number of samples between FFT calculations (int)
    n_fft: number of samples per FFT calculation, usually 4x hl (int)
    n_mels: number of mel bands, frequency groups for FFT calculations (int)
    normalize: normalize the sample before performing FFT if True (bool)
    NO OUT
    """

    assert_msg = "Error: files arg must be either 'all' or 'labeled'"
    assert files == 'all' or files == 'labeled', assert_msg

    existing = set()
    
    for spec_fpath in glob(dir_out + "/*.npy"):
        chunk_id = os.path.splitext(os.path.basename(spec_fpath))[0]
        existing.add(chunk_id)

    chunk_queue = set()
    
    if files == 'all':
        for wav_fpath in glob(dir_in + "/*.wav"):
            chunk_id = os.path.splitext(os.path.basename(wav_fpath))[0]
            chunk_queue.add(chunk_id)
    if files == 'labeled':
        labeled_ids = kdb.test_songs.find(
            {"labeled": True}
        )
        for doc in labeled_ids:
            chunk_queue.add(doc['chunk_id'])
    else:
        pass
        # expand here to accept a custom search term for MongoDB

    # remove chunk IDs with existing spectros from the queue
    chunk_queue -= existing

    try:
        new_specs = 0
        for chunk_id in chunk_queue:
            y, _ = audio_loader(
                chunk_id,
                dir_in=dir_in,
                sample_rate=sample_rate,
                duration=5.0
            )
            spectro = make_spectro(
                y,
                sample_rate=sample_rate,
                hl=hl,
                n_fft=n_fft,
                n_mels=n_mels,
                normalize=normalize
            )
            spec_path_out = os.path.join(dir_out, chunk_id)
            np.save(spec_path_out, spectro)
            new_specs += 1
        print("{} spectrograms created".format(new_specs))
    except:
        print("Something bad has happened!") 


def arr_stats(ndarray):
    """
    Prints basic stats for any np array.
    ---
    IN
    ndarray: any np array
    NO OUT
    """

    print("Min:", np.min(ndarray))
    print("Max:", np.max(ndarray))
    print("Mean:", np.mean(ndarray))
    print("Std:", np.std(ndarray))
    print("Shape:", np.shape(ndarray))


def spectro_viz(
    spectro,
    sample_rate=22050,
    hl=256,
    show=True,
    cmap='magma',
    margin=True,
    save=False,
    dir_out="../specs",
    chunk_id=None,
    fig_dims=(8,8)
    ):
    """
    Make image of dB-scaled spectrogram given chunk ID and path to WAV file.
    ### DO DOCSTRING IN/OUt
    """

    # creates figure of same aspect ratio as original
    if fig_dims:
        fig = plt.figure(figsize=fig_dims, dpi=128)
    else:
        w, h = figaspect(spectro)
        fig = plt.figure(figsize=(w,h), dpi=128)
    
    ax = plt.subplot(111)
    
    if margin == False:
    # these next two create a subplot with no margins
    	# ax = plt.subplot(111)
    	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
    	                   )

    # creates visuals for display or saving
    librosa.display.specshow(
        spectro,
        # librosa.power_to_db(spec_array, ref=np.max),
        sr=sample_rate,
        hop_length=hl,
        y_axis='mel', # mel, log, fft
        x_axis='time', # time
        cmap=cmap
    )

    # change font and tick size/frequency
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.xticks(range(6))
    plt.xlabel("Time (sec)", fontsize=16)
    plt.ylabel("Frequency (Hz)", fontsize=16)

    # if save is chosen, it will not show in Jupyter Notebook
    if save:
        img_fpath = os.path.join(dir_out, chunk_id + ".png")
        plt.savefig(img_fpath, dpi=fig.dpi)
    
    plt.show();


### OLD CODE FOR FIRST ROUND OF FILE CONVENTIONS

def make_spectro_old(
        fname, 
        sample_rate=22050, 
        n_fft=1024,
        hl=256, 
        n_mels=512,
        cmap='magma',
        show=True, 
        save=False
    ):
    """
    The beginnings of a grand function for making and storing a spectrogram 
    for each file using librosa.
    """
    
    # update this with os.path.join()
    fpath = "../audio/" + fname + ".wav"
    y, sr = librosa.load(fpath,
                         sr=sample_rate,
                         duration=5.0,
                        )
    
    # make the spectrogram matrix on mel scale
    M = librosa.feature.melspectrogram(y=y,
                                       sr=sample_rate,
                                       hop_length=hl, 
                                       n_mels=n_mels
                                      )
    
    # creates figure of same aspect ratio as original
    w, h = figaspect(M)
    fig = plt.figure(figsize=(w,h), dpi=108)
    
    # these next two create a subplot with no margins
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, 
                        wspace=0, hspace=0
                       )
    
    # creates visuals for display or saving
    if show or save:
        librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                                 sr=sample_rate,
                                 hop_length=hl,
                                 y_axis='mel', # mel, log, fft
                                 x_axis='time', # time
                                 cmap=cmap
                                )

    if show:
        plt.show()
  
    if save:
        img_fpath = "../specs/" + fname + ".png"
        plt.savefig(img_fpath, dpi=fig.dpi)
        plt.close(fig)
    
    return M


def rename_files(regex_pattern, 
                 rename_code,
                 directory='dump'
                ):
    """
    For batch renaming of files that do not fit the estabilshed convention.
    ---
    IN
    regex_pattern: pattern used to match files needing edits (str)
    rename_code: code used to rename the file, assuming static edit (str)
    directory: sub-directory of '/audio' to scan through (str)
    NO OUT
    """
    
    root = ('/Users/dluther/ds/metis/metisgh/projects/05-kojak/audio/' 
            + directory)
    p = re.compile(regex_pattern)
    edits = 0

    for fname in os.listdir('../audio/' + directory):
        if p.match(fname):
            print("Renaming", fname)
            os.rename(root + '/' + fname, root + '/' + eval(rename_code))
            edits += 1
    
    print("{} files renamed".format(edits))


def dispatch_files(dispatch_dir='dump'):
    """
    Moves audio files from 'dump' directory to proper audio sub-folders.
    ---
    IN
    dispatch_dir: directory from which to start if not 'dump'
    NO OUT
    """
    
    root = '/Users/dluther/ds/metis/metisgh/projects/05-kojak/audio/'
    
    for file in os.listdir('../audio/' + dispatch_dir):
        fpath_current = root + dispatch_dir + '/' + file
        if not re.match(r'^\w{2}\d{4}.wav', file):
            print("No match:", file)
            continue
        if re.match('^ns', file):
            target = 'no_sax/'
        if re.match('^sc', file):
            target = 'sax_sec/'
        if re.match('^ss', file):
            target = 'sax_solo/'
        os.rename(fpath_current, root + target + file)


def make_spectrograms_old(spectros=None, 
                          overwrite=False, 
                          cmap='magma',
                          subdirs=['no_sax', 'sax_sec', 'sax_solo']
                         ):
    """
    *** VERSION 1, USES OLD FILE FORMAT ***
    Makes spectrograms for all audio files, or for those for which spectrograms
    have not yet been made.
    ---
    IN
    overwrite: if True, will overwrite any pre-existing spectrograms (bool)
    cmap: matplotlib colormap, usually 'magma' or 'gray' (str)
    spectros: dictionary of pre-existing spectrograms, or None (dict)
    OUT
    spectros: dictionary of spectrograms with root filename as key
    """

    file_ids = set()
    if not spectros:
        spectros = {}

    if not overwrite:
        for root, dirs, fnames in os.walk('../specs'):
            for fname in fnames:
                if re.match(r'\w{2}\d{4}', fname):
                    file_ids.add(fname[:6])

    print(file_ids)

    for subdir in subdirs:
        for fname in os.listdir('../audio/' + subdir):
            f_id = fname[:6]
            if f_id not in file_ids and re.match(r'\w{2}\d{4}', f_id):
                print(f_id)
                fp = subdir + '/' + f_id
                spectros[f_id] = make_spectro(fp, 
                                              show=False, 
                                              save=True,
                                              cmap=cmap
                                             )

    return spectros
