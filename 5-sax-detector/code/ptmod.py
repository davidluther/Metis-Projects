# METIS PROJECT 5 - KOJAK
#
# module of all pytorch-related code for Kojak

import komod
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
import torchvision
# from torchvision import transforms, utils
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, OrderedDict
from copy import deepcopy
import pickle
import time
import os

import matplotlib.pyplot as plt


# GLOBAL VARIABLES AND OBJECTS

client = MongoClient(
    "mongodb://{}:{}@{}/kojak".format(
        str(os.environ['mdbUN']),
        str(os.environ['mdbPW']),
        str(os.environ['mdbIP'])
    )
)
kdb = client.kojak

rs = torch.manual_seed(42)

# DATA LOADING

class SpectroDataset(data_utils.Dataset):
    """
    Spectrogram dataset class
    ---
    INIT PARAMS
    dataset_name: name of dataset as labeled in MongoDB (str)
    group: 'train' or 'test' or 'val', must be present in records of 
        dataset_name (str)
    scaling: factor by which to scale the spectrogram dimensions, e.g. a 
        factor of 0.5 would make a spectrogram 1/4 the size (int, float)
    dir_in: path to directory in which .wav files reside (str)
    trasform: transform function or functions if not None
    """
    
    def __init__(
            self, 
            datagroup_df,
            scaling=1, 
            dir_in="../audio/wav_chunked",
            transform=None
        ):
        self.sample_frame = datagroup_df
        self.scaling = scaling
        # this is a dictionary of audio parameters necessary for scaling
        # spectrograms during creation
        self.audio_params = {
            'hl': 256,
            'n_fft': 1024,
            'n_mels': 512
        }
        self.dir_in = dir_in
        self.transform = transform

    def __len__(self):
        return self.sample_frame.shape[0]
    
    def __getitem__(self, ix):
        """Makes and returns spectrogram of requested item"""
        chunk_id = self.sample_frame.loc[ix, 'chunk_id']
        y, sr = komod.audio_loader(chunk_id)
        sample = komod.make_spectro(
            y, 
            sr,
            hl = int(self.audio_params['hl'] / self.scaling),
            n_fft = int(self.audio_params['n_fft'] / self.scaling),
            n_mels = int(self.audio_params['n_mels'] * self.scaling)
        )
        # add singleton dimension
        sample = np.expand_dims(sample, 0)
        # normalize on -1-1 scale as done in PyTorch tutorial
        sample = normalize_spec(sample, low=-1)
        # convert to torch float tensor
        sample = torch.from_numpy(sample).float()

        if self.transform:
            sample = self.transform(sample)        

        return sample, self.sample_frame.loc[ix, 'actual'], chunk_id


# CNN ARCHITECTURES

class CNN_cpcpff(nn.Module):
    """
    params: Pass input params as a dictionary where each item is a layer and 
    each value is a list, following this convention:
    
    Convolutional: c1: [kernel, stride, channels_out]
    Max Pooling: p1: [kernel, stride]
    Fully Connected: f1: [channels_in, channels_out]
    
    For example:
    
        params = {
            'c1': [5,1,10],
            'p1': [2,2],
            'c2': [5,1,20],
            'p2': [2,2],
            'f1': [2600,50],
            'f2': [50,2]
        }
    
    All values must be integers.
    
    rs: random seed (int)

    normal: if True, update parameters with normal distribution based with
        mean = 0 and std = 1 / sqrt(input_dims * kernel_w * kernel_h) (bool)
    """
    
    def __init__(self, params, rs=23, normal=True):
        super(CNN_cpcpff, self).__init__()
        # (in channels, out channels, kernel, stride=s)
        self.p = params
        self.rs = rs
        self.seed_gen = torch.manual_seed(self.rs)
        self.conv1 = nn.Conv2d(1, 
                               self.p['c1'][2], 
                               self.p['c1'][0], 
                               stride=self.p['c1'][1])
        # (2x2 kernel, stride=2 -- stride defaults to kernel)
        self.pool1 = nn.MaxPool2d(self.p['p1'][0], self.p['p1'][1])
        self.conv2 = nn.Conv2d(self.p['c1'][2], 
                               self.p['c2'][2], 
                               self.p['c2'][0], 
                               stride=self.p['c2'][1])
        self.pool2 = nn.MaxPool2d(self.p['p2'][0], self.p['p2'][1])
        self.fc1 = nn.Linear(self.p['f1'][0], self.p['f1'][1])
        self.fc2 = nn.Linear(self.p['f2'][0], self.p['f2'][1])
        if normal:
            self.apply(init_norm_auto)
        self.seed_gen = None
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # need to reshape for fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def save_myself(self, fname, dir_out='../data'):
        """
        Saves current object as a .pkl file.
        ---
        fname: filename of choice (str)
        dir_out: path to save directory (str)
        """
        
        # add timestamp here
        fpath = os.path.join(dir_out, fname + '.p')
        with open(fpath, 'wb') as pf:
            pickle.dump(self, pf)


# CNN DESIGN HELPERS

def reduce_axis(pix_in, kernel_size, stride, drop_last=False):
    """
    Calculate output pixels along one axis given input pixels,
    filter size, and stride.
    ---
    IN
    pix_in: number of pixels along input axis (int)
    kernel_size: assuming a square filter, pixels on one size (int)
    stride: pixels per step (int)
    drop_last: if True, ignore pixels on last step if fewer than
        filter_dim (bool)
    OUT
    pix_out: number of pixels along output axis
    """
    
    pix_out = (pix_in - kernel_size) // stride + 1
    if not drop_last:
        if (pix_in - kernel_size) % stride > 0:
            pix_out += 1
    
    return pix_out


def cnn_pixels_out(dim_in, layers, drop_last=False):
    """
    Computes CNN output pixels given input dimensions and layer info.
    Assumes a square kernel and drop_last=False for reduce_axis process.
    ---
    IN
    dim_in: (C, W, H) format, where each is an integer (tup or list)
    layer_info: ((kernel, stride, filters_out), ...) format, where each is an 
        int. If a max pooling layer, set filters to 0 (tup or list)
    OUT
    pixels_out: number of pixels going into FC layer (int)
    """

    c = dim_in[0]
    w = dim_in[1]
    h = dim_in[2]
    print("{} x {} x {}".format(c,w,h))
    
    for layer in layers:
        if layer[2] != 0:
            c = layer[2]
        w = reduce_axis(w, layer[0], layer[1], drop_last=drop_last)
        h = reduce_axis(h, layer[0], layer[1], drop_last=drop_last)
        print("{} x {} x {}".format(c,w,h))
            
    return c * w * h


# MODEL UTILITY FUNCTIONS

def normalize_spec(ndarray, low=0, high=1, min_db=-80):
    """
    Normalize dB-scale spectrogram from low-high given min dB at creation.
    """

    factor = min_db / (high-low)

    # might just be able to do ndarray /= -min_db
    # would invert the image though
    ndarray -= factor
    ndarray /= abs(factor)
    
    return ndarray


def init_norm_auto(m):
    """Based on self.reset_parameters() in nn.Linear and nn.Conv2n"""
    seed_gen = torch.manual_seed(23)
    # print(m)
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
        if isinstance(m, nn.Linear):
            n = m.weight.size(1)
        std = 1. / np.sqrt(n)
        m.weight.data.normal_(mean=0, std=std)
        if m.bias is not None:
            m.bias.data.normal_(mean=0, std=std)
        # print(m.weight)


def fit(cnn, 
        dataset, 
        optimizer, 
        criterion, 
        num_epochs, 
        batch_size, 
        minibatches=None
    ):
    """
    Runs feed-forward and back-prop to train CNN model.
    *** ROLL INTO CNN CLASS?
    ---
    IN
    cnn: CNN instance 
    dataset: built SpectroDataset object
    optimizer: PyTorch optimizer for back-prop
    criterion: PyTorch loss object for loss metric
    num_epochs: number of times to cycle through data (int)
    batch_size: number of records per batch (int)
    minibatches: print loss and time every n minibatches (COMING SOON) (int)
    OUT
    loss_by_epoch: average loss per epoch (ndarray)
    """
    
    train_loader = data_utils.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        drop_last=False # not sure the merits of doing this or not?
    )

    loss_by_epoch = []

    for epoch in range(num_epochs):
        print("Epoch", epoch+1)
        running_loss = 0.0
        loss_per_batch = np.array([])
        before = deepcopy(cnn.state_dict()['conv2.weight'])
        then = time.perf_counter()
        for i, data in enumerate(train_loader, 1):
            sub_then = time.perf_counter()
            # separate input data and labels, dump chunk IDs
            spectros, labels, _ = data
            # wrap in Variable for GD
            spectros, labels = Variable(spectros), Variable(labels)
            # zero parameter gradients, else accumulate
            optimizer.zero_grad()
            # forward prop
            outputs = cnn(spectros)
            # calculate loss
            loss = criterion(outputs, labels)
            # backprop
            loss.backward()
            # update weights
            optimizer.step()         
            #verbosity
            sub_now = time.perf_counter()
            print("\r * {} loss: {:.3f}\tTime: {:.3f} ms"
                  .format(i, loss.data[0], (sub_now-sub_then)*1000), end='')
            loss_per_batch = np.append(loss_per_batch, loss.data[0])
            running_loss += loss.data[0]
#             # print every n minibatches
#             running_loss += loss.data[0]
#             if i%minibatches == minibatches:
#                 print('[%d, %5d] loss: %.3f' % (
#                     epoch+1, i, running_loss/minibatches))
#                 running_loss = 0.0
        now = time.perf_counter()
        after = cnn.state_dict()['conv2.weight']
        update = not np.allclose(before.numpy(), after.numpy())
        avg_loss = running_loss/i
        loss_by_epoch.append(loss_per_batch)
        print("\r * Avg loss: {:.3f}\tTime: {:.3f} ms"
              .format(running_loss/i, (now-then)*1000))
        print(" * Weights updated:", update)
    print('\n\aTraining Complete')

    return np.vstack(loss_by_epoch)


def predict(cnn, dataset, batch_size=4, res_format='df'):
    """
    Predicts values on trained CNN.
    *** ROLL INTO CNN CLASS?
    ---
    IN
    cnn: trained CNN instance
    dataset: built SpectroDataset object
    batch_size: number of records per batch
    res_format: results format, either 'df' for pandas dataframe or 'dict'
        for dictionary (str)
    OUT
    results: if 'dict', dictionary with chunk ID as key, and a tuple of (actual,
        predicted, output_array) as value (dict); if 'df', pandas dataframe
    """
    
    test_loader = data_utils.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, # set for False for test set
        num_workers=2
    )
    
    results = {}
    
    # could clean this up to load it straight into df instead of dict
    for data in test_loader:
        spectros, labels, chunk_ids = data
        outputs = cnn(Variable(spectros))
        _, pred = torch.max(outputs.data, 1)
        for c_id, y, y_hat, out in zip(chunk_ids, labels, pred, outputs.data):
            results[c_id] = (y, y_hat, out)
            
    if res_format == 'df':
        results = results_to_df(results)
    
    return results


def results_to_df(results):
    """
    Converts predict results to Pandas dataframe.
    ---
    IN
    results: dictionary generated by results function (dict)
    OUT
    df: pandas dataframe of results 
    """

    cols = ['chunk_id', 'actual', 'pred', 'e0', 'e1']
    results_trans = OrderedDict.fromkeys(cols)
    for k in results_trans.keys():
        results_trans[k] = []

    for k, v in results.items():
        for col, val in zip(cols, [k, v[0], v[1], v[2][0], v[2][1]]):
            results_trans[col].append(val)
    
    df = pd.DataFrame(results_trans)
    
    return df


# SCORING, CV, GRID SEARCH

def get_scores(train_df, test_df, verbose=True):
    """
    Calculates accuracy, recall, and specificity for train and test
    predictions.
    ### add precision?
    ---
    IN
    train_df: predict results df of train set
    test_df: predict results df of test set
    OUT
    scores_dict: scores bundle (dict)
    """
    
    scores_dict = defaultdict(list)
    score_types = [
        'acc', 
        # 'pre', 
        'rec', 
        'spec'
    ]
    
    for df in [train_df, test_df]:
        df_scores = []
        df_scores.append(
            metrics.accuracy_score(df.actual, df.pred))
        # df_scores.append(
        #     metrics.precision_score(df.actual, df.pred))
        df_scores.append(
            metrics.recall_score(df.actual, df.pred))
        df_scores.append(
            metrics.recall_score(df.actual, df.pred, pos_label=0))
        for n, s in zip(score_types, df_scores):
            scores_dict[n].append(s)
        
    return scores_dict


def print_scores(scores_dict, title=None):
    """
    Print scores in table given scores dictionary as created by get_scores().
    ---
    IN
    scores_dict: dictionary of classification scores (dict)
    title: title, if given (str)
    NO OUT
    """
    
    if title:
        print(title)
    print("Score\tTrain\tTest")
    print("-" * 24)
    for score in scores_dict.keys():
        print("{}\t{:.3f}\t{:.3f}".format(
            score.capitalize(), 
            scores_dict[score][0],
            scores_dict[score][1])
        )


def crossval(
        cnn_params, 
        datagroup_df, 
        scaling,
        optim_partial, 
        criterion, 
        num_epochs, 
        batch_size,
        folds=4,
        rs=23
    ):
    """
    Performs cross validation on dataset with model, optimizer, criterion, and 
    hyperparameters as specified.
    ---
    IN
    cnn: untrained CNN model object
    datagroup_df: dataframe of datagroup (df)
    scaling: degree of scaling to apply to spectros, 0-1 (float)
    optim_partial: partial object of optimizer, with all parameters preset
    criterion: loss/error criterion object on which to optimize
    num_epochs: number of epochs to run per fold (int)
    batch_size: number of spectros per batch (int)
    folds: number of folds for cross val (int)
    rs: random seed for torch random generator (int)
    OUT
    scores_cv: average scores per fold, in- and out-of-sample (dict)
    scores_bundle: in- and out-of-sample scores for each fold (dict)
    losses_per_fold: list of avg loss per epoch for each fold (list)
    """ 

    # add folds column to dataset df
    df = komod.assign_cv_groups(datagroup_df, folds=folds)

    scores_bundle = {}
    losses_per_fold = []

    for i in range(folds):
        print("\n*** Fold {} ***".format(i+1))
        train_df = (df[df.cv != i]
                    .filter(['chunk_id', 'actual'])
                    .reset_index(drop=True))
        test_df = (df[df.cv == i]
                    .filter(['chunk_id', 'actual'])
                    .reset_index(drop=True))
        # sanity check of tt lengths
        print("\nTrain set length: {}".format(train_df.shape[0]))
        print("Test set length: {}".format(test_df.shape[0]))
        # create dataset objects for train and test
        train_dataset = SpectroDataset(train_df, scaling=scaling)
        test_dataset = SpectroDataset(test_df, scaling=scaling)
        # spawn model with specified parameters
        cnn = CNN_cpcpff(cnn_params, rs=rs)
        print("Random seed: {}\n".format(cnn.rs))
        # train model
        loss_by_epoch = fit(
            cnn, 
            train_dataset, 
            optim_partial(cnn.parameters()), 
            criterion, 
            num_epochs, 
            batch_size
        )
        losses_per_fold.append(loss_by_epoch)
        # get in- and out-of-sample predictions
        train_res = predict(cnn, train_dataset)
        test_res = predict(cnn, test_dataset)
        # calculate scores 
        scores_fold = get_scores(train_res, test_res)
        scores_bundle[i] = scores_fold
        print("\n", end="")
        print_scores(scores_fold)
        
    scores_cv = defaultdict(list)

    # cycle through fold, score, and train/test to average all scores
    for score_type in scores_fold.keys():
        for ix in range(2):
            score_sum = 0
            for fold in range(folds):
                score_sum += scores_bundle[fold][score_type][ix]
            scores_cv[score_type].append(score_sum / folds)
    
    print("\n", end="")
    print_scores(scores_cv, "CV Average Scores")

    return scores_cv, scores_bundle, losses_per_fold
    

# OTHER UTILITIES 

def tensor_stats(tensor):
    """
    Prints basic stats for any torch tensor.
    ---
    IN
    tensor: any torch tensor
    NO OUT
    """

    print("Min:", tensor.min())
    print("Max:", tensor.max())
    print("Mean:", tensor.mean())
    print("Std:", tensor.std())
    print("Shape:", tensor.size())


def plot_loss(
        lbe, 
        model_name, 
        plot_values='all', 
        line_style='b-', 
        save=False, 
        fname=None
    ):
    """
    Plots the loss by epoch.
    ---
    IN
    lbe: list of avg loss at the end of each epoch (list)
    model_name: name of model (str)
    plot_values: if 'all', plots loss for each batch; if 'epoch', plots
        average loss per epoch (str)
    NO OUT
    """

    msg = "plot_values must be 'all' or 'epoch'"
    assert plot_values == 'all' or plot_values == 'epoch', msg

    if isinstance(lbe, list):
        lbe = np.array([np.array(l) for l in lbe])

    fig = plt.figure(figsize=(12,8))
    
    if plot_values == 'all':
        y = lbe.reshape(1,-1).squeeze()
        unit = 'Batch'
    else:
        y = [epoch.mean() for epoch in lbe]
        unit = 'Epoch'
        
    x = range(1, len(y)+1)
    plt.plot(x, y, line_style)
    plt.grid(b=True)
    plt.xlabel(unit)
    plt.ylabel("Cross-Entropy Loss")
    if model_name:
        plt.title("{}: Loss Per {}".format(model_name, unit))
    if save:
        fpath = os.path.join("../viz", fname + ".png")
        plt.savefig(fpath, dpi=128)
    plt.show();
