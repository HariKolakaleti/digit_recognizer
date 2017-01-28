#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
#%%
"""

# import modules

import numpy as np
import os
import random
import sys
import gzip
import idx2numpy 
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image

np.random.seed(133)

#%% 

# MNIST dataset
mnist_url = 'http://yann.lecun.com/exdb/mnist/'

#%%

# download, extract and display sample images

mnist_data = './mnist_data/'
if not os.path.isdir(mnist_data):    
    print ('Creating dir:', mnist_data)
    os.mkdir(mnist_data)

# reused/modified from tensorflow 1_notmnist.ipynb

last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        

def maybe_download(url, filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url+filename, mnist_data+filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return mnist_data+filename

mnist_train_images_gz = maybe_download(mnist_url, 'train-images-idx3-ubyte.gz', 9912422)
mnist_train_labels_gz = maybe_download(mnist_url, 'train-labels-idx1-ubyte.gz', 28881)
mnist_test_images_gz  = maybe_download(mnist_url, 't10k-images-idx3-ubyte.gz', 1648877)
mnist_test_labels_gz  = maybe_download(mnist_url, 't10k-labels-idx1-ubyte.gz', 4542)

def maybe_extract(filename):
    fname = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isfile(fname):
        print('Skipping extraction of %s.' % (filename))
    else:
        cmd = 'gunzip {}'.format(filename)
        os.system(cmd)
    return fname

maybe_extract('mnist_data/train-images-idx3-ubyte.gz')
maybe_extract('mnist_data/train-labels-idx1-ubyte.gz')
maybe_extract('mnist_data/t10k-images-idx3-ubyte.gz')
maybe_extract('mnist_data/t10k-labels-idx1-ubyte.gz')

def display_samples(data_folder, num_samples=1):
    for i in range(num_samples):
        im_name = random.choice(os.listdir(data_folder))
        im_file = data_folder + "/" + im_name
        #display(Image(filename=im_file))

#display_samples(svhn_train_folder)
#display_samples(svhn_test_folder)
#display_samples(svhn_extra_folder)


# read data and convert idx file to numpy array
ndarr = idx2numpy.convert_from_file('mnist_data/train-images-idx3-ubyte')
labels_raw = idx2numpy.convert_from_file('mnist_data/train-labels-idx1-ubyte')

dataset_size = ndarr.shape[0]/5
image_height = 28
image_width = 140

def createSequences():
    dataset = np.ndarray(shape=(dataset_size, image_height, image_width),
                         dtype=np.float32)
    
    data_labels = []
    
    i = 0
    w = 0
    while i < dataset_size:
        temp = np.hstack(
            [ndarr[w], ndarr[w + 1], ndarr[w + 2], ndarr[w + 3], ndarr[w + 4]])
        dataset[i, :, :] = temp
        temp_str = (labels_raw[w], labels_raw[
            w + 1], labels_raw[w + 2], labels_raw[w + 3], labels_raw[w + 4])
        data_labels.append(temp_str)
        w += 5
        i += 1
        
        np.array(data_labels)
        
        return dataset, data_labels
        
        
dataset, data_labels = createSequences()
