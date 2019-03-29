# FROM https://gist.github.com/timehaven/257eef5b0e2d9e2625a9eb812ca2226b

"""akmtdfgen:  A Keras multithreaded dataframe generator.

Works with Python 2.7 and Keras 2.x.

For Python 3.x, need to fiddle with the threadsafe generator code.


Test the generator_from_df() functions by running this file:

    python akmtdfgen.py


Threadsafe generator code below taken from the answer of user 

   https://github.com/parag2489

on the Keras issue

    https://github.com/fchollet/keras/issues/1638

which uses contributions from

    http://anandology.com/blog/using-iterators-and-generators/


The rest of this file was written by

  Ryan Woodard | AppNexus | Data Science | 2017



If you have bcolz errors like:


    `start`+`nitems` out of boundsException RuntimeError:
        RuntimeError('fatal error during Blosc
        decompression: -1',) in
        'bcolz.carray_ext.chunk._getitem' ignored


check that your versions are up to date.  Here is what I am using:

In [1]: import bcolz

In [2]: bcolz.print_versions()
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
bcolz version:     1.1.2
NumPy version:     1.13.1
Blosc version:     1.11.2 ($Date:: 2017-01-27 #$)
Blosc compressors: ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
Numexpr version:   2.6.2
Dask version:   not available (version >= 0.9.0 not detected)
Python version:    2.7.13 |Continuum Analytics, Inc.| (default, Dec 20 2016, 23:09:15)
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
Platform:          linux2-x86_64
Byte-ordering:     little
Detected cores:    12
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

"""
from __future__ import print_function

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import numpy as np
import pandas as pd
import bcolz
import threading

import os
import sys
import glob
import shutil


bcolz_lock = threading.Lock()
# old_blosc_nthreads = bcolz.blosc_set_nthreads(1)
# assert bcolz.blosc_set_nthreads(1) == 1

def safe_bcolz_open(fname, idx=None, debug=False):
    """Threadsafe way to read bcolz arrays.

    bcolz might have issues with multithreading and underlying blosc
    compression code.  Lots of discussion out there, here are some
    starting points:

      http://www.pytables.org/latest/cookbook/threading.html
      https://github.com/dask/dask/issues/1033

    Since our threads are read-only on the static bcolz array on disk,
    we'll probably be ok, but no guarantees.  Test, test, test!  It is
    so important that the auxiliary matrix rows stay properly aligned
    with the images DataFrame rows.
    """
    with bcolz_lock:

        if idx is None:
            X2 = bcolz.open(fname)
        else:
            X2 = bcolz.open(fname)[idx]

        if debug:
            
            df_debug = pd.DataFrame(X2, index=idx)
            # print(len(idx))

            assert X2.shape[0] == len(idx)
            assert X2.shape == df_debug.shape

            # Should see index matching int() of data values.
            # print(df_debug.iloc[:5, :5])
            # print(df_debug.iloc[-5:, -5:])

            df_debug = df_debug.astype(int)
            # print(df_debug.iloc[:5, :5])
            # print(df_debug.iloc[-5:, -5:])

            # Here is why we made the test data as we did.  Make sure
            # data cast to int (not rounded up!) matches index values.
            test_idx = (df_debug.subtract(df_debug.index.values, axis=0) == 0).all(axis=1)
            assert test_idx.all(), df_debug[~test_idx]

    return X2


class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
  #      assert self.lock is not bcolz_lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.

    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def generator_from_df(df, batch_size, target_size, features=None,
                      debug_merged=False):
    """Generator that yields (X, Y).

    If features is not None, assume it is the path to a bcolz array
    that can be indexed by the same indexing of the input df.

    Assume input DataFrame df has columns 'imgpath' and 'target', where
    'imgpath' is full path to image file.

    https://github.com/fchollet/keras/issues/1627
    https://github.com/fchollet/keras/issues/1638

    Be forewarned if/when you modify this function: some errors will
    not be explicit, appearing only as a generic:

      ValueError: output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: None

    It usually means something in your infinite loop is not doing what
    you think it is, so the loop crashes and returns None.  Check your
    DataFrame in this function with various print statements to see if
    it is doing what you think it is doing.

    Again, error messages will not be too helpful here--if in doubt,
    print().

    """
    if features is not None:
        assert os.path.exists(features)
        assert safe_bcolz_open(features).shape[0] == df.shape[0], "Features rows must match df!"

    # Each epoch will only process an integral number of batch_size
    # but with the shuffling of df at the top of each epoch, we will
    # see all training samples eventually, but will skip an amount
    # less than batch_size during each epoch.
    nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)

    # At the start of *each* epoch, this next print statement will
    # appear once for *each* worker specified in the call to
    # model.fit_generator(...,workers=nworkers,...)!
    #     print("""
    # Initialize generator:  
    #   batch_size = %d
    #   nbatches = %d
    #   df.shape = %s
    # """ % (batch_size, nbatches, str(df.shape)))

    count = 1
    epoch = 0

    # New epoch.
    while 1:

        # The advantage of the DataFrame holding the image file name
        # and the labels is that the entire df fits into memory and
        # can be easily shuffled at the start of each epoch.
        #
        # Shuffle each epoch using the tricky pandas .sample() way.
        df = df.sample(frac=1)  # frac=1 is same as shuffling df.
        
        epoch += 1
        i, j = 0, batch_size

        # Mini-batches within epoch.
        mini_batches_completed = 0
        for _ in range(nbatches):

            # Callbacks are more elegant but this print statement is
            # included to be explicit.
            # print("Top of generator for loop, epoch / count / i / j = "\
            #       "%d / %d / %d / %d" % (epoch, count, i, j))

            sub = df.iloc[i:j]

            try:

                # preprocess_input()
                # https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py#L389
                X = np.array([

                        (2 *

                         # Resizing on the fly is efficient enough for
                         # pre-caching when a GPU is training a
                         # mini-batch.  Here is where some additional
                         # data augmentation could take place.
                         (img_to_array(load_img(f, target_size=target_size))

                          / 255.0 - 0.5))

                        for f in sub.imgpath])

                Y = sub.target.values

                if features is None:

                    # Simple model, one input, one output.
                    mini_batches_completed += 1
                    yield X, Y

                else:

                    # For merged model: two input, one output.
                    #
                    # HEY: You should probably test this very
                    # carefully!

                    # Make (slightly) more efficient by removing the
                    # debug_merged check.
                    X2 = safe_bcolz_open(features, sub.index.values, debug=debug_merged)

                    mini_batches_completed += 1

                    yield [X, X2], Y
                    # Or:
                    # yield [X, bcolz.open(features)[sub.index.values]], Y

            except IOError as err:

                # A type of lazy person's regularization: with
                # millions of images, if there are a few bad ones, no
                # need to find them, just skip their mini-batch if
                # they throw an error and move on to the next
                # mini-batch.  With the shuffling of the df at the top
                # of each epoch, the bad apples will be in a different
                # mini-batch next time around.  Yes, they will
                # probably crash that mini-batch, too, but so what?
                # This is easier than finding bad files each time.

                # Let's decrement count in anticipation of the
                # increment coming up--this one won't count, so to
                # speak.
                count -= 1

                # Actually, we could make this a try...except...else
                # with the count increment.  Homework assignment left
                # to the reader.
                
            i = j
            j += batch_size
            count += 1



def file_path_from_db_id(db_id, pattern="blah_%d.png", top="/tmp/path/to/imgs"):
    """Return file path /top/yyy/xx/blah_zzzxxyyy.png for db_id zzzxxyyy.
    
      The idea is to hash into 1k top level dirs, 000 - 999, then 100
      second level dirs, 00-99, so that the following database ids
      result in the associated file paths:
    
      1234567     /tmp/path/to/imgs/567/34/blah_1234567.png
          432     /tmp/path/to/imgs/432/00/blah_432.png
        29847     /tmp/path/to/imgs/847/29/blah_29847.png
         1432     /tmp/path/to/imgs/432/01/blah_1432.png

      Notice that changing pattern to pattern="blah_%09d.png" and
      top="" would result in:

      1234567     567/34/blah_001234567.png
          432     432/00/blah_000000432.png
        29847     847/29/blah_000029847.png
         1432     432/01/blah_000001432.png

      In general, this will give a decent spread for up to 100 million images.

      If you have more than 10 million images, or your database ids are
      higher, then this function is easily modified.
    """
    s = '%09d' % db_id
    return os.path.join(top, s[-3:], s[-5:-3], pattern % db_id)


#
# Helper functions, just for blog post demo.
#
def new_tricks_from_old_dogs(stage, label):
    """Convert list of Kaggle data files into DataFrame generator format.

    That is, go from:


        cd /path/to/kaggle/data/
        ls train/dogs| head

        dog.1000.jpg
        dog.1001.jpg
        dog.1002.jpg
        dog.1003.jpg
        dog.1004.jpg
        dog.1005.jpg
        dog.1006.jpg
        dog.1007.jpg
        dog.1008.jpg
        dog.1009.jpg

    to this:

                                                     new                         orig  label
        760         /tmp/path/to/imgs/760/00/dog_760.jpg  validation/dogs/dog.760.jpg    dog
        7724       /tmp/path/to/imgs/724/07/dog_7724.jpg  validation/dogs/dog.7724.jpg   dog
        7685       /tmp/path/to/imgs/685/07/dog_7685.jpg  validation/dogs/dog.7685.jpg   dog


    Only including 'cat' and 'dog' in 'new' file name because the
    numbers in the cats/dogs directories are non-unique.  This avoids
    collisions.
    """
    s = "data/%s/%ss/*.jpg" % (stage, label)
    #print(s, os.abspath(os.curdir))
    old_dogs = glob.glob(s)
    print(len(old_dogs), stage, label)
    index = list(map(int, [d.split('.')[-2] for d in old_dogs]))
    new_tricks = [file_path_from_db_id(i, pattern='%s_%%d.jpg' % label) for i in index]
    return pd.DataFrame({'orig': old_dogs, 'new': new_tricks, 'label': label},  index=index)


def mv_to_new_hierarchy(row, orig='orig', new='new'):
    """Copy file from orig to new."""
    if os.path.exists(row[new]):
        return
    d, f = os.path.split(row[new])
    os.path.exists(d) or os.makedirs(d)  # , exist_ok=True)
    #os.rename(row[orig], row[new])  # If you just want to move, not copy.
    shutil.copy(row[orig], row[new])


def get_demo_data():
    """Create train and validation DataFrames for blog post demo.

    Create something like this:

    dftrain.sample(5)

                                             imgpath  target                     orig label
    object_id                                                                              
    1797       /tmp/path/to/imgs/797/01/cat_1797.jpg       0  train/cats/cat.1797.jpg   cat
    1678       /tmp/path/to/imgs/678/01/cat_1678.jpg       0  train/cats/cat.1678.jpg   cat
    1348       /tmp/path/to/imgs/348/01/dog_1348.jpg       1  train/dogs/dog.1348.jpg   dog
    1430       /tmp/path/to/imgs/430/01/cat_1430.jpg       0  train/cats/cat.1430.jpg   cat
    1664       /tmp/path/to/imgs/664/01/cat_1664.jpg       0  train/cats/cat.1664.jpg   cat

    dfvalid.sample(5)

                                             imgpath  target                          orig label
    object_id                                                                                   
    7625       /tmp/path/to/imgs/625/07/cat_7625.jpg       0  validation/cats/cat.7625.jpg   cat
    7729       /tmp/path/to/imgs/729/07/cat_7729.jpg       0  validation/cats/cat.7729.jpg   cat
    760         /tmp/path/to/imgs/760/00/dog_760.jpg       1   validation/dogs/dog.760.jpg   dog
    7724       /tmp/path/to/imgs/724/07/dog_7724.jpg       1  validation/dogs/dog.7724.jpg   dog
    7685       /tmp/path/to/imgs/685/07/dog_7685.jpg       1  validation/dogs/dog.7685.jpg   dog
    """
    
    df_train = pd.concat([new_tricks_from_old_dogs('train', 'dog'),
                          new_tricks_from_old_dogs('train', 'cat')])

    df_valid = pd.concat([new_tricks_from_old_dogs('validation', 'dog'),
                          new_tricks_from_old_dogs('validation', 'cat')])

    # The only time we'll copy image files, just for directory hierarchy demo.
    res = df_train.apply(mv_to_new_hierarchy, axis=1)
    res = df_valid.apply(mv_to_new_hierarchy, axis=1)
    
    # Belt and suspenders for demo purposes.
    assert all([df['new'].apply(lambda n: os.path.exists(n)).all()
                for df in (df_train, df_valid)])
    
    # dog will be target 1, cat 0.
    df_train['target'] = (df_train['label'] == 'dog').astype(int)
    df_valid['target'] = (df_valid['label'] == 'dog').astype(int)

    df_train.index.name = 'object_id'
    df_valid.index.name = 'object_id'

    cols = ['imgpath', 'target', 'orig', 'label']  # For ordering.
    df_train = df_train.rename(columns={'new': 'imgpath'})[cols].reset_index()
    df_valid = df_valid.rename(columns={'new': 'imgpath'})[cols].reset_index()

    pd.options.display.width = 200
    print("Some samples:", "", "df_train:", df_train.sample(5), sep='\n')
    print("df_valid:", "", df_valid.sample(5), sep='\n')
  
    return df_train, df_valid


def test_generator():
    """Simple function to test return behavior of generator code above.

    This runs with and without merged model version.

df_train:
      object_id                                imgpath  target                          orig label
7          1518  /tmp/path/to/imgs/518/01/dog_1518.jpg       1  data/train/dogs/dog.1518.jpg   dog
1113       1662  /tmp/path/to/imgs/662/01/cat_1662.jpg       0  data/train/cats/cat.1662.jpg   cat
980        1409  /tmp/path/to/imgs/409/01/dog_1409.jpg       1  data/train/dogs/dog.1409.jpg   dog
1615       1813  /tmp/path/to/imgs/813/01/cat_1813.jpg       0  data/train/cats/cat.1813.jpg   cat
1029       1760  /tmp/path/to/imgs/760/01/cat_1760.jpg       0  data/train/cats/cat.1760.jpg   cat
df_valid:

     object_id                                imgpath  target                               orig label
787       7747  /tmp/path/to/imgs/747/07/cat_7747.jpg       0  data/validation/cats/cat.7747.jpg   cat
165       7563  /tmp/path/to/imgs/563/07/dog_7563.jpg       1  data/validation/dogs/dog.7563.jpg   dog
749       7517  /tmp/path/to/imgs/517/07/cat_7517.jpg       0  data/validation/cats/cat.7517.jpg   cat
458       7742  /tmp/path/to/imgs/742/07/cat_7742.jpg       0  data/validation/cats/cat.7742.jpg   cat
225       7479  /tmp/path/to/imgs/479/07/dog_7479.jpg       1  data/validation/dogs/dog.7479.jpg   dog

    """

    pd.np.set_printoptions(linewidth=150)
    
    df_train, df_valid = get_demo_data()

    img_width, img_height = 150, 150
    batch_size = 64
    target_size = (img_width, img_height)
    
    print("\nTest basic generator.\n")
    for df in (df_train, df_valid):
        i = 0
        for X, Y in generator_from_df(df, batch_size, target_size, features=None):
            print(X[:3, :3, 0])
            print(Y[:3])
            i += 1
            if i > 1:
                break

    # Create random array for bcolz test.
    #
    # In the end, this test does not use bcolz.
    # But, if it did, here are some hints to get you there.
    print("\nTest merged generator.\n")
    
    nfeatures = 74

    # features_train = pd.np.random.randn(df_train.shape[0], nfeatures)
    # features_valid = pd.np.random.randn(df_valid.shape[0], nfeatures)

    # Make a 2D array, where each row is filled with the values of its
    # index, which will be very convenient for testing the merged
    # model generator.
    # [[0, 0, 0, ...],
    #  [1, 1, 1, ...],
    #  [2, 2, 2, ...],
    #  ...
    # ]
    features_train = np.repeat(np.arange(df_train.shape[0], dtype=float)
                               .reshape((-1, 1)),
                               nfeatures, axis=1)
    features_valid = np.repeat(np.arange(df_valid.shape[0], dtype=float)
                               .reshape((-1, 1)),
                               nfeatures, axis=1)

    # Add a litle noise in [0, 1] just to pretend we have "real" data.
    features_train += np.random.rand(*features_train.shape)
    features_valid += np.random.rand(*features_valid.shape)
    
    fname_train = "mm_features_train_bc"
    if not os.path.exists(fname_train):
        c = bcolz.carray(features_train, rootdir=fname_train, mode='w')
        c.flush()

    fname_valid = "mm_features_valid_bc"
    if not os.path.exists(fname_valid):
        c = bcolz.carray(features_valid, rootdir=fname_valid, mode='w')
        c.flush()
        
    # Big assumption here: each row of a features matrix corresponds
    # exactly with the image represented by the row of the associated
    # train or valid df.  *YOU* will have to ensure this in your own
    # code.  This is only demo code!

    for df, fname in ((df_train, fname_train),
                      (df_valid, fname_valid)):

        nbatches = df.shape[0] / float(batch_size)

        for i, ((X, features), Y) in enumerate(
                generator_from_df(df, batch_size, target_size,
                                  features=fname, debug_merged=True)):

            if i == 0:
                print(X[:3, :3, 0])
                print(features[:3, :5])
                print(Y[:3])
            else:
                if (i + 1) % 20 == 0:
                    print("%d / %d" % (i + i, nbatches), end=', ')
                    sys.stdout.flush()

            # Keras automatically breaks out of the infinite "while 1"
            # loop in the generator_from_df().  For this test, we need
            # to break manually.
            if i >= nbatches:
                break

    print("\nSuccessful (I think...) test of multithreaded read of bcolz!")

    print("Note that for this test, all of the above X2 rows should"\
          "have the same int() values within a row.")


if __name__ == '__main__':
    test_generator()

