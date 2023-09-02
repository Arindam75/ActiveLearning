
import tensorflow as tf
import numpy as np
from tensorflow.image import resize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

def parse_function(filename, label):
    
    img = tf.io.read_file(filename)
    img= tf.image.decode_image(img,channels=3,expand_animations = False)
    img=resize(img, (64,64))
    img = tf.cast(img, tf.float32)/255
    return img, label

def build_dataset(x,y,repeat=True,batch=16, shuffle = True):

    '''
    Returns a tensorflow batched dataset, that can be used
    in the keras.model.fit method

        Parameters:
                x : numpy.ndarray
                        a numpy array with image filenames
                        e.g
                        array(['C:\\path\\to\\file\\dog.11481.jpg',
                               'C:\\path\\to\\file\\cat.680.jpg',
                               'C:\\path\\to\\file\\cat.9695.jpg'],
                                dtype='<U59')
                
                y : numpy.ndarray
                        one hot encoded labels
                        e.g
                        array([[0., 1.],
                               [1., 0.],
                               [1., 0.]], dtype=float32)
                repeat : boolean
                        True : when iterated over , the dataset starts generating batches
                        from the beginning if the end is reached. Suitable
                        for training set. 
                        False : will not start afresh, if iteration reaches end. suitable for
                        validation and test set
                batch : int
                        the number of samples in each batch the dataset generates when iterated.
                shuffle : boolean
                        True : Shuffle all the samples once epoch end is reached.
                        False : do not shuffle
    
        Returns:
                dataset : tensorflow.python.data.ops.dataset_ops.BatchDataset
                
    '''
    
    images = tf.constant(x) 
    labels = tf.constant(y)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(y.shape[0],reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(parse_function)
    dataset=dataset.batch(batch)

    return dataset


def simple_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

    
    return model

def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["binary_accuracy"]
    val_accuracy = val_accuracy + history.history["val_binary_accuracy"]
    return losses, val_losses, accuracy, val_accuracy

