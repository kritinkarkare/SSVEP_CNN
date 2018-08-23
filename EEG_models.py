from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Flatten
import keras.backend as K
from keras.constraints import max_norm
from keras.regularizers import l1_l2
from keras.layers.core import Dense, Activation, Dropout, Permute, Reshape


def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   

def safe_log(x):
    return K.log(x + 1e-7)


n_ch = 9
n_samp = 247
n_class = 8
dropout_rate = 0.5


def ShallowConvNet(input_shape):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), arXiv 1703.0505
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25
    """


#    if K.image_data_format() == 'channels_first':
#        input_shape = (1, Chans, Samples)
#    else:
#        input_shape = (Chans, Samples, 1)
#    print(input_shape)
    # start the model
    input_EEG = Input(input_shape) 
    block1       = Conv2D(10, (1, 13), 
                                 input_shape=(1, n_ch, n_samp),
                                 kernel_constraint = max_norm(2.))(input_EEG)
    block1       = Conv2D(10, (n_ch, 1), use_bias=False, 
                          kernel_constraint = max_norm(2.))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 30), strides=(1, 10))(block1)
    block1       = Activation(safe_log)(block1)
    block1       = Dropout(dropout_rate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(n_class, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_EEG, outputs=softmax)