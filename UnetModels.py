import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, concatenate, Add, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal


##################################################
############## Standard UNet Model ###############
##################################################


def conv_block(x, filters, dropout_rate):
    """Convolutional block: two sets of (CONV -> RELU)."""
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(x)
    x = Dropout(dropout_rate)(x)
    return x

def unet_standard(input_shape, initial_filters=64, dropout_rate=0.5):
    inputs = Input(shape=input_shape)
    
    # Contracting/downsampling path
    c1 = conv_block(inputs, initial_filters, dropout_rate)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, initial_filters*2, dropout_rate)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, initial_filters*4, dropout_rate)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, initial_filters*8, dropout_rate)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = conv_block(p4, initial_filters*16, dropout_rate)
    
    # Expanding/upsampling path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = conv_block(u6, initial_filters*8, dropout_rate)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = conv_block(u7, initial_filters*4, dropout_rate)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = conv_block(u8, initial_filters*2, dropout_rate)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = conv_block(u9, initial_filters, dropout_rate)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    # model.summary()
    
    return model

# Example usage
# For a 128x128 RGB image
# model = unet_standard(input_shape=(128, 128, 3))  
# model.summary()






##################################################
########### Residual UNet (Modified) #############
##################################################

def res_block(x, filters, l2_strength, dropout_rate=0.5):
    """Residual block with batch normalization, L2 regularization, and dropout."""
    res = Conv2D(
        filters, 
        (3, 3), 
        activation='relu', 
        padding='same',
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(l2_strength)
        )(x)
    res = BatchNormalization()(res)
    res = Conv2D(
        filters, 
        (3, 3), 
        activation='relu', 
        padding='same', 
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(l2_strength)
        )(res)
    res = BatchNormalization()(res)
    res = Dropout(dropout_rate)(res)
    
    # Residual connection
    shortcut = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(l2_strength))(x)
    return Add()([shortcut, res])

def unet_with_residuals(input_shape, num_classes=1, initial_filters=64, l2_strength=0.01, dropout_rate=0.5):
    inputs = Input(shape=input_shape)
    
    # Contracting/downsampling path
    c1 = res_block(inputs, initial_filters, l2_strength, dropout_rate)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = res_block(p1, initial_filters*2, l2_strength, dropout_rate)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = res_block(p2, initial_filters*4, l2_strength, dropout_rate)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = res_block(p3, initial_filters*8, l2_strength, dropout_rate)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = res_block(p4, initial_filters*16, l2_strength, dropout_rate)
    
    # Expanding/upsampling path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = res_block(u6, initial_filters*8, l2_strength, dropout_rate)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = res_block(u7, initial_filters*4, l2_strength, dropout_rate)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = res_block(u8, initial_filters*2, l2_strength, dropout_rate)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = res_block(u9, initial_filters, l2_strength, dropout_rate)
    
    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    # model.summary()

    return model

# Example usage
# For a 128x128 RGB image with 2 classes
# model = unet_with_residuals(input_shape=(128, 128, 3), 
#                             num_classes=2,
#                             l2_strength=0.01,
#                             dropout_rate=0.5)  
                            




##################################################
############## MultiRes UNet Model ###############
##################################################

# Code from https://github.com/nibtehaz/MultiResUNet

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = Add()([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = Add()([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = Add()([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])
    # model.summary()

    return model

if __name__ == '__main__':
    pass