from keras.models import *
from keras.layers import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )

VGG_Weights_path = file_path+"/weights/standard/vgg16_weights_th_dim_ordering_th_kernels_notop.h5"


def VGGSegnet_flex( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    return VGGSegnet(n_classes, None, None, vgg_level)


def VGGSegnet( n_classes ,  input_height, input_width , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first' )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first' )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first' )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first' )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_first' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_first' )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first' )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_first' )(x)
    f5 = x

    vgg  = Model(  img_input , x  )
    vgg.load_weights(VGG_Weights_path)

    for layer in vgg.layers:
        layer.trainable = False

    o = f4

    o = (ZeroPadding2D((1, 1), data_format='channels_first'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)

    o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_first'))(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(o)

    o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_first' ))(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)

    o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format='channels_first'))(o)

    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )( o )
    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]
    o = (Reshape((  n_classes  , -1 )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model


if __name__ == '__main__':
    m = VGGSegnet( 101 , 416, 608)
    from keras.utils import plot_model
    plot_model( m , show_shapes=True , to_file='model.png')

