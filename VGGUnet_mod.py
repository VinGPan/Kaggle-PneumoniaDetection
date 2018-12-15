from keras.models import *
from keras.layers import *

import os
file_path = os.path.dirname( os.path.abspath(__file__) )


VGG_Weights_path = file_path+"/weights/standard/vgg16_weights_th_dim_ordering_th_kernels_notop.h5"

IMAGE_ORDERING = 'channels_first'


def VGGUnet_mod( n_classes ,  input_height, input_width , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    f3 = x

    # Block 6
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block6_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block6_conv2', data_format=IMAGE_ORDERING)(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool', data_format=IMAGE_ORDERING)(x2)
    f6 = x2

    # Block 7
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block7_conv1', data_format=IMAGE_ORDERING)(x2)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block7_conv2', data_format=IMAGE_ORDERING)(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool', data_format=IMAGE_ORDERING)(x2)
    f7 = x2

    # Block 8
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block8_conv1', data_format=IMAGE_ORDERING)(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block8_conv2', data_format=IMAGE_ORDERING)(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block8_conv3', data_format=IMAGE_ORDERING)(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block8_pool', data_format=IMAGE_ORDERING)(x2)
    f8 = x2

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    f5 = x

    vgg  = Model(  img_input , x  )
    vgg.load_weights(VGG_Weights_path)

    for layer in vgg.layers:
        layer.trainable = False

    x = Concatenate(axis=1)([f3, f8])
    f3_new = x
    # Block 9
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block9_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block9_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block9_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block9_pool', data_format=IMAGE_ORDERING)(x)
    f9 = x

    o = f9

    o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

    o = ( concatenate([ o , f3_new],axis=1 )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

    o = ( concatenate([o,f2, f7],axis=1 ) )
    o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

    o = ( concatenate([o,f1, f6],axis=1 ) )
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
    if (input_height is not None) and (input_width is not None):
        o = ( BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]
    o = (Reshape((  n_classes  , -1   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model


if __name__ == '__main__':
    m = VGGUnet_mod(2, 224, 224)
    m.summary()



