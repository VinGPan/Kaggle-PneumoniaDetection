
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# fc weights into the 1x1 convs  , get_upsampling_weight



from keras.models import *
from keras.layers import *
from keras.regularizers import l2

import os
file_path = os.path.dirname( os.path.abspath(__file__) )

VGG_Weights_path = file_path+"/weights/standard/vgg16_weights_th_dim_ordering_th_kernels_notop.h5"

IMAGE_ORDERING = 'channels_first'

# crop o1 wrt o2
def crop2( o1 , o2 , i, aug_input):
    if aug_input is None:
        o_shape2 = Model(i, o2).output_shape
        outputHeight2 = o_shape2[2]
        outputWidth2 = o_shape2[3]

        o_shape1 = Model(i, o1).output_shape
        outputHeight1 = o_shape1[2]
        outputWidth1 = o_shape1[3]
    else:
        o_shape2 = Model([i, aug_input], o2).output_shape
        outputHeight2 = o_shape2[2]
        outputWidth2 = o_shape2[3]

        o_shape1 = Model([i, aug_input], o1).output_shape
        outputHeight1 = o_shape1[2]
        outputWidth1 = o_shape1[3]

    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)

    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

    return o1 , o2


def FCN8_mod( nClasses ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))
    aug_input = Input(shape=(3,))
    concat_axis = 1

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    # x = Dropout(0.5)(x)
    # x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(1E-4),
    #                        beta_regularizer=l2(1E-4))(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    # x = Dropout(0.5)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    # x = Dropout(0.5)(x)
    f3 = x

    # Block 6
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block6_conv1', data_format=IMAGE_ORDERING )(img_input)
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block6_conv2', data_format=IMAGE_ORDERING )(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool', data_format=IMAGE_ORDERING )(x2)
    f6 = x2

    # Block 7
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block7_conv1', data_format=IMAGE_ORDERING )(x2)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block7_conv2', data_format=IMAGE_ORDERING )(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool', data_format=IMAGE_ORDERING )(x2)
    f7 = x2

    # Block 8
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block8_conv1', data_format=IMAGE_ORDERING )(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block8_conv2', data_format=IMAGE_ORDERING )(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block8_conv3', data_format=IMAGE_ORDERING )(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block8_pool', data_format=IMAGE_ORDERING )(x2)
    f8 = x2

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
    # x = Dropout(0.5)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    # x = Dropout(0.5)(x)
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

    # Block 10
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block10_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block10_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block10_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block10_pool', data_format=IMAGE_ORDERING)(x)
    f10 = x

    o = f10

    o = ( Conv2D( 512 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.1)(o)
    o = ( Conv2D( 512 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.1)(o)

    o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o2 = f9
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)

    o , o2 = crop2( o , o2 , img_input, None)

    o = Add()([ o , o2 ])

    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o2 = f3_new
    o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)

    tmp_shape = Model(img_input, o2).output_shape
    dense_sz = tmp_shape[1] * tmp_shape[2] * tmp_shape[3]
    o2 = (Flatten())(o2)
    o2 = Concatenate()([o2, aug_input])
    o2 = (Dense(dense_sz, activation='relu', kernel_initializer='he_normal'))(o2)
    o2 = (Reshape((tmp_shape[1], tmp_shape[2], tmp_shape[3])))(o2)

    o2 , o = crop2( o2 , o , img_input, aug_input )
    o  = Add()([ o2 , o ])

    o = Conv2DTranspose( nClasses , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o_shape = Model([img_input, aug_input] , o ).output_shape

    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape((  nClasses  , -1   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( [img_input, aug_input] , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model



if __name__ == '__main__':
    m = FCN8_mod(2, 224, 224)
    m.summary()
