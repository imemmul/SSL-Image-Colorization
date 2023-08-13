from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, Activation, concatenate, LeakyReLU
from keras.optimizers import Adam
import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, inshape, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.base = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, input_shape=inshape, weights="imagenet")
        self.conv1= Conv2D(64, (3,3), strides=(2,2), padding="same", activation=LeakyReLU())
        self.conv2= Conv2D(128, (3,3), strides=(1,1), padding="same", activation=LeakyReLU())
        self.conv3= Conv2D(128, (3,3), strides=(2,2), padding="same", activation="relu")
        self.conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation="relu")
        self.conv5 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation="relu")
        self.conv6 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation="relu")
        self.conv7 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation="relu")
        self.dropout = Dropout(0.25)
        # got 7x7
        self.fusion = Conv2D(512, (1, 1), padding="same", activation="relu")
    def call(self, inputs):
        out_base = self.base(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        res_skip_1 = self.conv5(x)
        x = res_skip_1
        res_skip_2 = self.conv6(x)
        x = res_skip_2
        x = self.conv7(x) # last layer of conv2
        f = self.fusion(concatenate([out_base, x]))
        skip_f = concatenate([f, x])
        return skip_f, self.dropout(res_skip_1), self.dropout(res_skip_2) # res_skip conv4_

class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.convt1 = Conv2DTranspose(512, (3,3), strides=(2,2), padding="same", activation="relu")
        self.dropout = Dropout(0.25)
        self.convt2 = Conv2DTranspose(256, (3,3), strides=(2,2), padding="same", activation="relu")
        self.convt3 = Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", activation="relu")
        self.convt4 = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same", activation="relu")
        self.convt5 = Conv2DTranspose(64, (3,3), strides=(1,1), padding="same", activation="relu")
        self.convt5 = Conv2DTranspose(32, (3,3), strides=(2,2), padding="same", activation="relu")
        self.output_layer = Conv2D(2, (1,1), activation="tanh")
    def call(self, skip_f, res_skip_1, res_skip_2):
        dec = self.convt1(skip_f)
        dec = self.dropout(dec)
        dec = self.convt2(concatenate([dec, res_skip_2]))
        dec = self.dropout(dec)
        dec = self.convt3(concatenate([dec, res_skip_1]))
        dec = self.dropout(dec)
        dec = self.convt4(dec)
        dec = self.dropout(dec)
        dec = self.convt5(dec)
        return self.output_layer(dec)
            
        

class Colorization_Model(Model):
    def __init__(self, inshape):
        super(Colorization_Model, self).__init__()
        
        #encoder
        self.encoder = Encoder(inshape=inshape)
        
        #decoder
        self.decoder = Decoder()
        
    def call(self, inputs):
        skip_f, dropout_res_skip_1, dropout_res_skip_2 = self.encoder(inputs)
        return self.decoder(skip_f, dropout_res_skip_1, dropout_res_skip_2)




    
    
