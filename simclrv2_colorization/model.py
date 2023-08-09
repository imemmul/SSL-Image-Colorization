import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, UpSampling2D
from keras.models import Model

# TODO first train a network with grayscale images with strong augmentation on them
# TODO second fine tune last layer with %10 of labeled data
# TODO knowledge distillation

class ProjectionHeadPretext(tf.keras.layers.layer):
    def __init__(self, color_channels=3, **kwargs):
        super(ProjectionHeadFineTuning, self).__init__(**kwargs)
        # should classification linear layers for just learning ResNet50

class ProjectionHeadFineTuning(tf.keras.layers.Layer):
    def __init__(self, color_channels=3, **kwargs):
        super(ProjectionHeadFineTuning, self).__init__(**kwargs)
        self.color_channels = color_channels
        self.conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.up1 = UpSampling2D(size=(2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.up2 = UpSampling2D(size=(2, 2))  # Add this upsampling layer
        self.conv3 = Conv2D(self.color_channels, (3, 3), activation='sigmoid', padding='same')
    
    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)  # Upsample again to match the desired output shape
        colorized_output = self.conv3(x)
        return colorized_output

class ColorizationModel(Model):
    def __init__(self, color_channels=3):
        super(ColorizationModel, self).__init__()
        self.encoder = tf.keras.applications.ResNet50V2()
        self.decoder = ProjectionHeadPretext()

    def call(self, inputs, training):
        features = self.encoder(inputs, training=training)
        colorized_images = self.decoder(features, training=training)
        return colorized_images
