from tensorflow.keras import layers, models
import tensorflow as tf

def build_3d_cnn_multilabel(input_shape, num_outputs):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x)
    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_outputs, activation='sigmoid')(x)
    return models.Model(inputs, outputs)
