from preprocess import train, val, train_steps, val_steps
from keras.applications import Xception
from keras import layers, models
from keras import callbacks, optimizers
from preprocess import train, train_steps, val, val_steps
# Uneven class distribution
class_weight = {
    train.class_indices['normal']:1,
    train.class_indices['glaucoma']: 1.86
}

# Using pretrained Xception Net as Convolutional feature extractor
conv_base = Xception(include_top=False,weights='imagenet',input_shape=(299,299,3))

train = conv_base.predict_generator(train,train_steps)