from keras.preprocessing.image import ImageDataGenerator

# Data Augmentation and rescale
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=40,
    zoom_range=0.4,)

# Only rescale
test_val_gen = ImageDataGenerator(rescale=1./255)

# Resizing for Xception Net
train = train_gen.flow_from_directory('/data/train', target_size=(299, 299), class_mode='binary', batch_size=16)
val = test_val_gen.flow_from_directory('/data/validation', target_size=(299, 299), class_mode='binary',batch_size=16)
test = test_val_gen.flow_from_directory('/data/test', target_size=(299, 299), class_mode='binary',batch_size=16)
# Total samples/batch_size(32) 
train_steps = 30
val_steps = 8
test_steps = 4


