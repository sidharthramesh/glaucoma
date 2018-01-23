from preprocess import train, val, train_steps, val_steps
from keras.applications import Xception
from keras import layers, models
from keras import callbacks, optimizers
# Uneven class distribution
class_weight = {
    train.class_indices['normal']:1,
    train.class_indices['glaucoma']: 1.86
}

# Using pretrained Xception Net as Convolutional feature extractor
conv_base = Xception(include_top=False,weights='imagenet',input_shape=(299,299,3))
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(520, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='softmax'))
model.compile(optimizers.RMSprop(1e-5),'binary_crossentropy',metrics=['acc'])

print(model.summary())
# Output to tensorboard
tensorboard = callbacks.TensorBoard('/output',1)

# Train model!
model.fit_generator(train,train_steps,20,callbacks=[tensorboard],validation_data=val,validation_steps=val_steps)

# Save to output
model.save('/output/model_data.h5')
