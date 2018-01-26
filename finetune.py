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

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizers.RMSprop(1e-5),'binary_crossentropy',metrics=['accuracy'])
model.load_weights('/model/best_weights.h5')

conv_base.trainable = True
for layer in conv_base.layers:
    if layer.name != 'block14_sepconv2':
        layer.trainable = False

model.compile(optimizers.RMSprop(1e-5),'binary_crossentropy',metrics=['accuracy'])


print("Starting fine tuning")


model.fit_generator(train,train_steps,30,callbacks=[callbacks.TensorBoard('/output',1), callbacks.ModelCheckpoint('/output/best_weights.h5', save_best_only=True, verbose=1)],validation_data=val,validation_steps=val_steps, class_weight=class_weight)
model.save('/output/fine_tuned.h5')