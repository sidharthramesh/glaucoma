from preprocess import train, val, train_steps, val_steps, test, test_steps
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
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizers.RMSprop(1e-5),'binary_crossentropy',metrics=['accuracy'])


print(model.summary())
print("Loading weights...")
model.load_weights('/model/best_weights.h5')
print("Evaluating on test data")
print(model.evaluate_generator(test, test_steps))
# Output to tensorboard
# Train model!
#model.fit_generator(train,train_steps,30,callbacks=[callbacks.TensorBoard('/output',1), callbacks.ModelCheckpoint('/output/best_weights.h5', save_best_only=True, verbose=1)],validation_data=val,validation_steps=val_steps, class_weight=class_weight)

# Save to output
#model.save('/output/trained_dense.h5')