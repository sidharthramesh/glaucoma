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
conv_base = Xception(include_top=False,weights=None,input_shape=(299,299,3))
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizers.RMSprop(1e-5),'binary_crossentropy',metrics=['accuracy'])
print(model.summary())


print("Loading pre-finetune weights...")
model.load_weights('/model0/trained_dense.h5')
print("Test Accuracy:{}".format(model.evaluate_generator(test, test_steps)[1]))

print("Loading pre-finetune Best Validation weights...")
model.load_weights('/model0/best_weights.h5')
print("Test Accuracy:{}".format(model.evaluate_generator(test, test_steps)[1]))

with open('/output/not_finetuned.json','w') as f:
    f.write(model.to_json())

conv_base.trainable = True
for layer in conv_base.layers:
    if layer.name != 'block14_sepconv2':
        layer.trainable = False

model.compile(optimizers.RMSprop(1e-5),'binary_crossentropy',metrics=['accuracy'])

print("Loading Fine tuned weights...")
model.load_weights('/model/fine_tuned.h5')
print("Test Accuracy:{}".format(model.evaluate_generator(test, test_steps)[1]))

print("Loading Fine tuned Best Validation weights...")
model.load_weights('/model/best_weights.h5')
print("Test Accuracy:{}".format(model.evaluate_generator(test, test_steps)[1]))

print("Writing model to file...")
with open('/output/finetuned.json','w') as f:
    f.write(model.to_json())

# Output to tensorboard
# Train model!
#model.fit_generator(train,train_steps,30,callbacks=[callbacks.TensorBoard('/output',1), callbacks.ModelCheckpoint('/output/best_weights.h5', save_best_only=True, verbose=1)],validation_data=val,validation_steps=val_steps, class_weight=class_weight)

# Save to output
#model.save('/output/trained_dense.h5')