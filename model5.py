# EXPERIMENTS
# 1. NORMAL MODEL
# 2. With data normalization
# 3. With data augmentation
# 4. Change in number of features
# 5. Change in depth
# 6. Change in epoch
# 7. Change in Image size
# 8. Include dropout
# 9. (If needed) Transfer learning from InceptionV3 model
# 10. Change in optimizer with learning rate

# IMPORTS ###############
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import datetime

print(tf.__version__, keras.__version__)
# CREATE CONST ###########
TRAIN_PATH = '../../../new_internet_dataset/train/'
TEST_PATH = '../../../new_internet_dataset/test/'
DATASET_PATH = '../../../new_internet_dataset/'

# TODO: Change these parameter as per the need
EPOCH = 100
BATCH_SIZE = 8
STEPS_PER_EPOCH = 8
TARGET_SIZE = (150, 150)
VAL_STEPS = 4


class myCallbyack():
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.50):
            print("Reached 50% accuracy!")
            self.model.stop_training = True


callbyack = myCallbyack()

tensorboard = keras.callbacks.TensorBoard(
    log_dir='.\logs_model5_1',
    histogram_freq=1,
    write_images=True,
    write_graph=False
)
callbyack = [tensorboard]

test_generator = ImageDataGenerator(
    rescale=(1.0 / 255),
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_generator = ImageDataGenerator(
    rescale=(1.0 / 255)
)

train_gen = test_generator.flow_from_directory(
    TRAIN_PATH,
    # TODO: Change class_mode to 'categorical'
    # For multiclass classification use 'categorical' mode instead of 'binary' mode
    class_mode='binary',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE
    # color_mode='grayscale'
)

val_gen = val_generator.flow_from_directory(
    TEST_PATH,
    # TODO: Change class_mode to 'categorical'
    # For multiclass classification use 'categorical' mode instead of 'binary' mode
    class_mode='binary',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE
    # color_mode='grayscale'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    # TODO: Replace the last layer according to number of classes
    #
    # For categorical classification the output node will be the number of categories,
    # for binary classification the output node is 1
    # Also use 'softmax' activation funciton for categorical classification instead of 'sigmoid'
    keras.layers.Dense(1, activation='sigmoid')
])

# TODO: Uncomment me
# For Categorical Classification
# model.compile(
#     optimizer=keras.optimizers.Adam(lr=0.0001),
#     loss=keras.losses.categorical_crossentropy,
#     metrics=['accuracy']
# )

# TODO: Comment me
# For Binary Classification
model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCH,
    verbose=1,
    callbacks=callbyack,
    validation_data=val_gen,
    validation_steps=VAL_STEPS
)


model.save('my_model5')

from matplotlib import pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
fig, axs = plt.subplots(2)
axs[0].plot(epochs, acc, 'r-', label='Training accuracy')
axs[0].plot(epochs, val_acc, 'b', label='Validation accuracy')
axs[0].set_title('Training and validation accuracy')

# plt.figure()

axs[1].plot(epochs, loss, 'r-', label='Training Loss')
axs[1].plot(epochs, val_loss, 'b', label='Validation Loss')
axs[1].set_title('Training and validation loss')
plt.legend()

plt.show()
