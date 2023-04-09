# Importing Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

#tensorboard = TensorBoard(log_dir='/Users/garimagandhi/Downloads')
# Creating and initializing the CNN model

classifier = Sequential()

# Adding first convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# The input shape is set to the pooled feature maps from the previous convolution layer
# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=4, activation='softmax'))

# Compiling the model
#categorical_crossentropy for multi-class classification
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Preparing the train/test data and training the model

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Adding early stopping callback
#earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

#checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)


classifier.fit(
        training_set,
        steps_per_epoch=1000/32, # No of images in training set
        epochs=8,
        validation_data=test_set,
        validation_steps=100/32)
        #callbacks=[checkpoint])
        #callbacks=[earlystop])

# Saving the models
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')

classifier.summary()