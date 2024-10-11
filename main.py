import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #This library is used preprocessing and augmenting image data
from tensorflow.keras import layers, models

#Paths for training and testing data sets
train_dir = 'C:\\Users\\jackd\\OneDrive\\Documents\\Projects\\Brain tumor detection\\Training'
test_dir = 'C:\\Users\\jackd\\OneDrive\Documents\\Projects\\Brain tumor detection\\Testing'

#Image size and batch size
image_size = (128,128) #Image size will be 128x128 pixels
BATCH_SIZE = 32        #Process 32 images per batch

#Data preprocessing, and augmentation --> This will increase the size as well as the diversity of the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255, #Normalize pixel values to be between 0 and 1
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest',
    validation_split = 0.2
)

#Rescale the test data set
test_datagen = ImageDataGenerator(rescale=1./255) #Normalize pixel values to be between 0 and 1 --> No need augmentation


#Preparing the training dataset --> loading image from the directory
train_generator = train_datagen.flow_from_directory(
    train_dir, #specify the directory
    target_size = image_size,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'training'
)


validation_generator = train_datagen.flow_from_directory(
    train_dir, #specify the directory
    target_size = image_size,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'validation' #This portion of the dataset will be set aside to evaluate how well the model performs on data it hasn't seen
)


#Load and label the testing dataset
test_generator = test_datagen.flow_from_directory(
    test_dir, #specify the directory
    target_size = image_size,
    batch_size  = BATCH_SIZE,
    class_mode = 'categorical'
)


#Build the Convolutional Neural Network (CNN)
model = models.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening the output and fully connected layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Adding dropout to prevent overfitting
    layers.Dense(4, activation='softmax')  # 4 output classes (glioma, meningioma, pituitary, notumor)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

#Train the model
history = model.fit(train_generator, epochs=20, validation_data=validation_generator)

#Save the trained model
model.save('tumor_classification_model.keras')

# Load and evaluate the saved model on the test dataset
model = tf.keras.models.load_model('tumor_classification_model.keras')
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")