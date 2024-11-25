import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, InceptionV3
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img

data_dir = '/kaggle/input/banana-disease-recognition-dataset/Banana Disease Recognition Dataset/Augmented images/Augmented images'

datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255)
train_generator = datagen.flow_from_directory(data_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(data_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_generator.classes), y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

def create_lenet():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = True  # Fine-tuning the ResNet50 base model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_inception():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = True  # Fine-tuning the InceptionV3 base model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

lenet = create_lenet()
history_lenet = lenet.fit(train_generator, validation_data=val_generator, epochs=3, class_weight=class_weights)

resnet = create_resnet()
history_resnet = resnet.fit(train_generator, validation_data=val_generator, epochs=3, class_weight=class_weights)

inception = create_inception()
history_inception = inception.fit(train_generator, validation_data=val_generator, epochs=3, class_weight=class_weights)

lenet.save('lenet.h5')
resnet.save('resnet.h5')
inception.save('inception.h5')

class_mapping = {
    0: "Banana Black Sigatoka Disease",
    1: "Banana Bract Mosaic Virus Disease",
    2: "Banana Healthy Leaf",
    3: "Banana Insect Pest Disease",
    4: "Banana Moko Disease",
    5: "Banana Panama Disease",
    6: "Banana Yellow Sigatoka Disease"
}

def predict_with_voting(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_lenet = lenet.predict(img_array)
    pred_resnet = resnet.predict(img_array)
    pred_inception = inception.predict(img_array)
    
    print("LeNet Prediction:", pred_lenet)
    print("ResNet Prediction:", pred_resnet)
    print("Inception Prediction:", pred_inception)
    
    final_pred = (pred_lenet + pred_resnet + pred_inception) / 3
    predicted_class = np.argmax(final_pred, axis=1)[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Disease: {class_mapping[predicted_class]}")
    plt.show()
    
    return class_mapping[predicted_class]

test_image = '/kaggle/input/banana-disease-recognition-dataset/Banana Disease Recognition Dataset/Augmented images/Augmented images/Augmented Banana Insect Pest Disease/Augmented Banana Insect Pest Disease (1).jpg'
predicted_disease = predict_with_voting(test_image)
print(f"Predicted Disease: {predicted_disease}")

def plot_accuracy(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_accuracy(history_lenet, 'LeNet')
plot_accuracy(history_resnet, 'ResNet50')
plot_accuracy(history_inception, 'InceptionV3')
