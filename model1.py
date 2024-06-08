import os
import csv
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten, BatchNormalization, LayerNormalization
from keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.initializers import he_normal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_PATH = '/gpfs/scratch/nct_299/MAMe/MAMe_data_256/data_256'
METADATA_PATH = '/gpfs/scratch/nct_299/MAMe/MAMe_metadata'

np.random.seed(42)
tf.random.set_seed(314)

batch_size = 128
n_epochs = 50

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH,'train'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42)

val_datagen = ImageDataGenerator(
    rescale=1 / 255.0)

val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH,'val'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

test_datagen = ImageDataGenerator(
    rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH,'test'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

def load_data_labels(filename):
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[1])
    return labels

def plot_training_curve(history):
    # Plot the training and validation loss and accuracy
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(history.history['loss'], label='train_loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='train_acc')
    ax[1].plot(history.history['val_accuracy'], label='val_acc')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].grid(True)
    ax[1].legend()
    
    # Save the training curves with a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('Timestamp curves of this experiment:',timestamp)
    plt.savefig(f'training_curves_{timestamp}.pdf')


def plot_loss_variation(history):
    # Calcula la variació de la pèrdua per època
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_loss_variation = np.diff(train_loss)
    val_loss_variation = np.diff(val_loss)

    # Ploteja la variació de la pèrdua
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_variation, label='Variació de Pèrdua d\'Entrenament')
    plt.xlabel('Èpoques')
    plt.ylabel('Variació de la Pèrdua')
    plt.title('Variació de la Pèrdua d\'Entrenament')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_loss_variation, label='Variació de Pèrdua de Validació')
    plt.xlabel('Èpoques')
    plt.ylabel('Variació de la Pèrdua')
    plt.title('Variació de la Pèrdua de Validació')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def train_cnn():
    # Create model architecture
    img_rows, img_cols, channels = 256, 256, 3
    input_shape = (img_rows, img_cols, channels)
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=he_normal()))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer=he_normal()))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(16, activation='relu', kernel_initializer=he_normal()))
    model.add(Dense(29, activation='softmax', kernel_initializer=he_normal()))

    # Choose optimizer and compile the model
    learning_rate = 0.1
    mom = 0.8
    sgd = SGD(learning_rate=learning_rate,momentum=mom)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    #Check model summary!
    print(model.summary())
    
    #Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    
    # Train the model
    history = model.fit(train_generator, validation_data = val_generator, epochs=n_epochs, verbose=1, callbacks=[early_stop])
    
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(loss, accuracy))
    #Classification outputs
    y_pred = model.predict(val_generator)
    y_true = val_generator.classes
    print('y_pred:',y_pred)
    #Assign most likely label
    y_pred = np.argmax(y_pred, axis=1)
    #Read data labels
    labels = load_data_labels(os.path.join(METADATA_PATH,'MAMe_labels.csv'))
    print(labels) 
    print(classification_report(y_true, y_pred,target_names=labels))
    print(confusion_matrix(y_true, y_pred))
    #Curves
    print('Plotting curves')
    plot_training_curve(history)
    plot_loss_variation(history)
    print('Plotting curves done')
    
    # Save and download the model with the name of the file
    model_filename = f'MAMe_model_1'
    model.save(model_filename)
    print(f'Model saved as {model_filename}')
    
    # Save all outputs in a folder with the name of the file
    output_folder = f'outputs_{model_filename}'
    os.makedirs(output_folder, exist_ok=True)
    plot_training_curve(history)
    plot_loss_variation(history)
    print(f'Outputs saved in {output_folder}')


if __name__ == "__main__":
    train_cnn()

