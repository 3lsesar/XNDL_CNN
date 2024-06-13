# AlexNet2

import os
import csv
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten, BatchNormalization, LayerNormalization, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2

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
n_epochs = 1000
img_size = (256, 256)


def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label


def load_dataset(directory, subset):
    dataset_dir = os.path.join(directory, subset)
    classes = sorted([d for d in os.listdir(dataset_dir)
                     if os.path.isdir(os.path.join(dataset_dir, d))])
    class_indices = dict((name, index) for index, name in enumerate(classes))

    filepaths = []
    labels = []

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith('.jpg'):
                filepaths.append(os.path.join(class_dir, fname, fname))
                labels.append(class_indices[class_name])

    filepaths = np.array(filepaths)
    labels = np.array(labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes))

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if subset == 'train':
        dataset = dataset.shuffle(len(filepaths))
    dataset = dataset.map(
        parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


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
    print('Timestamp curves of this experiment:', timestamp)
    plt.savefig(f'training_curves_model2.pdf')


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
    # Definir la arquitectura del modelo siguiendo la imagen
    img_rows, img_cols, channels = 256, 256, 3

    input_shape = (img_rows, img_cols, channels)
    
    model = Sequential()
    
    # Primera capa convolucional y de max pooling
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Segunda capa convolucional y de max pooling
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Tercera capa convolucional y de max pooling
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Cuarta capa convolucional y de max pooling
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Quinta capa convolucional y de max pooling
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Aplanar las capas para pasar a las capas completamente conectadas
    model.add(GlobalAveragePooling2D())
    
    # Capas completamente conectadas
    model.add(Dense(4096, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(29, activation='softmax', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01)))

    # Choose optimizer and compile the model
    learning_rate = 0.001
    mom = 0.8
    sgd = SGD(learning_rate=learning_rate, momentum=mom)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    # Check model summary!
    print(model.summary())

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

    # Train the model
    train_dataset = load_dataset(DATASET_PATH, 'train')
    val_dataset = load_dataset(DATASET_PATH, 'val')

    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=n_epochs, verbose=1, callbacks=[early_stop])

    loss, accuracy = model.evaluate(val_dataset, verbose=1)
    print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(
        loss, accuracy))

    y_pred = model.predict(val_dataset)
    y_true = np.concatenate([y for x, y in val_dataset], axis=0)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    labels = load_data_labels(os.path.join(METADATA_PATH, 'MAMe_labels.csv'))
    print(classification_report(y_true, y_pred, target_names=labels))
    print(confusion_matrix(y_true, y_pred))

    print('Plotting curves')
    plot_training_curve(history)
    print('Plotting curves done')


if __name__ == "__main__":
    from time import time
    start = time()
    train_cnn()
    print('Total time:', time()-start)

