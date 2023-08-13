from model import Colorization_Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dataset_ab = '/home/emir/Desktop/dev/myResearch/dataset/colorization_lab.npy'
dataset_gray = '/home/emir/Desktop/dev/myResearch/dataset/l/gray_scale.npy'

def save_plots(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plots to files
    plt.tight_layout()
    plt.savefig('accuracy_loss_plots.png')

def train(model, gray, ab, epochs, batch_size):
    # Setup the training input data (grayscale images)
    train_in = gray[:3000]
    del gray
    train_in = np.repeat(train_in[..., np.newaxis], 3, -1)
    print("emir")
    
    train_out = ab[:3000]
    del ab
    # Normalize the data
    train_in = (train_in.astype('float32') - 127.5) / 127.5
    train_out = (train_out.astype('float32') - 127.5) / 127.5
    history = model.fit(
        train_in,
        train_out,
        epochs=epochs,
        validation_split=0.2,
        batch_size=batch_size
    )
    model.save_weights('colorization_weights.h5')
    save_plots(history=history)

def run():
    input_shape = (None, 224, 224,3)
    model = Colorization_Model((224, 224, 3))
    model.build(input_shape)
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.0003), loss='mse', metrics=['accuracy'])
    gray = np.load(dataset_gray)
    ab = np.load(dataset_ab)
    print(len(gray))
    train(model=model, gray=gray, ab=ab, epochs=50, batch_size=8)


if __name__ == "__main__":
    run()
