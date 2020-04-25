import matplotlib.pyplot as plt
import numpy as np


def plot_sample_images(model, images, expected_labels, class_names):
    predictions = model.predict(images)
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions[i])
        if predicted_label == expected_labels[i]:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} ({})".format(class_names[predicted_label],
                                    class_names[expected_labels[i]]),
                   color=color)


def plot_n_images(images: list):
    plt.figure(figsize=(15, 5))
    N = len(images)

    for i in range(N):
        plt.subplot(2, 5, i + 1)
        # print(images[i].shape)
        nX, nY, _ = iter(images[i].shape)
        x = images[i].reshape((nX, nY))
        # print(x.shape)
        plt.imshow(x, cmap='gray')
        label = 'original'
        if i & 1 == 0:
            color = 'blue'
        else:
            color = 'red'
            label = 'after DAE propagation'
        plt.xlabel(label, color=color)


def plot_history(history):
    plt.rcParams["figure.figsize"] = (15, 7)
    plt.plot(history.history['loss'], label='loss (training data)')
    plt.plot(history.history['val_loss'], label='loss testing data')

    plt.plot(history.history['acc'], label='loss testing data')
    plt.plot(history.history['val_acc'], label='loss testing data')
    plt.title('Loss denoising autoencoder')
    plt.ylabel('MSE')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
