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
        nX, nY, _ = iter(images[i].shape)
        plt.imshow(images[i].reshape(shape=(nX,nY)), cmap='gray')
        label = 'original'
        if i & 1 == 0:
            color = 'blue'
        else:
            color = 'red'
            label = 'after DAE propagation'
        plt.xlabel(label, color=color)
