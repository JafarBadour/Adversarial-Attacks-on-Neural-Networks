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
        shape = (images[i].shape[0], images[i].shape[1])
        plt.imshow(images[i].reshape(shape), cmap=plt.cm.binary)
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


def plot_confidence(x_train_adv_images, model, train_images):
    bcons = []
    acons = []
    print(x_train_adv_images.shape)
    Num = len(x_train_adv_images)
    for id in range(Num):
        benign = train_images[id]
        adversarial = x_train_adv_images[id]
        probas = np.array([benign, adversarial])
        y_prob = model.predict(probas)
        benign_conf = max(y_prob[0])
        adversarial_conf = max(y_prob[1])
        bcons.append(benign_conf)
        acons.append(adversarial_conf)
    # plt.imshow(benign.reshape((28,28)),cmap='gray')
    # plt.show()
    plt.rcParams["figure.figsize"] = (30, 15)
    plt.plot([i for i in range(Num)], bcons, '--r')
    plt.plot([i for i in range(Num)], acons, '--b')
    plt.title('Confidence of benign vs adversarial samples', fontsize=37)
    plt.ylabel('Confidence %', fontsize=37)
    plt.xlabel('id of picture', fontsize=37)

    plt.show()
