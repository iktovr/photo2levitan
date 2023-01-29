import matplotlib.pyplot as plt
import numpy as np


def plot_image(ax, image, interp='nearest'):    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set(aspect='equal')
    ax.imshow((np.array(image).transpose(1, 2, 0) + 1) / 2, interpolation=interp)


def plot_images(images, indexes=None, w=7, h=None, titles=None, interp='nearest'):
    indexes = range(len(images)) if indexes is None else indexes
    
    h = h or (len(indexes) - 1) // w + 1
    assert(w * h >= len(indexes))
    
    fig = plt.figure(figsize=(w * 3, h * 3))
    for i, k in enumerate(indexes, 1):
        ax = fig.add_subplot(h, w, i)
        plot_image(ax, images[k], interp=interp)
        if titles is not None and k < len(titles):
            ax.set_title(titles[k])
    
    plt.show()


def plot_history(losses_g, losses_d):
    plt.figure(figsize=(10, 6))
    plt.plot(losses_g, label='generator loss')
    plt.plot(losses_d, label='discriminator loss')
    plt.legend()
    plt.show()
