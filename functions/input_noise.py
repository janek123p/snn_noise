import numpy as np
import matplotlib.pyplot as plt

def salt_and_pepper(mnist_1d_arr, alpha):
    num_salt = int(alpha*28*28/2)
    rand_indices = np.random.choice(28*28, num_salt * 2)
    salt_indices = rand_indices[:num_salt]
    pepper_indices = rand_indices[num_salt:]
    copy = np.copy(mnist_1d_arr)
    copy[salt_indices] = 0
    copy[pepper_indices] = 255
    return copy

def remove_rectangle(mnist_1d_arr, min_size, max_size):
    width = min_size + int(np.random.rand()*(max_size-min_size+1))
    height = min_size + int(np.random.rand()*(max_size-min_size+1))
    x = int(np.random.rand() * (28-width))
    y = int(np.random.rand() * (28-height))
    copy = mnist_1d_arr.reshape(28,28)
    copy[y:y+height+1, x:x+width+1] = 0
    return copy.flatten()

def plot_salt_and_pepper_examples(mnist_arrs, num_rows, num_cols, alpha,datapath):
    fig, axes = plt.subplots(num_rows, num_cols, num = 22, clear = True, constrained_layout = True)
    indices = np.random.choice(mnist_arrs.shape[0],num_rows * num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            mnist_arr = mnist_arrs[indices[i*num_rows+j],:,:].flatten()
            mnist_arr = salt_and_pepper(mnist_arr, alpha).reshape(28,28)
            im = axes[i,j].imshow(mnist_arr, interpolation="nearest", cmap = 'gray', aspect='equal', extent = [0, 28,0,28], vmin=0., vmax=255)

    fig.colorbar(im, ax = axes.ravel().tolist())
    fig.suptitle("Salt and pepper noise alpha = %.3f" % alpha)
    plt.axis('off')
    plt.savefig(datapath+'plots/salt_and_pepper_examples.png')

def plot_remove_rectangle_examples(mnist_arrs, num_rows, num_cols, min_w, max_w, datapath):
    fig, axes = plt.subplots(num_rows, num_cols, num = 23, clear = True, constrained_layout = True)
    indices = np.random.choice(mnist_arrs.shape[0],num_rows * num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            mnist_arr = mnist_arrs[indices[i*num_rows+j],:,:].flatten()
            mnist_arr = remove_rectangle(mnist_arr, min_w, max_w).reshape(28,28)
            im = axes[i,j].imshow(mnist_arr, interpolation="nearest", cmap = 'gray', aspect='equal', extent = [0, 28,0,28], vmin=0., vmax=255)

    fig.colorbar(im, ax = axes.ravel().tolist())
    fig.suptitle("MNIST images with removed rectangle\nmin width/height = %d and max width/height = %d" % (min_w, max_w))
    plt.axis('off')
    plt.savefig(datapath+'plots/remove_rectangle_examples.png')