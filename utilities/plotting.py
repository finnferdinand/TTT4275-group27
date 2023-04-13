from matplotlib import pyplot as plt
import numpy as np
import classifiers as cl

def draw_images(selection_data):
    images = selection_data[0]
    labels = selection_data[1]
    N_IMAGES = len(images) # must be even
    draw_images.counter += 1
    plt.figure(draw_images.counter)
    for index in range(N_IMAGES):
        plt.subplot(2, N_IMAGES//2, index+1)
        square_test_image = np.reshape(images[index], [cl.knn.row_size, cl.knn.col_size])
        plt.imshow(square_test_image, interpolation='nearest')
        plt.title(f"Classified: {labels[index]}")
draw_images.counter = 0
