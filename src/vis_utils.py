import numpy as np
import matplotlib.pyplot as plt


def plot_image(ax, x, y=None):
    if not y is None:
        print 'Class: %d' % y

    ax.imshow(x, cmap='gray')
    ax.axis('off')


def plot_array(fig, X, Y, num_classes=10, samples_per_class=7):

    for y in range(num_classes):
        idxs = np.flatnonzero(Y == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)

        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            ax = fig.add_subplot(samples_per_class, num_classes, plt_idx)
            ax.imshow(X[idx][:,:,0], cmap='gray')
            ax.axis('off')


def plot_weights(fig, weights):

    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    for i, class_name in enumerate(classes):
        ax = fig.add_subplot(2, 5, i + 1)
        reshaped = weights[:, i].reshape((28, 28))
        
        pmin = np.min(reshaped)
        pmax = np.max(reshaped)
        
        normalised_first_row = ((reshaped - pmin) * (255 / (pmax - pmin))).round()
        ax.set_title(class_name.capitalize())
        ax.axis('off')

        ax.imshow(normalised_first_row, cmap='gray')


def plot_confusion_matrix(
    ax, matrix, labels, title='Confusion matrix', fontsize=9):

    ax.set_xticks([x for x in range(len(labels))])
    ax.set_yticks([y for y in range(len(labels))])
    # Place labels on minor ticks
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels, rotation='90', fontsize=fontsize, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize, minor=True)
    # Hide major tick labels
    ax.tick_params(which='major', labelbottom='off', labelleft='off')
    # Finally, hide minor tick marks
    ax.tick_params(which='minor', width=0)

    # Plot heat map
    proportions = [1. * row / sum(row) for row in matrix]
    ax.pcolor(np.array(proportions[::-1]), cmap=plt.cm.Reds)

    # Plot counts as text
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            confusion = matrix[::-1][row][col]
            if confusion != 0:
                ax.text(col + 0.5, row + 0.5, confusion, fontsize=fontsize,
                    horizontalalignment='center',
                    verticalalignment='center')

    # Add finishing touches
    ax.grid(True, linestyle=':')
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('prediction', fontsize=fontsize)
    ax.set_ylabel('actual', fontsize=fontsize)
    # fig.tight_layout()
    plt.show()


def plot_activation_maps(fig, activation_maps, num_rows, num_cols):

    for i in range(activation_maps.shape[-1]):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.axis('off')
        ax.imshow(activation_maps[0,:,:,i], cmap='gray')


if __name__ == '__main__':
    matrix = np.random.randint(0, 9, (10, 10))
    labels = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(matrix, labels)
