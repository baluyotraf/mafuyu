import numpy as np
import matplotlib.pyplot as plt


def print_distribution(stat, name, rows=3, plot_size=(10, 2)):
    stat = np.asarray(stat)
    ps = np.arange(0, 100 + 1, 10)
    stat_p = np.stack([ps, np.percentile(stat, ps)], axis=-1)
    stat_rows = [stat_p[i::rows] for i in range(rows)]

    print('{} Percentiles: '.format(name))

    for row in stat_rows:
        for item in row:
            print('{:6.2f}%: {:7.4f}'.format(item[0], item[1]), end='\t')
        print()

    plt.figure(figsize=plot_size)
    plt.hist(stat, bins=100)
    plt.show()
