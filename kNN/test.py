import kNN
import numpy as np
import operator

# kNN example
data_set = np.array([[1.0, 1.0], [1.0, 0.9], [0.1, 0.1], [0.1, 0.2]])
labels = ['A', 'A', 'B', 'B']
in_x = [0.2, 0.3]

classfied = kNN.classify0(in_x, data_set, labels, 3)
print('in_x classified: %s' % classfied)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('kNN Algorithm Example')
ax.scatter(data_set[0:2, 0], data_set[0:2, 1], c='r', label='A')
ax.scatter(data_set[2:4, 0], data_set[2:4, 1], c='b', label='B')
for i in range(4):
    ax.text(data_set[i][0]-0.04, data_set[i][1]-0.01, labels[i])
ax.scatter([0.2], [0.3], c='b')
ax.text(0.21, 0.29, '<-- Data classfied here')
ax.legend(loc='upper left')
plt.show()
