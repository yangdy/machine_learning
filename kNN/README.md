# kNN notes

- Props: High accuracy, insensitive to outliers, no assumptions about data
- Cons: Computationlly exquires a lot of memory
- Works with: Numeric values, nominal values

# Algorithm

    For every point in dataset:
        calculate the distance between inX and the current point
        sort the distance in increasing order
        take k item with lowest distance to inX
        find the majority class among these items
        return the majority class as out prediction for the class of inX

# Appendix

## 1. Extend data set

Using numpy.vstack((a, b)) method to add classfied data to dataset.

    data_set = numpy.array([[1, 1], [1, 2]])
    data_set = numpy.vstack((data_set, [2, 2]))
    ...
    [[1, 1],
     [1, 2],
     [2, 2]]

Or using numpy.concatenate((a, b)).
