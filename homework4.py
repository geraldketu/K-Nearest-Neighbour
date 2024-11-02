import numpy as np
import matplotlib.pyplot as plt

# Load datasets
train_data = np.loadtxt('train.txt')
test_data = np.loadtxt('test.txt')

# Split the data into features and labels
train_X, train_y = train_data[:, :2], train_data[:, 2]
test_X, test_y = test_data[:, :2], test_data[:, 2]

# Function to plot data
def plot_data(X, y, title):
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    plt.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()

# Plot training data
plot_data(train_X, train_y, 'Training Data')

# Plot test data
plot_data(test_X, test_y, 'Test Data')
