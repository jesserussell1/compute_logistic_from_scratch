import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(12)
num_observations = 5000

# Generate three sets of features
x1 = np.random.multivariate_normal([0, 0, 0], [[1, 0.75, 0.5], [0.75, 1, 0.25], [0.5, 0.25, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4, 2], [[1, 0.75, 0.5], [0.75, 1, 0.25], [0.5, 0.25, 1]], num_observations)
x3 = np.random.multivariate_normal([2, 2, 2], [[1, 0.75, 0.5], [0.75, 1, 0.25], [0.5, 0.25, 1]], num_observations)

# Combine features
features = np.vstack((x1, x2, x3)).astype(np.float32)

# Add labels for each observation
labels = np.hstack((np.zeros(num_observations), np.ones(num_observations), np.ones(num_observations)))


# Function to apply sigmoid function
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

# Function to calculate log-likelihood
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    loglike = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return loglike

# Function to compute gradient
def compute_gradient(features, target, predictions):
    output_error_signal = target - predictions
    gradient = np.dot(features.T, output_error_signal)
    return gradient

# Function to update weights
def update_weights(weights, gradient, learning_rate):
    return weights + learning_rate * gradient

# Function to apply logistic regression
def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        # Add a column of ones
        intercept = np.ones((features.shape[0], 1))
        # Stack the intercept to features
        features = np.hstack((intercept, features))

    # Initialize weights
    weights = np.zeros(features.shape[1])

    # Apply logistic regression algorithm
    for step in range(num_steps):
        # Perform a matrix multiplication (dot product) between the features matrix and the weights vector
        scores = np.dot(features, weights)
        # Apply the sigmoid function to the calculated scores
        predictions = sigmoid(scores)

        # Calculate the gradient of the log-likelihood function with respect to the weights
        gradient = compute_gradient(features, target, predictions)

        # Update the weights based on the gradient just calculated
        weights = update_weights(weights, gradient, learning_rate)

        # Print log-likelihood every 1000 steps just to watch progress
        if step % 1000 == 0:
            print(log_likelihood(features, target, weights))

    # Return weights, which represent the (intercept and) coefficients
    return weights

# Call the logistic regression function with the features and the labels
weights = logistic_regression(features, labels,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)

# Print the results of the logistic regression
print(weights)


# Check the results against the sklearn package results
clf = LogisticRegression(fit_intercept=True, C = 1e15)
clf.fit(features, labels)
print (clf.intercept_, clf.coef_)