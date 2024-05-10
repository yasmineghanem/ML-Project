from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracies(train_accuracies, test_accuracies, n_estimators_options, train_label='Train Accuracy'):
    plt.plot(n_estimators_options, test_accuracies, label='Test Accuracy')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.show()
    plt.plot(n_estimators_options, train_accuracies,
             label=train_label, color='orange')
    plt.xlabel('Number of Estimators')
    plt.ylabel(train_label)
    plt.legend()
    plt.show()


class GenericBoosting:

    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def create_estimator(self):
        # create a decision stump as a weak estimator
        return DecisionTreeClassifier(max_depth=1, random_state=0, class_weight='balanced')

    def fit_and_predict(self, X_train, Y_train, X_test, Y_test, weight_multiplier=1.0):
        # apply AdaBoost on weak estimators

        # convert the labels to -1 and 1 to get the effect of the wieghts that are missclassified
        # if a classifier missclassifies most of the sample then it will have a negative weight
        # this means that it has a higher weight in value but we take the opposite of the prediction
        Y_train = np.where(Y_train == 0, -1, 1)
        Y_test = np.where(Y_test == 0, -1, 1)

        # initialize the training and test data with empty array placeholders
        pred_train = np.empty((self.n_estimators, X_train.shape[0]))
        pred_test = np.empty((self.n_estimators, X_test.shape[0]))

        # initialize weights
        W = np.ones((X_train.shape[0],)) / X_train.shape[0]

        # loop over the boosting iterations
        for idx in range(self.n_estimators):

            # create and fit a new decision stump
            model = self.create_estimator().fit(X_train, Y_train, sample_weight=W)

            # predict classes for the training data and test data
            pred_train_idx = model.predict(X_train)
            pred_test_idx = model.predict(X_test)

            # TODO: calculate the miss Indicator
            # what is the miss indicator??
            # get the indices where the prediction is not equal to the actual label => test of training?
            miss_indicator = np.where(pred_train_idx != Y_train)[0]

            # Increase the weights of misclassified points
            miss_indicator_weights = np.ones_like(W)

            # Adjust the weight multiplier as needed
            miss_indicator_weights[miss_indicator] *= weight_multiplier

            # TODO: calculate the error for the current classifier
            # iterate over the dataset and calculate the error
            # error = sum of the weights of the misclassified samples
            cls_err = np.sum(W[miss_indicator])

            # TODO: calculate current classifier weight
            cls_alpha = 0.5 * np.log((1 - cls_err) / cls_err)

            # TODO: update the weights
            # W = W * np.exp(-cls_alpha * Y_train * pred_train_idx)
            # Update the weights with more weight given to misclassified points
            W = W * np.exp(-cls_alpha * Y_train * pred_train_idx * miss_indicator_weights)

            # TODO: add to the overall predictions
            # the weighted classifier
            pred_train[idx] = cls_alpha * pred_train_idx
            pred_test[idx] = cls_alpha * pred_test_idx

            # normalize weights
            # try using balanced wights 
            # we divide the weights of each class by the sum of the weights in the class
            W = W / np.sum(W)

        # TODO: return accuracy on train and test sets
        # for all the estimators
        # the final model for the adaboost is the sum of the predictions of all the classifiers
        train_accuracy = np.mean(
            np.sign(np.sum(pred_train, axis=0)) == Y_train)
        test_accuracy = np.mean(np.sign(np.sum(pred_test, axis=0)) == Y_test)

        final_pred_train = np.sign(np.sum(pred_train, axis=0))
        final_pred_test = np.sign(np.sum(pred_test, axis=0))

        # return the classes to 0 and 1
        final_pred_train[final_pred_train < 0] = 0
        final_pred_test[final_pred_test < 0] = 0

        return train_accuracy, test_accuracy, final_pred_train, final_pred_test


def get_scores(n_estimators, X_train, y_train, X_test, y_test, weight_multiplier=1.0):
    # run model boosting and compute train and test accuracy
    model = GenericBoosting(n_estimators=n_estimators)
    train_accuracy, test_accuracy, predicted_train, predicted_test = model.fit_and_predict(
        X_train, y_train, X_test, y_test, weight_multiplier)

    return train_accuracy, test_accuracy, predicted_train, predicted_test


def run_boosting(X_train, y_train, X_test, y_test):
    n_estimators_options = [5, 10, 50, 100, 200, 500]
    train_accuracies = []
    test_accuracies = []
    for n_estimators in n_estimators_options:
        train_accuracy, test_accuracy, predicted_train, predicted_test = get_scores(
            n_estimators, X_train, y_train, X_test, y_test)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    return train_accuracies, test_accuracies, n_estimators_options
    # # TODO: plot the output scores against n_estimators
    # plot_accuracies(train_accuracies, test_accuracies, n_estimators_options)
