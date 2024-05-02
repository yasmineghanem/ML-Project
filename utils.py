import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, GridSearchCV

def getGridSearch(estimator, parametersGrid, X, y, scoring, crossValidation=10):
    # Grid Search
    grid_search = GridSearchCV(
        estimator, parametersGrid, cv=crossValidation, scoring=scoring, return_train_score=True)
    grid_search.fit(X, y)
    return grid_search


def get_Learning_Curve(estimator, X, y,modelName, crossValidation=5, scoring='f1_weighted'):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=crossValidation,
                                                            train_sizes=np.linspace(.1, 1.0, 5),
                                                            scoring=scoring, shuffle=True, random_state=42)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title("Learning Curve for {}".format(modelName))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc='best')
    plt.show()