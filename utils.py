import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(feature_importance):
    # Extract feature names and importances
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())

    plt.figure(figsize=(10, 6))
    plt.bar(features, importances)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(' Absolute Feature Importance')
    plt.show()
