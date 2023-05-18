from dataset import df
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Split the dataset into ten equal-sized folds
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the classifiers with desired parameters
svm_classifier = SVC(max_iter=500)
decision_tree_classifier = DecisionTreeClassifier()
mlp_classifier = MLPClassifier(max_iter=500)
nb_classifier = GaussianNB()
rf_classifier = RandomForestClassifier()
gb_classifier = GradientBoostingClassifier()
knn_classifier = KNeighborsClassifier()

# Define a dictionary to store the performance metrics for each algorithm
performance_metrics = {
    'SVM': [],
    'Decision Tree': [],
    'MLP': [],
    'Naive Bayes': [],
    'Random Forest': [],
    'Gradient Boosting': [],
    'K-Nearest Neighbors': []
}

# Iterate over the ten folds and evaluate each algorithm
for train_index, val_index in kfold.split(df[['Number']], df['Label']):
    X_train, X_val = df['Number'].iloc[train_index].values.reshape(-1, 1), df['Number'].iloc[val_index].values.reshape(-1, 1)
    y_train, y_val = df['Label'].iloc[train_index], df['Label'].iloc[val_index]

    # Train and evaluate SVM classifier
    svm_classifier.fit(X_train, y_train)
    svm_predictions = svm_classifier.predict(X_val)
    svm_accuracy = accuracy_score(y_val, svm_predictions)
    performance_metrics['SVM'].append(svm_accuracy)

    # Train and evaluate Decision Tree classifier
    decision_tree_classifier.fit(X_train, y_train)
    dt_predictions = decision_tree_classifier.predict(X_val)
    dt_accuracy = accuracy_score(y_val, dt_predictions)
    performance_metrics['Decision Tree'].append(dt_accuracy)

    # Train and evaluate MLP classifier
    mlp_classifier.fit(X_train, y_train)
    mlp_predictions = mlp_classifier.predict(X_val)
    mlp_accuracy = accuracy_score(y_val, mlp_predictions)
    performance_metrics['MLP'].append(mlp_accuracy)

    # Train and evaluate Naive Bayes classifier
    nb_classifier.fit(X_train, y_train)
    nb_predictions = nb_classifier.predict(X_val)
    nb_accuracy = accuracy_score(y_val, nb_predictions)
    performance_metrics['Naive Bayes'].append(nb_accuracy)

    # Train and evaluate Random Forest classifier
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_val)
    rf_accuracy = accuracy_score(y_val, rf_predictions)
    performance_metrics['Random Forest'].append(rf_accuracy)

    # Train and evaluate Gradient Boosting classifier
    gb_classifier.fit(X_train, y_train)
    gb_predictions = gb_classifier.predict(X_val)
    gb_accuracy = accuracy_score(y_val, gb_predictions)
    performance_metrics['Gradient Boosting'].append(gb_accuracy)

    # Train and evaluate K-Nearest Neighbors classifier
    knn_classifier.fit(X_train, y_train)
    knn_predictions = knn_classifier.predict(X_val)
    knn_accuracy = accuracy_score(y_val, knn_predictions)
    performance_metrics['K-Nearest Neighbors'].append(knn_accuracy)

# Calculate the average performance metrics across the ten folds for each algorithm
average_metrics = {
    'SVM': np.mean(performance_metrics['SVM']),
    'Decision Tree': np.mean(performance_metrics['Decision Tree']),
    'MLP': np.mean(performance_metrics['MLP']),
    'Naive Bayes': np.mean(performance_metrics['Naive Bayes']),
    'Random Forest': np.mean(performance_metrics['Random Forest']),
    'Gradient Boosting': np.mean(performance_metrics['Gradient Boosting']),
    'K-Nearest Neighbors': np.mean(performance_metrics['K-Nearest Neighbors'])
}

#Compare the performance metrics and select the best algorithm
best_algorithm = max(average_metrics, key=average_metrics.get)

print("Performance Metrics:")
for algorithm, metrics in average_metrics.items():
    print(f"{algorithm}: {metrics}")

print("Best Algorithm:", best_algorithm)

