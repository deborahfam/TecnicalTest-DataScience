import streamlit as st
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.markdown("# Problem Description")
st.markdown("""
Write a program that given the numbers from 1 to 100 print “None” for each number. But for multiples of three print “Fizz” instead of “None” and for the multiples of five print “Buzz”. For numbers which are multiples of both three and five print “FizzBuzz”. Perform a ten-fold cross-validation using different classification algorithms and select the best among them.
""")

#Algorithm Descriptions
st.markdown("## Algorithm Descriptions")

#Decision Trees
st.markdown("### Decision Trees")
st.markdown("""
A Decision Tree is a supervised learning algorithm that is commonly used for both classification and regression tasks. It creates a flowchart-like tree structure, where each internal node represents a feature or attribute, each branch represents a decision based on that attribute, and each leaf node represents the outcome or prediction. The goal of the algorithm is to create a tree that can make accurate predictions on unseen data.

The decision-making process in a Decision Tree involves splitting the dataset based on the values of different features. The splitting criterion is typically determined using metrics like Gini impurity or information gain, which measure the homogeneity or purity of the target variable within each branch. The algorithm recursively partitions the data into smaller subsets based on the selected features until it reaches a stopping condition, such as reaching a maximum depth, having a minimum number of samples in a leaf node, or achieving a specific level of purity.
""")

#MLP
st.markdown("### MLP")
st.markdown("""
MLP is a type of feedforward artificial neural network that consists of multiple layers of nodes (neurons) connected in a directed acyclic graph. It is a powerful and flexible algorithm used for both classification and regression tasks. Each node in the MLP performs a weighted sum of its inputs, applies an activation function to produce an output, and passes it to the nodes in the next layer. The MLP learns to adjust the weights on the connections between nodes to minimize the error between the predicted and actual outputs.
""")

#Support Vector Machine (SVM)
st.markdown("## Support Vector Machine (SVM)")
st.markdown("""
Support Vector Machines (SVMs) are powerful and versatile supervised learning algorithms used for classification and regression tasks. The fundamental principle behind SVMs is to find an optimal hyperplane that separates the data points belonging to different classes with the largest margin.
""")


# Load the dataset
df = pd.read_csv('dataset.csv')

# Display the initial dataset
st.subheader("Initial Dataset")
st.dataframe(df)

# Split the dataset into features and labels
X = df[['Number']]
y = df['Label']

# Define the algorithms to evaluate
algorithms = {
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'MLP': MLPClassifier()
}

# Perform 10-fold cross-validation for each algorithm
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize a dataframe to store the results
results_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Iterate over the algorithms and perform cross-validation
for algorithm_name, algorithm in algorithms.items():
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Perform cross-validation
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the algorithm
        algorithm.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = algorithm.predict(X_test)

        # Calculate evaluation metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate average metrics
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    # Append the results to the dataframe
    results_df = results_df.append({
        'Algorithm': algorithm_name,
        'Accuracy': average_accuracy,
        'Precision': average_precision,
        'Recall': average_recall,
        'F1-Score': average_f1
    }, ignore_index=True)

# Display the results as a table
st.subheader("Cross-Validation Results")
st.dataframe(results_df)

# Calculate and display the confusion matrix for the best algorithm
best_algorithm = results_df.loc[results_df['Accuracy'].idxmax()]['Algorithm']
best_algorithm_model = algorithms[best_algorithm]
best_algorithm_model.fit(X, y)
y_pred = best_algorithm_model.predict(X)
confusion = confusion_matrix(y, y_pred)

st.subheader("Confusion Matrix")
st.dataframe(pd.DataFrame(confusion, index=best_algorithm_model.classes_, columns=best_algorithm_model.classes_))

# Display the classification report for the best algorithm
classification_rep = classification_report(y, y_pred, target_names=best_algorithm_model.classes_)
st.subheader("Classification Report")
st.text(classification_rep)


