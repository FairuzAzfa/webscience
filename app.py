import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter  # Import Counter to fix earlier issue

# Define custom DecisionTreeModel from notebook
class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def train(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, best_feature] < best_threshold
        right_idx = ~left_idx

        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def _find_best_split(self, X, y):
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] < threshold
                right_idx = ~left_idx
                gini = self._gini_index(y[left_idx], y[right_idx])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y, right_y):
        def gini(y):
            if len(y) == 0:
                return 0
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)

        total_samples = len(left_y) + len(right_y)
        return (len(left_y) / total_samples) * gini(left_y) + (len(right_y) / total_samples) * gini(right_y)

    def predict(self, X):
        return np.array([self._predict_one(sample, self.tree) for sample in X])

    def _predict_one(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        if sample[tree["feature"]] < tree["threshold"]:
            return self._predict_one(sample, tree["left"])
        else:
            return self._predict_one(sample, tree["right"])


# Streamlit UI
st.title("Streamlit Decision Tree Prediction App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset:")
    st.write(data.head())

    # Features and Target Selection
    target_column = st.selectbox("Select Target Column", data.columns)
    feature_columns = st.multiselect(
        "Select Feature Columns", data.columns, default=[col for col in data.columns if col != target_column]
    )

    if target_column and feature_columns:
        X = data[feature_columns].values
        y = data[target_column].values

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        max_depth = st.slider("Select Max Depth for Decision Tree", min_value=1, max_value=20, value=5)
        model = DecisionTreeModel(max_depth=max_depth)

        # Convert y_train to integer type
        y_train = y_train.astype(int)

        model.train(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Ensure y_test and y_pred have the same type
        y_test = y_test.astype(int)
        y_pred = np.array(y_pred).astype(int)

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write("### Model Evaluation")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        st.write("### Confusion Matrix")
        st.write(conf_matrix)
