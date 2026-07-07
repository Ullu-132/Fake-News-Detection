import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from xgboost import plot_importance

def load_and_preprocess_data(true_path, fake_path):
    true_news = pd.read_csv(true_path)
    fake_news = pd.read_csv(fake_path)

    true_news['label'] = 0
    fake_news['label'] = 1

    combined_news = pd.concat([true_news, fake_news], ignore_index=True)
    df = combined_news.sample(frac=1, random_state=42).reset_index(drop=True)

    df.fillna(' ', inplace=True)
    df.drop_duplicates(inplace=True)

    return df

def create_features_and_labels(df, text_cols):
    X = df[text_cols].agg(' '.join, axis=1)
    y = df['label']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test, stop_words='english', max_df=0.7):
    """Transforms text data into TF-IDF vectors."""
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=max_df)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    return tfidf_train, tfidf_test, tfidf_vectorizer

def evaluate_model(xgb_classifier, tfidf_test, y_test):
    predictions = xgb_classifier.predict(tfidf_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("Accuracy: ", accuracy)
    print("Classification Report: \n", report)
    return accuracy, report

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def _calculate_gini_impurity(self, y):
        if len(y)==0:
            return 0
        p = np.bincount(y.astype(int))/len(y)
        return 1-np.sum(p**2)

    def _find_best_split(self, X, y):
        best_gini = np.inf
        best_split = None
        n = X.shape[1]  #number of columns

        for feature in range(n):
            values = np.unique(X[:, feature])
            # Get all unique values of the current feature (to use as possible thresholds for splitting).
            # This avoids trying redundant thresholds.

            for threshold in values:
                #Split data
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                #Ensure both sides have data
                if(np.sum(left_mask)==0 or np.sum(right_mask)==0):
                    continue

                #Calculate weighted Gini impurity
                gini_left = self._calculate_gini_impurity(y[left_mask])
                gini_right = self._calculate_gini_impurity(y[right_mask])
                gini = (np.sum(left_mask)/len(y)) * gini_left + (np.sum(right_mask)/len(y)) * gini_right

                #Update best split if Gini impurity is lower
                if gini<best_gini:
                    best_gini = gini
                    best_split = (feature, threshold)

        return best_split

    def _build_tree(self, X, y, depth):
        if depth>=self.max_depth or len(np.unique(y))==1:
            return np.bincount(y.astype(int)).argmax()  #Leaf Node: return the majority class for classification

        split = self._find_best_split(X, y)

        if split is None:
            return np.bincount(y.astype(int)).argmax()  #If no split is possible, return the majority class
        
        #Split data:
        best_feature, threshold = split
        left_mask = X[:, best_feature] <= threshold
        right_mask = X[:, best_feature] > threshold

        left_child = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth+1)

        return (best_feature, threshold, left_child, right_child)

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)
        #Fits the decision tree to the data

    def predict_single(self, x, tree):
        # predicts the label for a single instance
        if not isinstance(tree, tuple):  #Base Case -> If the current "node" is not a tuple, it's a leaf node, and its value is the predicted result.
            return tree  #Leaf Node

        feature, threshold, left_child, right_child = tree

        if x[feature] <= threshold:
            return self.predict_single(x, left_child)
        else:
            return self.predict_single(x, right_child)

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])

class XGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Fits the gradient boosting model to the data
        predictions = np.zeros(len(y))

        for _ in range(self.n_estimators):
            residuals = y-predictions  #residuals -> negative gradient  #error = y_predicted - y_actual

            tree = DecisionTree(max_depth = self.max_depth)
            tree.fit(X, residuals)

            update = self.learning_rate * tree.predict(X)
            predictions += update

            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return np.round(predictions)  #round to get binary predictions

if __name__=="__main__":
    df = load_and_preprocess_data("True.csv", "Fake.csv")
    text_cols = ['title', 'text', 'subject', 'date']
    X, y = create_features_and_labels(df, text_cols)

    #Split data into training and testing sets
    # train_size = int(0.8*len(X))
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test, y_train, y_test = split_data(X, y)

    tfidf_train, tfidf_test, tfidf_vectorizer = vectorize_text(X_train, X_test)

    model = XGBoost(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(tfidf_train, y_train)

    predictions = model.predict(X_test)

    accuracy = np.mean(predictions == y_test)
    print("Accuracy: ", accuracy)
