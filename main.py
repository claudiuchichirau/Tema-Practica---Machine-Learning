import string
import os
import math
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from email import policy
from email.parser import BytesParser

def load_data(base_directory, folders):
    emails = []
    labels = []

    for folder in folders:
        folder_path = os.path.join(base_directory, folder).replace("\\", "/")

        print("Loading emails from folder:", folder_path)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
                emails.append(msg.get_body(preferencelist=('plain')).get_content())

            if "spm" in filename:
                labels.append(1)  # Spam
            else:
                labels.append(0)  # Non-spam

    df = pd.DataFrame({
        'email': emails,
        'label': labels
    })

    return df

def tokenize(text):
    return nltk.word_tokenize(text)

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.word_counts = None
        self.class_counts = None
        self.vocab = None

    def fit(self, X, y):
        self.classes = set(y)
        self.word_counts = {cls: {} for cls in self.classes}
        self.class_counts = {cls: 0 for cls in self.classes}
        self.vocab = set()

        for tokens, cls in zip(X, y):
            self.class_counts[cls] += 1
            for token in tokens:
                if token not in self.word_counts[cls]:
                    self.word_counts[cls][token] = 0
                self.word_counts[cls][token] += 1
                self.vocab.add(token)

    def predict(self, X, laplace_smooth=1):
        predictions = []
        for tokens in X:
            log_probs = {}
            for cls in self.classes:
                # Calculați log-probabilitățile pentru toate tokenurile simultan
                token_counts = np.array([self.word_counts[cls].get(token, 0) for token in tokens])
                log_probs[cls] = np.log(self.class_counts[cls]) + np.sum(np.log((token_counts + laplace_smooth) / (sum(self.word_counts[cls].values()) +  len(self.vocab) * laplace_smooth)))
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions

def leave_one_out(X, y):
    n = len(X)
    for i in range(n):
        train_X = np.concatenate((X[:i], X[i+1:]))
        train_y = np.concatenate((y[:i], y[i+1:]))
        test_X = np.array([X[i]])
        test_y = np.array([y[i]])
        yield train_X, train_y, test_X, test_y

def LOO_main(train_data):
    accuracies = []
    train_accuracies = []

    X = np.array(train_data["email_tokens"])
    y = np.array(train_data["label"])

    for train_X, train_y, test_X, test_y in leave_one_out(X, y):
        model = NaiveBayes()
        model.fit(train_X, train_y)
        
        # Predict on the training set
        train_predictions = model.predict(train_X)
        train_accuracy = sum(train_predictions == train_y) / len(train_y)
        print(f'Train Accuracy: {train_accuracy}')
        train_accuracies.append(train_accuracy)
        
        # Predict on the test set
        test_predictions = model.predict(test_X)
        test_accuracy = sum(test_predictions == test_y) / len(test_y)
        print("Test Accuracy:", test_accuracy)
        accuracies.append(test_accuracy)

    print("Average train accuracy:", np.mean(train_accuracies))
    print("Average test accuracy:", np.mean(accuracies))

    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(accuracies, label='Test Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Leave-One-Out Cross-Validation')
    plt.legend()
    plt.show()

def Naive_Bayes_main(train_data, test_data):
    # Antrenarea modelului și calcularea acurateții
    accuracies = []
    for laplace_smooth in [0.1, 5.0, 10.0, 100.0]:
        model = NaiveBayes()
        model.fit(train_data["email_tokens"], train_data["label"])

        # training set
        train_predictions = model.predict(train_data["email_tokens"], laplace_smooth)
        train_accuracy = sum(train_predictions == train_data["label"]) / len(train_data)
        print(f'Train Accuracy for laplace_smooth = {laplace_smooth}: {train_accuracy}')
        
        # test set
        test_predictions = model.predict(test_data["email_tokens"], laplace_smooth)
        test_accuracy = sum(test_predictions == test_data["label"]) / len(test_data)
        print(f'Test Accuracy for laplace_smooth = {laplace_smooth}: {test_accuracy}')
        
        accuracies.append(test_accuracy)

    # Generarea graficului
    plt.plot([0.1, 1.0, 10.0, 100.0], accuracies)
    plt.xlabel('Laplace Smoothing Parameter')
    plt.ylabel('Accuracy')
    plt.title('Model Performance')
    plt.show()

def self_training(train_data, unlabeled_data, test_data):
    model = NaiveBayes()
    cvloo_accuracies = []
    test_accuracies = []

    print("train_data:", len(train_data))

    for i in range(len(train_data)):
        print("\tIteration:", i)
        # Implement CVLOO
        train = train_data.drop(i)
        test = train_data.iloc[i]
        model.fit(train["email_tokens"], train["label"])
        prediction = model.predict([test["email_tokens"]])
        cvloo_accuracies.append(prediction[0] == test["label"])

        # Predict labels for the unlabeled data
        predictions = model.predict(unlabeled_data["email_tokens"])

        # Add the newly labeled data to the training data
        new_train_data = pd.concat([train_data, pd.DataFrame({"email_tokens": unlabeled_data["email_tokens"], "label": predictions})])

        # Retrain the model on the new training data
        model.fit(new_train_data["email_tokens"], new_train_data["label"])

        # Calculate accuracy on the test set
        test_predictions = model.predict(test_data["email_tokens"])
        test_accuracy = sum(test_predictions == test_data["label"]) / len(test_data["label"])
        test_accuracies.append(test_accuracy)

    return model, cvloo_accuracies, test_accuracies

def Self_training_main(train_data, unlabeled_data, test_data):
    # Train the model with self-training
    model, cvloo_accuracies, test_accuracies = self_training(train_data, unlabeled_data, test_data)

    # Generate the plots
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_data)), cvloo_accuracies, label='CVLOO Accuracy')
    plt.plot(range(len(train_data)), test_accuracies, label='Test Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    # train_data = load_data("data", ["part1", "part2", "part3", "part4", "part5", "part6", "part7", "part8", "part9"])
    train_data = load_data("data", ["part1", "part2", "part3", "part4"])
    test_data = load_data("data", ["part10"])
    unlabeled_data = load_data("data", ["part1", "part2"]) 

    # Curățarea datelor
    train_data["email"] = train_data["email"].str.lower()
    train_data["email"] = train_data["email"].str.translate(str.maketrans('', '', string.punctuation))

    test_data["email"] = test_data["email"].str.lower()
    test_data["email"] = test_data["email"].str.translate(str.maketrans('', '', string.punctuation))

    unlabeled_data["email"] = unlabeled_data["email"].str.lower()
    unlabeled_data["email"] = unlabeled_data["email"].str.translate(str.maketrans('', '', string.punctuation))

    # Tokenizarea
    train_data["email_tokens"] = train_data["email"].apply(tokenize)
    test_data["email_tokens"] = test_data["email"].apply(tokenize)
    unlabeled_data["email_tokens"] = test_data["email"].apply(tokenize)
    
    # Eliminarea cuvintelor de oprire
    stop_words = set(stopwords.words('english'))
    train_data["email_tokens"] = train_data["email_tokens"].apply(lambda tokens: [token for token in tokens if token not in stop_words])
    test_data["email_tokens"] = test_data["email_tokens"].apply(lambda tokens: [token for token in tokens if token not in stop_words])
    unlabeled_data["email_tokens"] = unlabeled_data["email_tokens"].apply(lambda tokens: [token for token in tokens if token not in stop_words] if isinstance(tokens, list) else [])

    # Stemming
    stemmer = PorterStemmer()
    train_data["email_tokens"] = train_data["email_tokens"].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
    test_data["email_tokens"] = test_data["email_tokens"].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
    unlabeled_data["email_tokens"] = unlabeled_data["email_tokens"].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

    # Naive_Bayes_main(train_data, test_data)

    LOO_main(train_data)

    # Self_training_main(train_data, unlabeled_data, test_data)

if __name__ == "__main__":
    main()
