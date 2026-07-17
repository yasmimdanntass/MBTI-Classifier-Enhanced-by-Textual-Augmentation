from pipeline.tfidf import TFIDFTokenizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


class MultinomialNaiveBayesClassifier:

    def __init__(self):
        self.tokenizer = TFIDFTokenizer()
        self.model = MultinomialNB()

    def prepare_data(self, dataframe, target_column):

        columns = ["posts", target_column]

        for column in columns:
            if column not in dataframe.columns:
                raise ValueError(
                    f"A coluna '{column}' não existe no DataFrame."
                )

        cleaned_dataframe = dataframe.dropna(subset = columns).copy()

        cleaned_dataframe = cleaned_dataframe[cleaned_dataframe["posts"].str.strip() != ""]

        x = cleaned_dataframe["posts"]
        y = cleaned_dataframe[target_column]

        return x, y

    def train(self, dataframe, target_column):
        x, y = self.prepare_data(dataframe, target_column)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size = 0.2,
            random_state = 42,
            stratify = y
        )

        x_train_tfidf = self.tokenizer.fit_transform(x_train)

        self.model.fit(x_train_tfidf, y_train)

        return x_test, y_test

    def predict(self, texts):
        x_tfidf = self.tokenizer.transform(texts)

        return self.model.predict(x_tfidf)

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average = "macro",
            zero_division = 0
        )

        recall = recall_score(
            y_test,
            predictions,
            average = "macro",
            zero_division = 0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average = "macro",
            zero_division = 0
        )

        report = classification_report(
            y_test,
            predictions,
            zero_division = 0
        )

        return accuracy, precision, recall, f1, report