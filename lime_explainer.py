import argparse
from pathlib import Path
from typing import List, Any

import numpy as np
from lime.lime_text import LimeTextExplainer
import sklearn.pipeline
import spacy


METHODS = {
    'logistic': {
        'class': "LogisticExplainer",
        'file': "data/sst/sst_train.txt"
    },
    'svm': {
        'class': "SVMExplainer",
        'file': "data/sst/sst_train.txt"
    },
    'fasttext': {
        'class': "FastTextExplainer",
        'file': "models/fasttext/sst-5.ftz"
    },
}


def tokenizer(text):
    "Tokenize input string using a spaCy pipeline"
    nlp = spacy.blank('en')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))  # Very basic NLP pipeline in spaCy
    doc = nlp(text)
    tokenized_text = ' '.join(token.text for token in doc)
    return tokenized_text


def explainer_class(method: str, filename: str) -> Any:
    "Instantiate class using its string name"
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_(filename)


class LogisticExplainer:
    """Class to explain classification results of a scikit-learn
       Logistic Regression Pipeline. The model is trained within this class.
    """
    def __init__(self, path_to_train_data: str) -> None:
        "Input training data path for training Logistic Regression classifier"
        import pandas as pd
        # Read in training data set
        self.train_df = pd.read_csv(path_to_train_data, sep='\t', header=None, names=["truth", "text"])
        self.train_df['truth'] = self.train_df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        self.train_df['truth'] = self.train_df['truth'].astype(int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        "Create sklearn logistic regression model pipeline"
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
            ]
        )
        # Train model
        classifier = pipeline.fit(self.train_df['text'], self.train_df['truth'])
        return classifier

    def predict(self, texts: List[str]) -> np.array([float, ...]):
        """Generate an array of predicted scores (probabilities) from sklearn
        Logistic Regression Pipeline."""
        classifier = self.train()
        probs = classifier.predict_proba(texts)
        return probs


class SVMExplainer:
    """Class to explain classification results of a scikit-learn linear Support Vector Machine
       (SVM) Pipeline. The model is trained within this class.
    """
    def __init__(self, path_to_train_data: str) -> None:
        "Input training data path for training Logistic Regression classifier"
        import pandas as pd
        # Read in training data set
        self.train_df = pd.read_csv(path_to_train_data, sep='\t', header=None, names=["truth", "text"])
        self.train_df['truth'] = self.train_df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        self.train_df['truth'] = self.train_df['truth'].astype(int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        "Create sklearn logistic regression model pipeline"
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='modified_huber',
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    tol=None,
                )),
            ]
        )
        # Train model
        classifier = pipeline.fit(self.train_df['text'], self.train_df['truth'])
        return classifier

    def predict(self, texts: List[str]) -> np.array([float, ...]):
        """Generate an array of predicted scores (probabilities) from sklearn
        Logistic Regression Pipeline."""
        classifier = self.train()
        probs = classifier.predict_proba(texts)
        return probs


class FastTextExplainer:
    """Class to explain classification results of FastText.
       Assumes that we already have a trained FastText model with which to make predictions.
    """
    def __init__(self, path_to_model: str) -> None:
        "Input fastText trained sentiment model"
        import fasttext
        self.classifier = fasttext.load_model(path_to_model)

    def predict(self, texts: List[str]) -> np.array([float, ...]):
        "Generate an array of predicted scores using the FastText"
        labels, probs = self.classifier.predict(texts, 5)

        # For each prediction, sort the probability scores in the same order for all texts
        result = []
        for label, prob, text in zip(labels, probs, texts):
            order = np.argsort(np.array(label))
            result.append(prob[order])
        return np.array(result)


def explainer(method: str,
              path_to_file: str,
              text: str) -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""

    model = explainer_class(method, path_to_file)
    predictor = model.predict

    # Create a LimeTextExplainer
    explainer = LimeTextExplainer(
        # Specify split option
        split_expression=lambda x: x.split(),
        # Our classifer uses trigrams or contextual ordering to classify text
        # Hence, order matters, and we cannot use bag of words.
        bow=False,
        # Specify class names for this case
        class_names=[1, 2, 3, 4, 5]
    )

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        text,
        classifier_fn=predictor,
        top_labels=1,
        num_features=20,
    )
    return exp


def main(samples: List[str]) -> None:
    # Get list of available methods:
    method_list = [method for method in METHODS.keys()]
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, nargs='+', help="Enter one or more methods \
                        (Choose from following: {})".format(", ".join(method_list)),
                        required=True)

    args = parser.parse_args()

    for method in args.method:
        if method not in METHODS.keys():
            parser.error("Please choose from the below existing methods! \n{}".format(", ".join(method_list)))
        path_to_file = METHODS[method]['file']
        # Run explainer function
        print("Method: {}".format(method.upper()))
        for i, text in enumerate(samples):
            text = tokenizer(text)  # Tokenize text using spaCy before explaining
            print("Generating LIME explanation for example {}: `{}`".format(i+1, text))
            exp = explainer(method, path_to_file, text)
            # Output to HTML
            output_filename = Path(__file__).parent / "{}-explanation-{}.html".format(i+1, method)
            exp.save_to_file(output_filename)


if __name__ == "__main__":
    # Evaluation text
    samples = [
        "It 's not horrible , just horribly mediocre .",
        "The cast is uniformly excellent ... but the film itself is merely mildly charming .",
    ]
    main(samples)
