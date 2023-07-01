import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def concat_content(dict, content, key):
    if key in dict.keys():
        content += str(dict[key])
        content += '. '
    return content

def getReviewContent(review_dict):
    content = ''
    # concat summary, bodytext, reviewerName and unixreviewtime of review together
    content = concat_content(review_dict,content,'summary')
    content = concat_content(review_dict,content,'reviewText')
    content = concat_content(review_dict,content,'unixReviewTime')
    content = concat_content(review_dict,content,'reviewerName')
    return content

def get15BestFeatures(vectorizer, X_t ,y_t):
    # nothing much to elaborate
    words = vectorizer.get_feature_names_out()
    words = np.array(words)

    skb = SelectKBest(k=15)
    skb.fit(X_t, y_t)
    best15 = skb.get_support()

    return words[best15]

def extract_features(data, label, source):
    # address only the verified reviews, since unverified could be biased
    if source['verified']:
        data.append(getReviewContent(source))
        label.append(source['overall'])


def read_data(file):
    # iterate over entries from file and extract
    # chosen features
    data = []
    labels = []
    with open(file,'r') as file_lines:
        for line in file_lines:
            review = json.loads(line)
            extract_features(data, labels, review)

    return data, labels



def classify(train_file, test_file):
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')
    # Read in the train and test data
    train_data, train_labels = read_data(train_file)
    test_data, test_labels = read_data(test_file)

    # Convert the text data into numerical feature vectors
    K = 978
    vectorizer = CountVectorizer(ngram_range=(1,1),max_features=K)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # refine with Tfidf transformer
    transformer = TfidfTransformer()
    train_vectors = transformer.fit_transform(train_vectors)
    test_vectors = transformer.transform(test_vectors)

    # Train a classifier using the training data
    classifier = LogReg(max_iter=1000)
    classifier.fit(train_vectors, train_labels)

    # Make predictions on the test data
    predictions = classifier.predict(test_vectors)

    # Confusion matrix = all VS all (true+,true-,false+,false-)
    conf_mat = confusion_matrix(test_labels, predictions, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
    print('\n----Confusion Matrix:----\n')
    print(conf_mat)
    print('\n----Best 15 Features----\n')
    print(get15BestFeatures(vectorizer, train_vectors, train_labels))
    print('\n')

    # Calculate the accuracy of the predictions and F1 score for each class
    # and return the results as a dictionary
    # F1 = one VS rest

    return {
            'class_1_F1': f1_score(test_labels, predictions, labels=[1.0], average='micro'),
            'class_2_F1': f1_score(test_labels, predictions, labels=[2.0], average='micro'),
            'class_3_F1': f1_score(test_labels, predictions, labels=[3.0], average='micro'),
            'class_4_F1': f1_score(test_labels, predictions, labels=[4.0], average='micro'),
            'class_5_F1': f1_score(test_labels, predictions, labels=[5.0], average='micro'),
            'accuracy': accuracy_score(test_labels, predictions)
            }


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    print('\n----F1 Scores ----\n')
    for k, v in results.items():
        print(k, v)

