# Amazon-Review-Classifier

This project focuses on text classification, which is one of the most common supervised tasks in the field of natural language processing. The goal of this project is to predict the rating of a review based on its text. The reviews are provided in three diverse domains, and the ratings range from 1 to 5, making it a 5-way classification problem.

## Implementation Details and Comments

1. **Using scikit-learn classifiers:** In this project, we will utilize the out-of-the-box classifiers provided by scikit-learn, a powerful Python library for machine learning. Scikit-learn offers convenient feature extraction using classes such as `CountVectorizer` and `TfidfVectorizer`. You can refer to the scikit-learn documentation at [https://scikit-learn.org](https://scikit-learn.org) for more details. The tutorial on the website includes a full pipeline example that you can use as a reference, understanding the functionality and exploring different parameter configurations.

2. **Feature Extraction:** We will use the Bag-of-Words (BOW) representation for text classification, and optionally consider word n-grams as additional features. While you can incorporate other features, please ensure that your code adheres to the specified runtime requirements.

3. **Vocabulary Reduction:** To optimize processing time, it is recommended to reduce the vocabulary to the top-K most frequent words or word n-grams in the entire training corpus. The value of K should be below 1000. Adding more words to the vocabulary may significantly increase processing time without substantially improving the final accuracy.

4. **Confusion Matrix:** After training and testing the classifier, it is important to print and analyze the confusion matrix. Identify which classes share the highest confusion and provide a short interpretation of the confusion matrix in the report you submit.

5. **Feature Selection:** Python's scikit-learn offers various functions to extract the most discriminative features (words or word n-grams) for a given classification task. One such function is `SelectKBest`. In this project, utilize `SelectKBest` to extract the 15 most effective features for classification. Include these features in the document you submit.

6. **Cross-Domain Classification:** To evaluate the performance of the classifier, conduct cross-domain classification. Train the model on one domain (e.g., sports training data) and test it on another domain (e.g., pets test data). Compare the results with in-domain classification and provide a brief interpretation in the document you submit.

7. **Runtime Limit:** It is crucial to ensure that your code's runtime does not exceed 5 minutes. The code should be designed to produce results and print them within this time frame.

Feel free to explore the scikit-learn documentation and experiment with different approaches and configurations to enhance the performance of your text classification model. Remember to document your findings, interpretations, and any modifications made to the provided guidelines.

Best of luck with your text classification project!
