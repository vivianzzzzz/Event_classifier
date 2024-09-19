import gensim
from gensim import corpora
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd


d = pd.read_csv("./events.csv")
d.dropna(subset=["title_details", "type_id"], inplace=True)
# print(d.head())
# print(sum(d["title_details"].isna()))
# documents = d["title_details"].tolist()
documents = list(map(lambda x:x.split(' '), d["title_details"].tolist()))  # Your preprocessed documents
# print(documents)
labels = list(map(lambda x:x-1,list(map(int, d["type_id"].tolist()))))# Labels for each document
assert len(documents) == len(labels)

# Creating the term dictionary and document-term matrix
dictionary = corpora.Dictionary(documents)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]

# Creating and training the LDA model
lda = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=5)

# Extracting features for the classifier
features = [lda.get_document_topics(bow) for bow in doc_term_matrix]
# You'll need to convert the features into a format suitable for the classifier

# Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# Training a classifier
classifier = LogisticRegression()
# print(list(zip(X_train, y_train)))


# Convert each inner list to a dictionary with labels as keys
dict_of_lists = [{label: value for label, value in inner_list} for inner_list in X_train]

# Determine the maximum label across all inner lists
max_label = max(max(d.keys(), default=-1) for d in dict_of_lists)

# Create a new list for each inner list with values for each label, defaulting to 0 if the label is missing
collapsed_X_train = [
    [d.get(label, 0.0) for label in range(max_label + 1)] for d in dict_of_lists
]



print(X_train)
print(collapsed_X_train)
classifier.fit(X_train, y_train)
exit()

# Predicting and evaluating
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
