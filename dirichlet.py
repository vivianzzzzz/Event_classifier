import gensim
from gensim import corpora
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Assuming you have a list of documents and a list of their corresponding labels
documents = [...]  # Your preprocessed documents
labels = [...]     # Labels for each document

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
classifier.fit(X_train, y_train)

# Predicting and evaluating
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
