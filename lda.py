# import logging

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import cm
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score, )
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from tomotopy import HDPModel

# from lda_classification.model import TomotopyLDAVectorizer
# from lda_classification.preprocess.spacy_cleaner import SpacyCleaner

# #############################################

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# workers = 4 #Numbers of workers throughout the project

# use_umap = False #make this True if you want to use UMAP for your visualizations

# min_df = 5 #Minimum number for document frequency in the corpus
# rm_top = 5 #Remove top n frequent words

# labels = [1, 2, 3, 4, 5] #Labels for the dataset



# # Load the 20 newsgroups dataset
# data = fetch_20newsgroups(subset='train', categories=labels)

# # Split the dataset into train and test sets
# docs_train, _ = train_test_split(data.data, test_size=0.2, random_state=42)

# hdp_model = HDPModel(min_df=min_df, rm_top=rm_top)
# hdp_model.optim_interval = 5
# for d in docs_train:
#     hdp_model.add_doc(d)
# hdp_model.burn_in = 100
# hdp_model.train(0, workers=workers)
# for i in range(0, 1000, 10):
#     hdp_model.train(10, workers=workers)
#     print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, hdp_model.ll_per_word, hdp_model.live_k))

# num_of_topics = hdp_model.live_k