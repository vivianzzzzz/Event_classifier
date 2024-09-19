# On Campus Event Classifier at Duke
**Student Event Classification Model Using NLP Techniques**

Team member: Xiyue(Vivian) Zhang, Gavin Li, Lisa Wang

#### Description:
This project aims to classify student group events at Duke University using natural language processing (NLP) techniques. By implementing models such as Logistic Regression with Latent Dirichlet Allocation (LDA) and Transformers (BERT), the project achieved a classification accuracy of up to 98%. 
Here is the full report of the project: 

#### Project Outline:
1. **Data Collection and Preparation**
   - Source: Duke Student Group events.
   - Web scraping using Python libraries (Beautiful Soup, Requests, Selenium).
   - Cleaning: Removal of missing data, standardization of event types, duplicate event checks.

2. **Exploratory Data Analysis (EDA)**
   - Analyzing event distribution.
   - Addressing data imbalance by focusing on Health/Wellness and Social event categories.
   - Tokenization and text preprocessing for model training.

3. **Model Implementation**
   - **Generative Model: LDA + Logistic Regression**
     - Used to identify latent topics in event descriptions.
     - Topics used as features for Logistic Regression.
     - Achieved 76% accuracy on real data and 77% on synthetic data.
   - **Discriminative Model: Transformer (BERT)**
     - Utilized for its contextual understanding of language.
     - Tokenized text and trained using a sequence classification approach.
     - Achieved 97% accuracy on real data and 95% on synthetic data.

4. **Experiment Setup**
   - Introduced a perturbation to the dataset by shuffling word order.
   - Evaluated models on real and synthetic datasets to assess robustness.

5. **Results and Analysis**
   - Comparison of models based on accuracy, training time, precision, recall, and F1 score.
   - LDA + Logistic Regression showed resilience to word order changes.
   - BERT model showed high accuracy but required more computational resources.
   - Analysis on the impact of word shuffling and synthetic data.

6. **Conclusion and Future Work**
   - Trade-offs between Transformer and LDA models discussed.
   - Future enhancements include expanding the dataset, incorporating additional features, and exploring other NLP tasks.

#### How to Implement:
1. **Data Retrieval:**
   - Use provided scripts to scrape event data from the Duke Student Group website.
   - Store collected data in CSV format.

2. **Data Cleaning and EDA:**
   - Clean and preprocess data using the provided Python scripts.
   - Conduct EDA to understand the dataset and prepare it for model training.

3. **Model Training:**
   - **LDA + Logistic Regression:**
     - Train using `sklearn` and `gensim` libraries.
     - Evaluate performance using accuracy, precision, recall, and F1 score.
   - **Transformer (BERT):**
     - Train using the `Hugging Face Transformers` library.
     - Evaluate using the same metrics as the generative model.

4. **Evaluation and Analysis:**
   - Run experiments on both real and synthetic datasets.
   - Analyze the impact of syntactic perturbations.

5. **Results Visualization:**
   - Visualize model performance using graphs and metrics.
   - Compare and interpret the results to draw conclusions.

#### Purpose of the Project:
To enhance event categorization at Duke University by leveraging NLP models, providing more targeted event promotion, and improving student engagement. This project contributes to efficient event management and understanding of event types through the application of machine learning models in natural language processing.
