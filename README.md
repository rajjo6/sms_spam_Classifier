# sms_spam_Classifier
sms-spam classifier is  classification model used to detect spam messages. 
Problem Statement
The surge in SMS usage as a primary communication method has resulted in a notable increase in unwanted messages. Recent surveys show that 76% of Indians receive 3 or more promotional or spam SMS messages daily, with financial services and real estate sectors being the top offenders. This project addresses this widespread issue by creating an automatic system to categorize SMS messages into either spam or legitimate (ham) groups.


Key Features-
Text Preprocessing Pipeline: Converts text to lowercase, removes special characters, tokenizes messages, eliminates stopwords, and applies Porter stemming algorithm

Feature Extraction: Utilizes Count Vectorizer and TF-IDF techniques to transform text data into numerical features

Advanced Feature Engineering: Analyzes message length patterns (spam messages tend to be longer) and word frequency distributions

Multiple Naive Bayes Models: Comparison between Gaussian, Multinomial, and Bernoulli variants with comprehensive accuracy metrics

Interactive User Interface: Simple and intuitive Streamlit frontend with text input and prediction functionality

High Accuracy Classification: Achieves 97% accuracy in distinguishing between spam and legitimate messages

Cloud Deployment: Accessible online through Render deployment



Technology Stack-
Core Language: Python 3.9

Data Processing: Pandas for data manipulation and analysis

Text Processing: NLTK for tokenization, stemming, and stopword removal

Machine Learning: Scikit-learn for model implementation and evaluation

Data Visualization: Matplotlib and Seaborn for EDA visualization

Web Interface: Streamlit for creating the interactive user interface

Deployment: Render platform for cloud hosting



sms-spam-classifier/
├── data/
│   └── spam.csv              # The dataset containing SMS messages and their labels
├── models/
│   └── model.pkl             # Trained Multinomial Naive Bayes model
│   └── vectorizer.pkl        # CountVectorizer for text transformation
├── notebooks/
│   └── sms_spam_detection.ipynb # Jupyter notebook with data analysis and model development
├── src/
│   ├── preprocess.py         # Text preprocessing functions
│   ├── train.py              # Model training functions
│   └── predict.py            # Prediction functions
├── app.py                    # Streamlit application
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation


Installation
To set up the project locally, follow these steps:

1)Clone the repository:
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier

2)Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3)Install the required dependencies:
Install the required dependencies:

Usage
1)Run the Streamlit application:
streamlit run app.py
2)The application will open in your default web browser. If it doesn't, navigate to the URL displayed in the terminal (typically http://localhost:8501).

3)Enter an SMS message in the text box and click the "Predict" button to see if the message is classified as spam or ham.





Implementation Details

Data Cleaning & EDA
Removed unnecessary columns from the dataset

Checked for null values and duplicates to ensure data integrity

Created visualizations to understand the distribution of spam vs. ham messages

Added feature columns including word count, character count, and sentence count

Generated histograms and pairplots to identify patterns between spam and legitimate messages

Discovered that spam messages tend to have more characters and words than ham messages

Text Preprocessing
Implemented a comprehensive transform_text function that performs the following operations:

Converts all text to lowercase

Tokenizes the text into individual words

Removes special characters and punctuation

Eliminates common English stopwords

Applies stemming using the Porter Stemming algorithm

Additionally, analyzed word frequency distributions to identify common words in spam and ham messages.

Model Building
Evaluated three Naive Bayes variants for text classification:

Gaussian Naive Bayes: Assumes features follow a normal distribution

Multinomial Naive Bayes: Suitable for discrete features (like word counts)

Bernoulli Naive Bayes: Models binary features (word presence/absence)

After comparing performance metrics:

Multinomial Naive Bayes achieved the highest accuracy and precision

Selected as the final model due to its effectiveness with text count features

Model evaluation included:

Confusion matrix analysis

Classification report with precision, recall, and F1-score

Cross-validation to ensure model stability

Deployment
The application is deployed on Render using the following steps:

Created a requirements.txt file with all dependencies

Set up a Render account and connected it to the GitHub repository

Configured the web service with the start command: streamlit run app.py

Deployed the application with automatic updates on code changes

The live application can be accessed at: [SMS Spam Classifier
](https://sms-spam-classifier-rszu.onrender.com)

Future Improvements
While the current model performs well, several enhancements could be made:

Ensemble Methods: Implement voting classifiers combining multiple models for improved accuracy

Advanced NLP Techniques: Explore word embeddings like Word2Vec or BERT for better text representation

Language Adaptability: Enhance the system to handle multiple languages and regional SMS patterns

User Feedback Loop: Add functionality to learn from user corrections

Explainable AI Features: Implement features to explain why a message is classified as spam or ham

Limitations
It's important to acknowledge the limitations of the Naive Bayes approach:

Independence Assumption: Naive Bayes assumes feature independence, which is rarely true in natural language

Zero-Frequency Problem: The model struggles with words not present in the training data

Context Insensitivity: The model doesn't account for word context or semantic meaning

Probabilistic Output: The probability estimates may not be as reliable as other models

Acknowledgments
The SMS Spam Collection dataset used for training

NLTK and scikit-learn communities for their excellent documentation

Streamlit for making the deployment of ML applications user-friendly

Render for providing hosting for the web application



