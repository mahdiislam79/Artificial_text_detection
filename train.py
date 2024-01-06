import pandas as pd
import seaborn as sns
import gc
import matplotlib.pyplot as plt
import joblib
from bpe_tokenizer import BPETokenizer
from transformers import PreTrainedTokenizerFast
from tfidf_vectorizer import CustomTfidfVectorizer
from ensemble import EnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/Dataset/train_v2_drcat_02.csv')
train_ex = pd.read_csv('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/Dataset/train_essays.csv')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size = 0.2, random_state=202)

bpe = BPETokenizer().train(train_ex)
tokenizer = bpe.get_fast_tokenizer()
gc.collect()

tokenized_texts_test = []

for text in X_test.tolist():
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []

for text in X_train.tolist():
    tokenized_texts_train.append(tokenizer.tokenize(text))

tokenized_texts_train_ex = []    
for text in train_ex['text'].tolist():
    tokenized_texts_train_ex.append(tokenizer.tokenize(text))
gc.collect()

with open('custom_bpe_tokenizer.pkl', 'wb') as file:
    joblib.dump(tokenizer, file)

tfidf_vectorizer = CustomTfidfVectorizer()
tfidf_vectorizer.fit(tokenized_texts_train_ex)
tf_train = tfidf_vectorizer.fit_transform(tokenized_texts_train)
tf_test = tfidf_vectorizer.transform(tokenized_texts_test)

with open('custom_vectorizer.pkl', 'wb') as file:
    joblib.dump(tfidf_vectorizer, file)
del tfidf_vectorizer
gc.collect()

# Instantiate the EnsembleClassifier
ensemble_classifier = EnsembleClassifier()

# Fit the model
ensemble_classifier.fit(tf_train, y_train)
gc.collect()

# Make predictions
predictions = ensemble_classifier.predict(tf_test)
cm = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
labels = ('Not artificially generated', 'Artificially Generated')
sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
plt.show()

