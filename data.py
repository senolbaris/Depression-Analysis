import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
porter = PorterStemmer()

# Can we create a auto-correction spelling?
# create before cleaning and after cleaning stats

data = pd.read_csv("Suicide_Detection.csv")
#data_info = data.info()
data = data.drop(["Unnamed: 0"], axis=1)

unique_values = data.nunique()
class_unique_values = data["class"].value_counts()

plt.bar(data["class"].unique(), data["class"].value_counts())
plt.title("Distribution of Classes")
#plt.show()

data_no_punc = data.copy()
data_no_punc["text"] = data_no_punc["text"].str.replace(r'[^\w\s]', '', regex=True)

data_no_punc["text"] = data_no_punc["text"].str.lower()

data_no_punc["text"] = data_no_punc["text"].apply(lambda text: " ".join(word for word in text.split(" ") if word not in stop_words))

data_no_punc["text"] = data_no_punc["text"].apply(lambda text: word_tokenize(text))

data_no_punc["text"] = data_no_punc["text"].apply(lambda text: [porter.stem(word) for word in text])

print(data_no_punc["text"])

