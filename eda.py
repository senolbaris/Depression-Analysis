import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import FreqDist

data = pd.read_csv("Suicide_Detection.csv")

unique_values = data.nunique()
class_unique_values = data["class"].value_counts()

plt.bar(data["class"].unique(), data["class"].value_counts())
plt.title("Distribution of Classes")
#plt.show()

# Total number of strings and average for each classes
train_string = " ".join(data["text"].values)
train_string = train_string.split()
print(len(set(train_string)))

freq_list = FreqDist(train_string)
print(freq_list.most_common(10))
 # Check most common words before and after cleaning data, also check for each classes separately
