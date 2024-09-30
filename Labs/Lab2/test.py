import nltk
nltk.download('reuters')
from nltk.corpus import reuters
from nltk import word_tokenize, sent_tokenize

nltk.download('punkt_tab')
import pandas as pd

# print("The number of documents are: ", len(reuters.fileids()))
# print("The number of words are: ", len(reuters.words()))
# print("The number of sentences are: ", len(reuters.sents()))

file_id = 'training/9920'
words_in_file = reuters.words(file_id)
sents_in_file = reuters.sents(file_id)

num_words_in_file = len(words_in_file)
print(f"Number of words in file '{file_id}': {num_words_in_file}")
prepositions = {"about", "above", "across", "after", "against", "along", "among", "around", "as", "at", "before",
                "behind", "below", "beneath", "beside", "between", "beyond", "but", "by", "concerning", "despite",
                "down", "during", "except", "for", "from", "in", "inside", "into", "like", "near", "of", "off", "on",
                "onto", "out", "outside", "over", "past", "regarding", "round", "since", "than", "through",
                "throughout", "to", "toward", "under", "underneath", "until", "up", "upon", "with", "within", "without"}
preposition_count = 0

for word in words_in_file:
    if word.lower() in prepositions:
        preposition_count += 1
print(f"Number of single-word prepositions in file '{file_id}': {preposition_count}")


from collections import defaultdict

categories = reuters.categories()
file_ids_by_category = defaultdict(list)
for category in categories:
    file_ids_by_category[category] = reuters.fileids(category)
for category, file_ids in file_ids_by_category.items():
    print(f"Category: {category}, FileIDs: {file_ids}")


#6
def word_freq(word, file_id):
    words_in_file = reuters.words(file_id)
    word_count = words_in_file.count(word)
    total_words = len(words_in_file)

    frequency = word_count / total_words
    return word_count, frequency

# Example usage:
word = "said"
file_id = "training/9920"
count, freq = word_freq(word, file_id)
print(f"Word '{word}' appears {count} times in file '{file_id}', with a frequency of {freq:.4f}")


print(reuters.raw('training/9920'))


