# main_analysis.py

# --- 1. Import necessary libraries ---
import os
import gensim # For Word2Vec model creation
import spacy # For advanced text processing (though not explicitly used in this snippet, likely used in president_helper.py)
# Import custom helper functions from another file
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# --- 2. Load Data ---
# Get a sorted list of all speech files from the current directory.
# This assumes the speech files are in the same folder as the script and end with '.txt'.
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])


# --- 3. Preprocess Speeches ---
# Read the content of each speech file into a list.
speeches = [read_file(file) for file in files]

# Process the raw text of each speech. This step likely involves cleaning the text,
# tokenizing it into sentences, and maybe lemmatization or stop-word removal.
processed_speeches = process_speeches(speeches)


# Merge all processed sentences from different speeches into a single list.
all_sentences = merge_speeches(processed_speeches)


# --- 4. General Analysis (All Presidents) ---
# Find and display the most frequently used words across all speeches.
most_freq_words = most_frequent_words(all_sentences)
print("Most frequent words across all presidents:", most_freq_words)


# --- 5. Word2Vec Model Creation ---
# Define a function to create a Word2Vec model.
# Parameters:
# - sentences: The list of tokenized sentences.
# - size=96: The dimensionality of the word vectors.
# - window=5: The maximum distance between the current and predicted word within a sentence.
# - min_count=1: Ignores all words with a total frequency lower than this.
# - workers=2: Use two worker threads to train the model.
# - sg=1: Use the Skip-gram model (sg=0 would be CBOW).
def create_embeddings(sentences):
    return gensim.models.Word2Vec(sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# Create a Word2Vec model using sentences from all presidents.
all_prez_embeddings = create_embeddings(all_sentences)

# Find and display the top 20 words most similar to 'freedom' in the combined model.
similar_to_freedom = all_prez_embeddings.most_similar("freedom", topn=20)
print("\nWords similar to 'freedom' (all presidents):", similar_to_freedom)


# --- 6. Specific Analysis (Franklin D. Roosevelt) ---
# Get all sentences from speeches by Franklin D. Roosevelt.
# The 'get_president_sentences' function likely filters sentences based on the filename.
roosevelt_sentences = get_president_sentences('franklin-d-roosevelt')

# Find the most frequently used words in Roosevelt's speeches.
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)
print("\nMost frequent words by Roosevelt:", roosevelt_most_freq_words)

# Create a Word2Vec model specifically for Roosevelt's sentences.
roosevelt_embeddings = create_embeddings(roosevelt_sentences)

# Find the top 30 words most similar to 'freedom' in Roosevelt's model.
roosevelt_similar_to_freedom = roosevelt_embeddings.most_similar('freedom', topn=30)
print("\nWords similar to 'freedom' (Roosevelt):", roosevelt_similar_to_freedom)


# --- 7. Group Analysis (Mount Rushmore Presidents) ---
# Get sentences from a specific group of presidents.
rushmore_prez_sentences = get_presidents_sentences(["washington", "jefferson", "lincoln", "theodore-roosevelt"])

# Find the most frequently used words by this group.
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)
print("\nMost frequent words by Mount Rushmore presidents:", rushmore_most_freq_words)

# Create a Word2Vec model for the Mount Rushmore presidents.
rushmore_embeddings = create_embeddings(rushmore_prez_sentences)

# Find the top 20 words most similar to 'freedom' in the Mount Rushmore model.
rushmore_similar_to_freedom = rushmore_embeddings.most_similar("freedom", topn=20)
print("\nWords similar to 'freedom' (Mount Rushmore):", rushmore_similar_to_freedom)
