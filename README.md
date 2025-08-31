# Presidential Speeches NLP Analysis 📜

This project performs a Natural Language Processing (NLP) analysis on a collection of US presidential speeches. It uses the `gensim` library to create Word2Vec models, which generate word embeddings (vector representations of words). These models allow us to explore word semantics, find the most frequently used words, and discover words with similar contextual meanings.

The analysis is conducted on three levels:
1.  All presidential speeches combined.
2.  Speeches from a specific president (e.g., Franklin D. Roosevelt).
3.  Speeches from a specific group of presidents (e.g., the Mount Rushmore presidents).



## ✨ Features

-   **Text Preprocessing**: Reads and cleans text data from speech files.
-   **Frequency Analysis**: Identifies the most common words in a given corpus.
-   **Word2Vec Model Training**: Creates custom word embedding models using `gensim`.
-   **Semantic Similarity**: Finds words that are contextually similar to a target word (e.g., "freedom").
-   **Comparative Analysis**: Allows for building and comparing models for different presidents or groups.

## 💾 Dataset

This project requires a dataset of presidential speeches, with each speech saved as a separate `.txt` file in the root directory of the project.

The file naming convention is important for filtering speeches by president. A suggested format is `president-name_speech-title.txt`. For example:
-   `franklin-d-roosevelt_inaugural-address.txt`
-   `george-washington_farewell-address.txt`

The helper functions in `president_helper.py` rely on these filenames to group speeches correctly.

## ⚙️ Installation & Setup

To run this project, you need to have Python 3 installed. Follow these steps to set up the environment:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required Python libraries:**
    It's recommended to use a virtual environment.
    ```bash
    pip install gensim spacy
    ```

3.  **Download the spaCy language model:**
    The preprocessing functions in `president_helper.py` likely use a spaCy model for tasks like tokenization or lemmatization.
    ```bash
    python -m spacy download en_core_web_sm
    ```

## 🚀 Usage

1.  Place your `.txt` speech files in the root directory of the project.
2.  Ensure you have a `president_helper.py` file containing the necessary helper functions (`read_file`, `process_speeches`, etc.).
3.  Run the main analysis script from your terminal:
    ```bash
    python main_analysis.py
    ```

The script will print the results of the analysis directly to the console, including the most frequent words and lists of similar words for each model.

## 📄 Code Overview

-   **`main_analysis.py`**: The main script that orchestrates the entire process. It loads the data, preprocesses it, and trains the different Word2Vec models for analysis.
-   **`president_helper.py`** (Not included in the prompt, but assumed): This file should contain all the helper functions for reading files, cleaning and processing text, and filtering sentences by president.
-   **`create_embeddings(sentences)` function**: A key function in `main_analysis.py` that encapsulates the `gensim.models.Word2Vec` training logic. This makes it easy to create multiple models with consistent parameters.

### Example Output

When you run the script, you can expect output similar to this:
