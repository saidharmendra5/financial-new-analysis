from pathlib import Path
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize objects
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def clean_dataset(input_file = Path("C:/Users/sai/OneDrive/Desktop/ST-16-Finsent/data/all-data.csv")):
    # Load dataset
    df = pd.read_csv(input_file)

    print("\n📌 Original Dataset Columns:", df.columns)  # Debugging: Print column names

    # Ensure 'Content' and 'Type' columns exist
    if 'Content' not in df.columns or 'Type' not in df.columns:
        raise KeyError("The required columns ('Content' and 'Type') are missing in the dataset! Check CSV file headers.")

    # Remove rows with missing values in 'Content'
    df = df.dropna(subset=['Content'])  

    # Clean text: remove special characters and extra spaces
    df['Cleaned_content'] = df['Content'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x).strip())

    # Tokenize, remove stopwords, and apply stemming
    stop_words = set(stopwords.words('english'))
    df['Cleaned_content'] = df['Cleaned_content'].apply(lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x) if word.lower() not in stop_words]))

    # Keep only the 'Type' and 'Cleaned_content' columns
    df = df[['Type', 'Cleaned_content']]

    print("\n✅ Cleaned Dataset Preview:")
    print(df.head())  # Show first few rows after processing

    # Overwrite the same file
    output_file = Path("C:/Users/sai/OneDrive/Desktop/ST-16-Finsent/data/cleaned_data.csv")
    df.to_csv(output_file, index=False)

    print(f"\n📂 Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    clean_dataset()
