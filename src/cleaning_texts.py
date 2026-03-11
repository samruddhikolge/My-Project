import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# File paths
input_csv = '/home/samruddhi/Project/data/dialogue_pairs.csv'
output_csv = '/home/samruddhi/Project/data/cleaned_dialogue_pairs.csv'

# Load the paired data
df = pd.read_csv(input_csv)

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define the cleaner
def clean_text(text):
    if not isinstance(text, str):
        return ''
    
    # lowercase
    text = text.lower()
    
    # remove punctuation, numbers, and non-letters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # tokenize words
    words = text.split()
    
    # remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return ' '.join(words)

# Apply cleaning
df['user_text'] = df['user_text'].apply(clean_text)
df['system_response'] = df['system_response'].apply(clean_text)

# Drop empty rows
df = df[(df['user_text'] != '') & (df['system_response'] != '')]

print(f"Cleaned {len(df)} dialogue pairs.")
print(df.head())

# Save output
df.to_csv(output_csv, index=False)
print(f" Saved cleaned pairs to {output_csv}")
