import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import nltk
import os

# Fix NLTK data download issues
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    # Simple preprocessing since your text is already cleaned
    return str(text).lower()

try:
    # Load dataset
    file_path = os.path.join('Data', 'Raw', 'cleaned_spam_dataset.csv')
    df = pd.read_csv(file_path)
    
    # Print dataset info for verification
    print("\nDataset loaded successfully!")
    print("Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
    
    # Assuming:
    # - 'cleaned_text' contains the message content
    # - One of the columns contains labels (spam/ham)
    text_column = 'cleaned_text'  # Your cleaned text column
    label_column = 'Column1'      # Adjust this to whichever column has spam/ham labels
    
    print(f"\nUsing '{text_column}' for text and '{label_column}' for labels")
    
    # Verify columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    
    # Preprocess text
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Create and train model
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    model.fit(df['processed_text'], df[label_column])
    
    # Save model
    os.makedirs('NoteBook', exist_ok=True)
    model_path = os.path.join('NoteBook', 'spam_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel successfully saved to {model_path}")
    print("Training complete!")

except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Check that 'cleaned_spam_dataset.csv' exists in Data/Raw/")
    print("2. Verify which column contains spam/ham labels (change label_column variable)")
    print("3. Ensure your CSV has at least one text column and one label column")
    print("4. If using OneDrive, try moving the project to a local directory")