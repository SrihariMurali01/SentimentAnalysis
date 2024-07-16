import nltk
import os

# Define the base directory for NLTK data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# Ensure the directory exists
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)
