# Twitter Sentiment Analysis

## Overview
This repository contains a comprehensive project for performing sentiment analysis on Twitter data using machine learning models. The project includes data preprocessing, vectorization, model training, and evaluation steps. The goal is to classify the sentiments expressed in tweets as positive, negative, or neutral.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features
- Preprocessing of raw tweet text
- TF-IDF vectorization of tweet data
- Training and evaluation of a sentiment classification model
- Saving and loading of the trained model and vectorizer
- Utilizes NLTK for natural language processing tasks
- Includes Jupyter Notebook for interactive analysis

## Installation
### Prerequisites
- Python 3.7 or higher
- Git
- Git LFS (Large File Storage)

### Clone the Repository
```bash
git clone https://github.com/SrihariMurali01/SentimentAnalysis.git
cd SentimentAnalysis
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Setup Git LFS
Ensure you have Git LFS installed and initialized:
```bash
git lfs install
git lfs track "samodel.pkl"
git lfs track "tfidf_vectorizer.pkl"
git lfs track "dataset.csv"
git lfs track "nltk_data/**"
```

### Download NLTK Data
```bash
python -m nltk.downloader -d ./nltk_data all
```

## Usage
### Running the Analysis
You can run the analysis using the provided Python script or Jupyter Notebook.

#### Using Python Script
```bash
python main.py
```

#### Using Jupyter Notebook
Open `main.ipynb` in Jupyter Notebook to interactively run the analysis and visualize the results.

### Directory Structure
```
├── dataset.csv               # The dataset of tweets
├── main.ipynb                # Jupyter Notebook for analysis
├── main.py                   # Python script for analysis
├── nltk_data                 # Directory containing NLTK data
├── nltk_resources.py         # Script for downloading NLTK resources
├── samodel.pkl               # Trained sentiment analysis model
├── templates                 # Directory for any HTML templates
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
└── requirements.txt          # List of dependencies
```

## Data
The dataset (`dataset.csv`) contains tweets with their respective sentiment labels. Each tweet is preprocessed and vectorized using TF-IDF before being fed into the machine learning model.

## Model
The sentiment analysis model is a machine learning model trained to classify tweets as positive, negative, or neutral. The model is saved as `samodel.pkl` and the TF-IDF vectorizer as `tfidf_vectorizer.pkl`.

## Results
The model's performance is evaluated using standard metrics such as accuracy, precision, recall, and F1-score. Detailed results and visualizations can be found in the Jupyter Notebook (`main.ipynb`).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

