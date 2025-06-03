@"
# Emotext-AI

A machine learning application that classifies text into 13 distinct emotions using BERT.

## Project Overview

This project uses a fine-tuned BERT model to analyze and classify text into 13 emotional categories including happiness, sadness, worry, surprise, and more. The model achieves 32.4% accuracy across these 13 classes, which is 4.2 times better than random guessing.

## Features

- Text classification into 13 distinct emotion categories
- Web interface for real-time emotion analysis
- Flask API for serving predictions
- Visualization of confidence scores and top emotions

## Note on Model Files

Large model files are not included in this repository due to GitHub size limitations. To use this project:

1. Train the model yourself by running:
python src/emotion_classifier.py

plaintext

Hide

2. Or download the pre-trained model from [Hugging Face Models](https://huggingface.co/models) (similar BERT emotion models)

## Project Structure

- `app.py`: Flask API for serving the emotion classification model
- `index.html`: Web interface for interacting with the model
- `src/`: Source code for model training and visualization
- `results/`: Visualizations of model performance

## Installation

1. Clone this repository:
git clone https://github.com/ANSHIKA21M/Emotext-AI.git
cd Emotext-AI

plaintext

Hide

2. Install dependencies:
pip install -r requirements.txt

plaintext

Hide

3. Train or download the model as described above

## Usage

1. Start the Flask API:
python app.py

plaintext

Hide

2. Open `index.html` in your browser

3. Enter text and analyze emotions!

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers (BERT)
- Flask
- HTML/CSS/JavaScript

## Results

The model achieves 32.4% accuracy across 13 emotion classes, with "happiness" showing the best performance (F1-score: 0.46).

## License

This project is licensed under the MIT License.
"@ | Out-File -FilePath README.md -Encoding utf8