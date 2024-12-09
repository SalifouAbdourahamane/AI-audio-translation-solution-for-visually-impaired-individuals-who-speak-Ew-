
# Ewe Audio Command Classification Challenge - Solution by Abdourahamane Ide Salifou - Engineering student in Artificial Intelligence at CMU-Africa

## Overview
This solution addresses the challenge of classifying basic directional commands in the Ewe language using audio data. The goal is to accurately classify commands like "up," "down," "go," and others . This solution emphasizes efficiency in feature extraction, robust model architecture, and optimization for deployment on edge device. This solution is tailored to the provided use case: a navigation voice assistant for visually impaired individuals who speak Ewe.

## Objectives
- Develop a robust model that accurately classifies accurately basic directionnal audio commands .
- Ensure efficient inference, especially on edge devices with limited computational resources .
## Data Preprocessing & Feature Engineering
1. **Feature Extraction**:
   - Audio data is converted into **Mel Spectrogram**, **MFCC**, **Delta**, and **Delta-Delta** features.
   - These features are concatenated into a combined feature matrix for each audio file.
   - Audio features are padded or truncated to ensure consistent input dimensions for the model.
   
2. **Preprocessing**:
   - The training data consists of 5334 audio samples. The data is split into training (80%) and validation (20%) sets.
   - The labels are one-hot encoded for multi-class classification.

3. **Inference Optimization**:
   - To reduce inference time, the test data features are precomputed and saved in a `.npy` file. This prevents feature extraction from becoming a bottleneck during real-time inference.

## Model Architecture
The model I built from scratch is a custom **Convolutional Neural Network (CNN)** with residual connections, designed for robustness and efficiency:
- **Conv2D Layers** are used for extracting local features.
- **Residual Blocks** improve learning capacity and prevent gradient vanishing.
- **Global Average Pooling** helps reduce overfitting by aggregating features globally.
- The model is trained using the **Adam optimizer** with an initial learning rate of 0.0001, and **L2 regularization** is applied to control overfitting.

## Training Setup
- **Hardware**: Model training was performed on a **Kaggle kernel** with **T4 GPU x2**.
- **Time**: Total training time was approximately 1 hour and 12 minutes.
- **Cross-Validation**: The model achieved an average validation accuracy of **98.%** with 3-fold cross-validation.
  
## Inference Time Optimization
- **Feature Precomputation**: By using a strategic approach of extracting audio features and a well crafted model architecture the inference time is **17.21 seconds** for the entire test set ~2000 audiofiles.
  
## Reproducibility
- **Random Seed**: A fixed random seed (SEED=42) is set to ensure the same results are produced across multiple runs.
- **Model Checkpoints**: The model is saved at its best-performing state during training to guarantee consistent results.

## Submission
- **Submission File**: The model predictions are saved in a CSV file (`final_ubmission.csv`) with the required format.
- **Predictions**: The final predictions are based on the provided test data for the challenge.

## Requirements
- **Python**: 3.10.14
- **TensorFlow**: 2.16.1
- **Librosa**: 0.10.2.post1
- **NumPy**: 1.26.4
- **Pandas**: 2.2.2
- **Seaborn**: 0.12.2
- **Matplotlib**: 3.7.5


## How to Run
1. **Set up Environment**:
   - Install dependencies: `pip install -r requirements.txt`
   - Import the required libraries 
   
2. **Prepare Data**:
   - Please Make sure to replace the path for audio files and CSV files with the ones provided for the competition  .
   
3. **Order in which the solution notebook must be run after importing the libraries**:
   - First :   run the data loading & exploration cell in the solution notebook: 
   - Second :  run the data processing and feature engineering cell in the solution notebook
   - Third :   run the model building and training cell in the solution notebook 
   - Finally : run the inference & submission cell in the solution notebook 
   

## Performance Metrics
- **Validation Accuracy**: ~99.12%
- **Public Leaderboard Accuracy**: 96.21%

## Acknowledgments
Special thanks to Zindi ,TechCabal and Umabaji for organizing the challenge and providing this opportunity to develop impactful AI solutions for social good particularly in underrepresented African languages.
