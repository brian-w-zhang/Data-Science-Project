# Multimodal Crisis Data Analysis

This project is a multimodal data science solution developed for the **50.038 Computational Data Science** course at SUTD.

## Problem Description
During a natural disaster or crisis, a vast amount of information is shared on social media, containing both text and images. The goal of this project is to build a multimodal classification system that can automatically categorize social media posts (tweets and images) into whether they are "informative" (related to a disaster) or "not informative" (safe). Filtering out non-informative content can significantly aid disaster response and management teams in prioritizing critical information.

## Dataset
The primary dataset used for this project is **CrisisMMD v2.0**, a large-scale multimodal dataset collected during natural disasters. It contains a collection of tweets paired with images, along with human annotations.

## Setup Instructions for Teammates
To keep the repository lightweight, the large dataset is not included in the Git history. Please follow these steps to set up the project locally:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/brian-w-zhang/Data-Science-Project.git
    ```
2.  **Download the Dataset**:
    Download the `CrisisMMD_v2.0` folder from our shared **Google Drive**.
3.  **Place the Dataset**:
    Move the downloaded `CrisisMMD_v2.0` folder into the **root directory** of this repository. The structure should look like this:
    ```text
    Data-Science-Project/
    ├── CrisisMMD_v2.0/   <-- Add this here
    ├── dataprep.ipynb
    ├── README.md
    └── ...
    ```
4.  **Install Dependencies**:
    Ensure you have your virtual environment set up and the required libraries (pandas, torch, transformers, etc.) installed.

## Repository Structure

*   **`dataprep.ipynb`**: Handles the data collection, pre-processing, and cleaning steps. It reads the `crisismmd_master.tsv` annotations, extracts and labels the text for the NLP model (saving it as `clean_crisismmd_tweets.csv`), and sorts the raw images into appropriate folders (`disaster` and `safe`) for the computer vision model.
*   **`nlpmodel.ipynb`**: Contains the training and evaluation for the text modality. It uses a Transformer-based NLP model (e.g., DistilBERT) to classify the text from the tweets.
*   **`resnetmodel.ipynb`**: Contains the training and evaluation for the image modality. It uses a ResNet-based Deep Learning model for computer vision to classify the images provided in the tweets.
*   **`CrisisMMD_v2.0/`**: Directory containing the raw images and annotations for the dataset.
*   **`train.csv` & `test.csv`**: Additional dataset files used for modeling purposes.

## Deliverables
- Check-off and presentations
- Usable user interface (UI) to demonstrate the model 
- Final project report in PDF format evaluating and discussing the methodologies, approaches, and findings.
