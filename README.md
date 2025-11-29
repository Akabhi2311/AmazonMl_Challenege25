🚀 Amazon Summer Challenge 2025 – Image Classification (Top 10%)

This repository contains my end-to-end solution for the Amazon Summer Challenge 2025, where participants were given a 75,000-image training dataset with associated feature metadata and were required to predict labels for an additional 75,000 unlabeled test images.

I achieved a Top 10% leaderboard ranking by building a multimodal (image + tabular) deep learning pipeline optimized for large-scale inference.

🧩 Problem Statement

Amazon encounters millions of product images uploaded daily. The challenge simulated a real workflow where participants needed to:

Build an image classification model using
✔ 75K labeled training images
✔ 75K unlabeled test images
✔ Metadata/features for each image (tabular inputs)

Predict the correct class for each test image

Generate a submission file in the required format

Optimize for both accuracy and inference performance

📁 Dataset Details

The dataset was divided into:

dataset/
│
├── train/
│   ├── train_images/            # 75,000 images
│   ├── train.csv                # metadata + labels
│
├── test/
│   ├── test_images/             # 75,000 images
│   ├── test.csv                 # metadata only
│
└── sample_submission.csv

Metadata Included:

Numerical features

Categorical attributes

Pre-extracted product information

Label (train.csv only)

🛠 Approach
1. Image Preprocessing

Resizing images (224×224)

Normalization

Augmentations: rotations, flips, color jitter

Loaded efficiently using PyTorch DataLoader

2. Tabular Feature Engineering

Missing value imputation

Encoding categorical fields

Normalization of continuous features

Feature interaction + frequency encoding

3. Model Architecture

A multimodal fusion model:

🔹 Vision Branch

Pretrained CNN (ResNet50 / EfficientNet)

Extracted 512–1024D image embeddings

🔹 Tabular Branch

2–3 layer MLP

ReLU + BatchNorm + Dropout

🔹 Fusion Layer

Concatenation of both embeddings

Dense layers → Softmax output

4. Training

Loss: CrossEntropy

Optimizer: AdamW

Scheduler: Cosine with Warmup

Trained for 15–25 epochs

Early stopping on Macro F1

5. Evaluation Metric

Macro F1 Score
(Used due to class imbalance.)

🏆 Results

Ranked in the Top 10% of all teams

Successfully handled 150K images end-to-end

Achieved strong generalization using a multimodal architecture

Built a scalable inference pipeline

📂 Repository Structure
Amazon-Summer-Challenge-2025/
│
├── data/
│   ├── train_images/
│   ├── test_images/
│   ├── train.csv
│   ├── test.csv
│
├── src/
│   ├── preprocess.py
│   ├── dataset_loader.py
│   ├── model.py
│   ├── train_model.py
│   ├── inference.py
│   ├── utils.py
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Training.ipynb
│   ├── Image_Feature_Fusion.ipynb
│
├── submission/
│   └── submission.csv
│
├── requirements.txt
└── README.md

⚡ How to Run
Install dependencies:
pip install -r requirements.txt

Preprocess the dataset:
python src/preprocess.py

Train the model:
python src/train_model.py

Generate predictions:
python src/inference.py

🎯 Key Highlights

📌 Multimodal ML: Vision + Tabular fusion

📌 Handles large datasets (150K images)

📌 Clean inference pipeline for fast batch predictions

📌 Reproducible code and modular design

📜 License

MIT License — feel free to use and modify.
