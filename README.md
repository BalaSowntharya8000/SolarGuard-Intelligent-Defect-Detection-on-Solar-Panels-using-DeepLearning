# SolarGuard-Intelligent-Defect-Detection-on-Solar-Panels-using-DeepLearning

## Problem Statement
Solar energy is a crucial renewable resource, but the accumulation of **dust, snow, bird droppings, and physical or electrical damage** on solar panels can significantly reduce their efficiency. Manual monitoring of these panels is often time-consuming, labor-intensive, and costly, particularly in large-scale solar farms.

This project aims to develop a **deep learning-based image classification model** that can automatically identify the condition of solar panels from images.

The objective is to accurately classify solar panel images into six categories:
**Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, and Snow-Covered.**

### Domain: Renewable Energy and Computer Vision

**Goal       :** Automatically classify defects in solar panel images  
**Approach   :** CNN-based image classification using EfficientNet architecture  
**Interface  :** Deployed via Streamlit for real-time image upload, prediction, and visualization  
**Dataset    :** Folder-based dataset with 6 classes:  
                  - Clean  
                  - Dusty  
                  - Bird-Drop  
                  - Electrical-Damage  
                  - Physical-Damage  
                  - Snow-Covered

**Key Input Features**
**Input Type  :** RGB images of individual solar panels  
**Image Size  :** Resized uniformly (e.g., 224x224) for model compatibility  
**Channels    :** 3 color channels (Red, Green, Blue)
**Format      :** Folder-structured image dataset (1 folder per class). Each folder name becomes the class label. 
**Classes     :** Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Snow-Covered  

**Key Highlights**
- Automated classification of solar panel defects using deep learning  
- CNN-based architecture (EfficientNetB3) chosen for high accuracy and efficiency  
- Real-time prediction interface built with Streamlit  
- Logs predictions and timestamps for monitoring & analysis  
- Visual dashboard to explore defect trends over time  
- Dataset organized into 6 clearly labeled classes for clean training  
- Lightweight and ready for deployment in smart solar farms

## Business Use Cases

🔹 **Automated Solar Panel Inspection** - Develop an AI-based system to automatically classify the condition of solar panels, reducing the need for manual inspections.
🔹 **Optimized Maintenance Scheduling** - Identify which panels require immediate cleaning or repair, optimizing maintenance efforts and reducing operational costs.
🔹 **Efficiency Monitoring** - Analyze how common issues like dust, snow, or damage impact panel performance and generate insights to improve efficiency.
🔹 **Smart Solar Farms** - Integrate the classification system into smart solar farms to trigger alerts for cleaning or maintenance, ensuring maximum energy output.

###  Technologies Used
- Python: Core language for data processing, model development, and deployment
- Pandas: Data manipulation and preprocessing (merging, filtering, grouping, CSV logging)
- NumPy: Efficient numerical operations and array handling
- TensorFlow / Keras: Deep learning framework used to build and train the EfficientNetB3 image classification model
- EfficientNetB3: Lightweight and high-performing CNN architecture used for defect classification
- Matplotlib: Basic plotting of data distribution (if used)
- Plotly: Interactive visualizations for defect trends, bar and line charts in the dashboard
- Streamlit: Used to build the web application interface for real-time image upload, classification, and visualization
- OS / Glob: For reading image files and navigating folder structures programmatically
- CSV (Logging): Used to store predictions and timestamps for monitoring and analysis

### Setup Instructions
To manage dependencies separately from the global Python environment, this project requires **Python 3.11** version on the system.

**Create a virtual environment** 
Inside your project folder, run the following command: python -m venv env

**Activate the environment** 
On Windows: - .\env\Scripts\activate

### Installation Instructions
To run the SolarGuard: Intelligent Defect Detection on Solar Panels project, install the required libraries using pip:
**pip install streamlit pandas numpy matplotlib plotly tensorflow keras**

#### Breakdown of Packages Used
- streamlit → Web app framework (Dashboard / Frontend for image upload & prediction)
- pandas → Data manipulation, CSV logging, and summarizing prediction results
- numpy → Numerical operations and image array preprocessing
- plotly → Interactive visualizations (bar chart, line chart for defect trends)
- matplotlib → Static plots (if required for data distribution or sample images)
- tensorflow / keras → Deep learning model training using EfficientNetB3 for classification
- os / glob → Accessing image folders, reading file paths, batch loading
- csv → Logging prediction results with date and time into prediction_results.csv

### Code File Structure
**1_EDA_SolarPanel_Classification.ipynb** -	Jupyter notebook containing EDA, preprocessing, and EfficientNetB3-based model training
**solar_guard_app.py** - Main Streamlit application script for real-time image classification and defect visualization
**Deep_Learning_Project_Guide** -	Project documentation guide outlining objectives, workflows, key concepts, and results

### Data Sources
**Dataset Link:** Faulty Solar Panel Image Dataset (Google Drive)
**Source Type:** Custom image dataset curated for solar panel defect classification
**Format:** Folder-structured dataset - Each folder name represents a class label

**Dataset Structure**
The dataset contains the following folders (each representing one class).
**Classes (6 Categories)**
- Clean
- Dusty
- Bird-Drop
- Electrical-Damage
- Physical-Damage
- Snow-Covered

**Image Format** - .jpg / .png
**Labeling** - Labels are automatically inferred from folder names during data loading using directory traversal.

#### Preprocessing Summary:
**Image Resizing:** All images are resized to 330 × 330 pixels (EfficientNetB3 standard input)
**Normalization:** Pixel values scaled to the range [0, 1] for better convergence
**Data Augmentation:** May include flips, rotations, zoom, and brightness adjustments during training to improve generalization

### Approach
**Data Cleaning & Preprocessing** 
- Verified and organized image files into a folder-based class structure.
- Resized all images to 330×330 pixels to match EfficientNetB3 input requirements.
- Normalized pixel values to the range [0, 1] for model efficiency.
- Auto-labeled classes using folder names (e.g., Clean, Dusty, Snow-Covered, etc.).

**Exploratory Data Analysis (EDA)**
- Conducted class distribution analysis to check for dataset imbalance.
- Displayed image counts per class using bar charts.
- Sampled image grids from each class to visually confirm labeling and quality.

#### Model Training
- **Used a pre-trained EfficientNetB3** CNN model with fine-tuning for solar panel defect classification.
- Applied transfer learning to adapt the model to the 6-class image dataset.
  - **Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Snow-Covered**
- Trained on normalized image data with batch-wise loading and real-time data augmentation for better generalization.

##### Architecture & Training Steps:
**Base Model:** EfficientNetB3 with include_top=False to remove the default classification head.
**Custom Layers Added:**
- GlobalAveragePooling2D() to flatten the output of the convolutional base.
- Final Dense(6, activation='softmax') layer for multi-class classification.
**Fine-Tuning:** Unfroze the top layers of EfficientNetB3 to allow training on domain-specific solar panel images.
  
##### Compilation:
**Loss Function:** categorical_crossentropy (since the problem is multi-class and labels are one-hot encoded).
One-hot encoded labels look like this for 3 classes:
[1, 0, 0] → Class 0
[0, 1, 0] → Class 1
[0, 0, 1] → Class 2
categorical_crossentropy compares the entire one-hot vector with the model’s softmax probabilities.
**Optimizer:** Adam (adaptive and efficient for CNNs).
**Metrics Used:** 
- During training: accuracy
- During evaluation: precision, recall, and F1-score for better performance insights.

**Accuracy**
→ The percentage of correctly predicted images out of all predictions. (Good for balanced datasets)
**Precision**
→ Measures how many of the predicted positive cases (e.g., "Dusty") are actually correct.
- Formula: TP / (TP + FP)
**Recall**
→ Measures how many of the actual positive cases were correctly identified.
- Formula: TP / (TP + FN)
**F1-Score**
→ The harmonic mean of precision and recall, balancing both false positives and false negatives. (Useful for imbalanced classes)

##### Data Feeding
Used ImageDataGenerator with:
  - rescale=1./255 for **normalizing pixel values**
  - Real-time image augmentation
  - flow_from_directory() for efficient batch loading from folder-structured dataset
**Input Shape:** Resized all images to 330×330×3 to match EfficientNetB3 requirements.
**Training:** Performed over multiple epochs with early stopping to avoid overfitting and ensure generalization.

#### Streamlit Web Features

