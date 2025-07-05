
# 🔶 Project Overview

# Title      : SolarGuard – Defect Detection in Solar Panels
# Goal       : Automatically classify defects in solar panel images
# Approach   : Use of CNN-based image classification (EfficientNet)
# Interface  : Deployed using Streamlit for real-time upload, prediction, and visualization
# Dataset    : Image folder-based dataset with 6 classes (Normal, Dusty, Broken, Burn Mark, Color Change, Snow Covered)


# 🔶 Libraries and Packages Used: 

# 1. Model Training & Evaluation

# 🔹 tensorflow / keras - Core deep learning framework used to build, train, compile, evaluate, and save the model.
# 🔹 keras.models       - Enables defining models (Sequential / Functional), saving and loading trained models.
# 🔹 keras.layers       - Provides layers like Dense, Dropout, GlobalAveragePooling2D used in CNN architecture.
# 🔹 keras.applications.efficientnet - Loads pretrained EfficientNetB0/B3 models for transfer learning.
# 🔹 keras.callbacks    - Includes EarlyStopping, ModelCheckpoint, ReduceLROnPlateau for controlling training.
# 🔹 keras.utils        - to_categorical for label encoding; plot_model for visualizing model structure.
# 🔹 sklearn.metrics    - Provides accuracy_score, f1_score, precision, recall, and confusion_matrix.
# 🔹 numpy (np)         - Supports matrix operations, predictions, argmax, and numerical preprocessing.
# 🔹 pandas (pd)        - Used to evaluate and structure model outputs and to analyze predictions.
# 🔹 tqdm               - Adds progress bar to custom loops during training or preprocessing.


#  2. Data Loading & Augmentation

# 🔹 keras.preprocessing.image - ImageDataGenerator used for loading images from directories and applying augmentations.
# 🔹 PIL (Pillow)              - Loads and manipulates images (resize, RGB conversion) for preprocessing.
# 🔹 os                        - Accesses file system, reads directories, and handles paths during dataset loading.
# 🔹 io                        - Handles image buffers and I/O for real-time uploads.


# 3. Visualization & Monitoring

# 🔹 matplotlib.pyplot (plt) - Used to create static plots like training curves and confusion matrices.
# 🔹 seaborn (sns)           - Enhances matplotlib visualizations, especially for heatmaps and confusion matrices.
# 🔹 plotly.express          - Generates interactive charts (bar, pie, line) used in Streamlit for prediction analysis.
# 🔹 datetime                - Adds timestamp to prediction logs, helping track user input history.


# 4. Streamlit Dashboard Deployment

# 🔹 streamlit (st) - Used to build an interactive web dashboard with pages for Upload, Predict, and Visualize.
# 🔹 pandas (pd)    - Displays prediction logs, summary statistics, and user-uploaded data inside Streamlit.
# 🔹 plotly.express - Plots defect frequency and prediction history interactively in the dashboard.
# 🔹 PIL            - Displays and preprocesses uploaded images in Streamlit interface.
# 🔹 io             - Reads in-memory uploaded images without saving them to disk.


# 5. CSV-Based Prediction Logging

# 🔹 csv      - Built-in Python module used to write prediction results to a CSV file for later analysis.
# 🔹 pandas   - Reads and processes prediction logs; enables visualization and sorting/filtering in the dashboard.
# 🔹 datetime - Attaches a timestamp for each prediction record in the CSV.
# 🔹 os       - Verifies if the CSV exists and appends new predictions appropriately.

# Short Note:
# numpy                - Numerical computing (Used in Model: For tensor operations, reshaping arrays, and matrix math)
# pandas               - Data Manipulation and Analysis
# Matplotlib & Seaborn - Data visualization
# tensorflow/keras     - Deep Learning framework
# sklearn(scikit-learn)- Machine learning utilities
# os                   - Operating system interface (Purpose: Directory and file path management)
# cv2 (OpenCV)         - Computer vision tasks
# streamlit            - Web application/dashboard development
# plotly               - Interactive plotting
# PIL (Pillow)         - Image processing
# glob                 - File handling
# warnings             - Suppress warning messages
# datetime             - Time-related operations

# 🔶 Deep Learning Basics

# 🔹Deep Learning is like teaching a computer to think and learn on its own, the way a human brain does.  
# 🔹Instead of writing rules manually, we give the model a lot of examples (like images or text), 
#    and it learns patterns from the data by itself.  
# 🔹It’s called 'deep' because it uses many layers of artificial neurons to understand complex things 
#    like recognizing faces, translating languages, or detecting defects in images.
# 🔹 It works particularly well with unstructured inputs like images, audio, or text.
# 🔹 In this project, deep learning is used to classify defects in solar panel images based on visual patterns in pixel data.
# 🔹 Instead of manually extracting features, the model learns them automatically through multiple layers (CNN + Dense).

# Why Deep Learning for Solar Panel Defect Detection?
# 🔹Defect types (dust, bird drop, burn, snow) are visual and spatial — ideal for deep learning.
# 🔹Traditional ML models can't handle raw image data without manual feature extraction.
# 🔹CNNs like EfficientNet can learn edge, texture, and color patterns relevant to defect classification.
# 🔹EfficientNet was chosen for its balance of accuracy and efficiency in image classification.

#Result:A model that generalizes well to new solar panel images and learns to detect defects without human-designed rules.

# 🔹 Neural Network      - A stack of layers where each neuron learns to detect features or patterns from input data.
# 🔹 Input Layer         - Accepts input data (e.g., 300x300x3 images for solar panels).
# 🔹 Hidden Layers       - Extract deeper abstract features using weights and activation functions 
# 🔹 Output Layer        - Predicts final class using softmax (multi-class probability distribution).
# 🔹 Swish Activation    - A smooth, trainable activation used in EfficientNet that helps improve model accuracy and convergence.
# 🔹 Softmax Activation  - Converts logits to probabilities across classes; used in the output layer for classification.
# 🔹 Forward Propagation - Input flows forward through layers to generate a prediction.
# 🔹 Backward Propagation- Gradients are calculated and weights updated during training to minimize loss.
# 🔹 Categorical Crossentropy - Loss function used for multi-class classification with one-hot encoded labels.
# 🔹 Adam Optimizer      - An adaptive optimizer that combines momentum and learning rate tuning for faster convergence.
# 🔹 Epoch               - One complete pass over the training dataset.
# 🔹 Batch               - Number of samples processed together before updating weights.
# 🔹 Steps per Epoch     - Number of batches processed in one epoch.
# 🔹 Overfitting         - When model performs well on training but poorly on test data — it memorized patterns.
# 🔹 Underfitting        - When model performs poorly on both training and test data — it didn’t learn enough.
# 🔹 Dropout             - Randomly disables neurons during training to reduce dependency and prevent overfitting.
# 🔹 Label Smoothing     - Adjusts one-hot labels slightly to reduce overconfidence and improve generalization.
# 🔹 Transfer Learning   - Uses pretrained models like EfficientNet to extract features; only new classifier layers are trained initially.
# 🔹 Fine-Tuning         - Unfreezing base_model.trainable = True allows updating pretrained layers for better adaptation.


# 🔹 Neural Network
# A network of layers where neurons learn patterns from input data.
# Each layer transforms the data slightly until a final prediction is made.

# 🔹 Input Layer
# Accepts input images (300x300x3 RGB) and feeds them into the model.

# 🔹 Hidden Layers
# Present inside EfficientNet and custom head.
# These intermediate layers extract abstract features like edges, textures, or 
   # defect patterns from the solar panel images.
# The Dense and GlobalAveragePooling layers also act as hidden layers between input and output.

# 🔹 Output Layer
# The final layer used to make the prediction.
# In this project, it's a Dense layer with softmax activation to output class probabilities.

# 🔹 Swish Activation 
# Used internally by EfficientNet layers to learn smoother nonlinear representations.
# More advanced than ReLU and helps the model converge faster with better accuracy.

# 🔹 Softmax Activation
# Converts raw output logits into a probability distribution across all classes.
# Ensures only one class is predicted with the highest confidence.

# 🔹 Forward Propagation
# The process of passing input data forward through the model to generate a prediction.

# 🔹 Backward Propagation
# After prediction, the error is calculated and gradients are back-propagated to update weights.

# 🔹 Categorical Crossentropy
# Measures how well the predicted class probabilities match the one-hot encoded true labels.
# Commonly used in multi-class classification problems.

# 🔹 Adam Optimizer
# Combines momentum and adaptive learning rates for faster, stable convergence.
# Efficient for training deep models like CNNs.

# 🔹 Epoch
# One full cycle through the entire training dataset.

# 🔹 Batch
# A group of samples processed together before the model updates its weights.

# 🔹 Steps per Epoch
# Total number of batches run in one epoch (dataset size ÷ batch size).

# 🔹 Overfitting
# When a model performs well on training data but poorly on unseen data.
# The model memorizes training data instead of learning general patterns.

# 🔹 Underfitting
# When a model performs poorly on both training and validation data.
# It means the model didn’t learn enough — often due to being too simple or trained for too few epochs.

# 🔹 Dropout
# Randomly disables a fraction of neurons during training.
# This forces the model to not rely too heavily on any one neuron and improves generalization.

# 🔹 Label Smoothing
# Slightly reduces the confidence in one-hot encoded labels (e.g., 1 → 0.9).
# Helps prevent overfitting and overconfidence in predictions.

# 🔹 Transfer Learning
# Uses a pretrained model (like EfficientNet) trained on a large dataset (ImageNet).
# Only the final layers are trained initially to adapt to solar panel defect classification.

# 🔹 Fine-Tuning
# After initial training, pretrained layers are unfrozen using base_model.trainable = True.
# This allows the full model to learn domain-specific patterns more deeply.

# 🔹 Confusion Matrix
# A summary table showing how often each class was correctly or incorrectly predicted.
# Helps identify which defects (e.g., Dusty vs Burnt) are being confused by the model.
# Often visualized as a heatmap to highlight misclassification zones.

# 🔹 EarlyStopping
# A training callback that monitors validation loss or accuracy.
# Stops training automatically when the model stops improving for a set number of epochs.
# Prevents overfitting and saves time by not overtraining the model.

# 🔹 ReduceLROnPlateau
# A callback that reduces the learning rate when a metric (like validation loss) stops improving.
# Helps the model escape local minima and converge better after hitting a plateau.
# Usually used with patience and factor settings (e.g., reduce LR by 0.1 after 3 epochs with no improvement).

# 📘 CNN (Convolutional Neural Network) 

# 🔹 CNN (Convolutional Neural Network)
# A special type of neural network designed for image data.
# EfficientNet used in this project is a CNN architecture.
# It includes convolutional layers to extract spatial and texture patterns from solar panel images.

# 🔹 Convolution Layers - To detect local patterns like edges, curves, textures
# Learn local patterns like edges, corners, and textures by sliding filters (kernels) over the image.
# These filters are automatically learned during training.

# 🔹 Pooling Layers - To reduce spatial dimensions and retain important features
# Used within EfficientNet to reduce the spatial dimensions of feature maps.
# This makes the model more efficient and helps it focus on the most important features.
# In this project, pooling is also used explicitly via GlobalAveragePooling2D before the final Dense layer.

# 🔹 GlobalAveragePooling2D
# A specific pooling layer that reduces each feature map to a single value.
# Acts as a bridge between the feature extractor (EfficientNet) and the classifier (Dense layer).
# Helps reduce overfitting by minimizing the number of parameters before the final prediction layer.

# 🔹 Fully Connected Layers (Dense) - To make final predictions based on extracted features
# Take pooled features and use them to classify the image into one of the defect categories.
# In this project, the final Dense layer uses softmax to output the predicted class.

# Summary 
# - CNNs are core to image classification tasks.
# - EfficientNet internally uses convolution and pooling layers.
# - GlobalAveragePooling and Dense layers form the classification head.

# In this project:
# - EfficientNetB0/B3 was used as the CNN architecture.
# - It was pretrained on ImageNet and then fine-tuned for defect classification.
# - The model was applied in the Upload & Predict page of the dashboard to classify the uploaded solar panel image.

# 📘 Classification Problem 

# 🔹 Classification Task
# The goal of this project is to classify images of solar panels into distinct defect categories.
# Each image belongs to exactly one class (e.g., Clean, Dusty, Bird-Drop, Snow, Burn, Electrical).

# 🔹 Multi-Class Classification
# This is a multi-class classification problem — where the model predicts one class from multiple possible classes.
# Classes are mutually exclusive: an image can only belong to one defect category.

# 🔹 Output Representation
# The final Dense layer of the model uses softmax activation to output class probabilities.
# The class with the highest probability is selected as the model's prediction.

# 🔹 Label Format
# Labels are one-hot encoded, meaning only one position is “1” (true class), and others are “0”.
# Example for 6 classes: [0, 0, 1, 0, 0, 0] → class index 2 is the correct label.

# 🔹 Why Deep Learning for This Classification Task?
# CNNs can automatically learn useful spatial features from images (like edges, textures, patterns).
# This eliminates the need for manual feature engineering and works well on visual data like solar panel defects.

# 🔹 Classes in This Project
# - Clean
# - Dusty
# - Bird-Drop
# - Snow
# - Burn
# - Electrical

# Summary:
# - This is a multi-class image classification problem.
# - The model outputs one label per image using softmax and categorical crossentropy.
# - Evaluation metrics like accuracy and F1-score help measure model performance.

# 📘 Evaluation Metrics 

# 🔹 Accuracy
# Measures the overall percentage of correct predictions.
# Formula: (Correct Predictions / Total Predictions)
# ⚠️ Can be misleading if classes are imbalanced.

# 🔹 Precision
# Out of all the predictions made for a given class, how many were actually correct?
# Helps reduce false positives.
# Example: If many clean panels are misclassified as dusty, precision for 'dusty' drops.

# 🔹 Recall
# Out of all actual instances of a class, how many did the model detect correctly?
# Helps reduce false negatives.
# Important for classes like 'burn' or 'electrical' where missing a defect can be critical.

# 🔹 F1-Score
# The harmonic mean of precision and recall.
# Balances both metrics and is more useful when dealing with imbalanced data.
# A higher F1-score indicates a more reliable classifier.

# 🔹 Confusion Matrix
# A grid showing actual vs. predicted classes.
# Used to identify confusion zones where the model struggles to distinguish between similar defects.
# Example: If 'snow' and 'bird-drop' are often confused, it shows in the off-diagonal cells.

# 🔹 Why These Metrics Matter in This Project:
# - Defect classes are not equally represented in the dataset.
# - Precision and recall help identify if the model favors common classes or misses rare ones.
# - F1-score gives a balanced view of how well the model is performing on all classes.

# 📊 Model Evaluation Summary

# Accuracy         : 98.86%
# Precision (avg)  : ~98.87%
# Recall (avg)     : ~98.86%
# F1-Score (avg)   : ~98.86%

# | Class             | Precision | Recall | F1-Score | Support |
# | ----------------- | --------- | ------ | -------- | ------- |
# | Bird-drop         | 0.958     | 0.974  | 0.966    | 117     |
# | Clean             | 1.000     | 1.000  | 1.000    | 117     |
# | Dusty             | 0.974     | 0.974  | 0.974    | 117     |
# | Electrical-damage | 1.000     | 1.000  | 1.000    | 117     |
# | Physical-Damage   | 1.000     | 0.983  | 0.991    | 117     |
# | Snow-Covered      | 1.000     | 1.000  | 1.000    | 117     |

# Summary:
# - The model achieves high precision, recall, and F1 across all defect classes.
# - Accuracy is 98.86% on the test set of 702 images.
# - The system is reliable for real-world deployment in solar panel inspections.

# 📘 Loss Function & Optimizer

# How the model learns and adapts during training

# 🔹 Loss Function – Categorical Crossentropy
# Measures the difference between predicted class probabilities and actual labels.
# Ideal for multi-class classification problems like this one.
# It penalizes incorrect predictions more heavily the more confident the model was.

# Example:
# - True label: [0, 0, 1, 0, 0, 0]
# - Predicted: [0.1, 0.1, 0.2, 0.3, 0.2, 0.1] → high loss (wrong class, low confidence)
# - Predicted: [0.0, 0.0, 0.9, 0.1, 0.0, 0.0] → low loss (correct class, high confidence)

# Final Test Accuracy  : 98.74%
# Final Test Loss      : 0.4517
# Early Stopping Triggered at Epoch 22 (Best Epoch: 16)

# The model showed excellent convergence with strong generalization and minimal overfitting.
# Validation stopped automatically using EarlyStopping to retain the best-performing weights.

# 🔹 Optimizer – Adam (Adaptive Moment Estimation)
# Combines momentum and adaptive learning rates.
# Automatically adjusts learning rate per parameter using gradient history.
# Great for noisy or sparse datasets like solar panel images.

# 🔹 Why We Used Adam:
# - Minimal manual tuning
# - High performance on image tasks
# - Works well with learning rate scheduling (e.g., ReduceLROnPlateau)

# 🔹 Learning Rate
# Controls how quickly model updates weights during training.
# ReduceLROnPlateau was used to dynamically lower the learning rate when validation performance plateaued.

# Summary
# - Categorical crossentropy is ideal for multi-class problems.
# - Adam optimizer ensured fast and stable convergence.
# - Final test loss of 0.4517 and accuracy of 98.74% confirms effective optimization.


# base_model.trainable = True

# 🔹 What is base_model.trainable = True?
# In transfer learning, we load a pretrained model like EfficientNet that already learned general features 
   # from a large dataset like ImageNet.
# Initially, we keep these pretrained layers "frozen" by setting base_model.trainable = False, 
   # so only the new layers we added (like Dense) are trained.

# Later, we set base_model.trainable = True - this is called fine-tuning.
# It means we allow the weights of the pretrained base model (even the early layers) to get updated during training.

# 🔹 Why do we unfreeze the base model?
# The features learned on ImageNet (like cats, trees, cars) are very generic.
# But our dataset — solar panel defects — is very specific.
# So unfreezing the base layers helps the model adjust those filters to better detect things like dust, burn marks, 
   # or snow-covered panels.

# 🔹 When should we apply it?
# 1. First freeze the base model (trainable = False) and train only the new layers (Dense).
# 2. After a few epochs, if validation loss is stable and there's no overfitting — then unfreeze (trainable = True) 
     # and continue training the full model.

# This avoids "catastrophic forgetting" and gives better results.

# 📘 What is Catastrophic Forgetting?

# Catastrophic forgetting happens when a neural network forgets previously learned knowledge 
  # after being trained on new data.

# In simple terms:
# - The model was good at one task.
# - You fine-tuned it on a new task.
# - Now it performs well on the new task but completely "forgets" how to do the old one.

# Why does it happen?
# - Neural networks update weights during training.
# - If we fine-tune the entire model (especially the pretrained base) without care,
#   the new weight updates can override all the old useful features learned from the original data (like ImageNet).

# In transfer learning:
# - If you immediately set base_model.trainable = True and train from the start,
#   the model may lose its **general feature extraction abilities**.
# - It forgets what it learned from the huge original dataset.

# To avoid catastrophic forgetting:
# - First freeze the base model → train the new layers only.
# - Later, unfreeze and fine-tune carefully using a low learning rate.
# - Monitor validation loss — stop if it increases sharply.

# In this project:
# - We avoided catastrophic forgetting by freezing EfficientNet at first.
# - Then, we fine-tuned only after the classifier was trained.


# 🔹 What happens internally?
# - When trainable = False: no gradient updates for the base model, only new layers get trained.
# - When trainable = True : all layers participate in training, including the initial convolutional filters.

# 🔹 What if we don't fine-tune?
# The model may rely only on generic patterns, which might not be enough for our use-case.
# Fine-tuning helps the model specialize and improve accuracy — which is exactly what we observed in this project.

# 🔹 Is there any risk?
# Yes - If the dataset is small or noisy, unfreezing too many layers too soon can cause overfitting.
# So it should be done after monitoring validation accuracy/loss trends.

# 🔹 In this project:
# - We used EfficientNet as base_model
# - Initially froze it while training the Dense layers
# - Later set base_model.trainable = True for fine-tuning
# - This helped improve accuracy and adapt the model to solar panel images

# Summary:
# - trainable = True enables fine-tuning the entire network
# - Helps the model adapt to domain-specific features
# - Works best when applied carefully after freezing phase

# 📘 Callbacks & Training Control

# 🔹 EarlyStopping
# - Monitors validation loss during training
# - Stops training when the loss doesn’t improve for a certain number of epochs (patience)
# - Prevents overfitting and saves time
# - In this project: 
#     → Used to stop training early (Epoch 22)
#     → Restored best weights from Epoch 16 using restore_best_weights = True

# 🔹 ModelCheckpoint
# - Automatically saves the model during training
# - Saves the “best” model (e.g., lowest validation loss)
# - Helps preserve the best-performing version even if later epochs degrade
# - In this project: Used to store the best EfficientNet model for future inference

# 🔹 ReduceLROnPlateau
# - Monitors validation loss
# - Reduces the learning rate by a factor (e.g., 0.1) if the loss stops improving
# - Helps the model escape plateaus and converge more smoothly
# - In this project: Prevented over-shooting and helped fine-tune weights in later epochs

# 📘 Why Callbacks are Important:
# - Training deep models can lead to overfitting if not monitored
# - Callbacks act as guardrails to:
#     → Stop early if the model stops improving
#     → Save the best model
#     → Adjust learning rate when needed

# Summary:
# - EarlyStopping    : Stops training when there's no improvement
# - ModelCheckpoint  : Saves the best version of the model
# - ReduceLROnPlateau: Dynamically adjusts learning rate during plateaus
# - Together, they ensure efficient and stable model training

# Streamlit Dashboard - Page by Page Overview

# Interface for Real-Time Image Upload, Prediction & Visualization

# 🔹 Home Page (🏠)
# - Provides a welcome message and instructions to the user.
# - Explains what the app does - predicts solar panel defects from uploaded images.
# - Introduces the three available pages: Home, Upload & Predict, Visualize.
# - Simple and user-friendly landing interface.

# 🔹 Upload & Predict Page (📤)
# - Allows users to upload a single image file (e.g., .jpg, .png).
# - The uploaded image is preprocessed to the target input size (300x300).
# - The model (EfficientNet) is loaded using keras.models.load_model().
# - Image is passed through the model to get class probabilities.
# - The predicted class is shown with a label (e.g., “Dusty”, “Clean”, etc.).
# - The prediction is stored in a CSV log file (prediction_results.csv) along with timestamp and filename.

# 🔹 Visualize Page (📊)
# - Reads the prediction_results.csv file to generate visual insights.
# - Displays charts to show:
#     → Bar chart : Count of predictions per defect type
#     → Pie chart : Proportion of each defect category
#     → Line chart: Predictions over time (date-wise trend)
# - Helps identify the most frequent defects and temporal patterns.

# Additional Streamlit Functionalities

# - Sidebar for navigation between pages
# - Image preview before prediction
# - Real-time updates to prediction logs
# - Clear layout using columns, markdown, and styling

# Summary
# - Home            : Overview and guidance
# - Upload & Predict: Inference and logging
# - Visualize       : Analytical dashboard of predictions
# - Together, these pages turn a trained DL model into an interactive, deployable application

#📘 Real-Time Prediction Logging + Output CSV Explanation

# Tracks user predictions and builds data for analysis

# 🔹 Purpose:
# - Every time a user uploads an image for prediction, the app logs:
#     → Image name
#     → Predicted class label (e.g., Burn, Dusty)
#     → Date & time of prediction
# - This is saved into a CSV file called: prediction_results.csv

# 🔹 File Location & Format:
# - File   : ./prediction_results.csv
# - Columns: ['Image Name', 'Prediction', 'Date']

# 🔹 Code Logic:
# - After the prediction is made using the model, this snippet is used:  
#   df = pd.DataFrame([[img_name, prediction_label, datetime.now().date()]],
#                     columns=["Image Name", "Prediction", "Date"])
#   df.to_csv("prediction_results.csv", mode='a', header=not os.path.exists("prediction_results.csv"), index=False)


# 🔹 Why This Matters:
# - Logs build a mini dataset for ongoing analysis
# - These logs are later used in the 📊 Visualize page to:
#     → Track which defect types are common
#     → Monitor predictions over time
#     → Display charts for inspection trends

# 🔹 Benefits:
# - Adds transparency: users can see a history of all predictions
# - Enables future analysis: seasonality, fault frequency, hotspot identification
# - CSV format allows easy integration with external tools (Excel, Power BI, etc.)


# Summary:
# - prediction_results.csv acts as a running log of all predictions
# - It powers the Visualize dashboard and adds analytics capability
# - This real-time tracking makes the model deployable in operational environments

# 📘 Final Summary & Deployment Notes

# 🔹 Project Overview:
# - This project builds a deep learning model to classify solar panel defects using image data.
# - The model identifies classes like: Dusty, Bird-drop, Electrical Damage, Physical Damage, Snow-Covered, and Clean.
# - The final solution is deployed as a real-time prediction web app using Streamlit.

# 🔹 Training Pipeline Summary:
# - Dataset       : Class-wise folders containing labeled solar panel images
# - Preprocessing : Resized to 300x300, normalized, and augmented
# - Model         : Transfer learning with EfficientNetB0/B3, softmax output
# - Loss          : Categorical Crossentropy
# - Optimizer     : Adam
# - Evaluation    : Accuracy, Precision, Recall, F1-score, Confusion Matrix
# - Control       : Callbacks like EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# - Final Accuracy: ✅ 98.86% on test set (702 images)

# 🔹 Deployment Pipeline:
# - The trained Keras model (.h5 or .keras) is loaded into a Streamlit app
# - Users upload images directly via the interface
# - Model predictions are displayed along with uploaded image
# - Predictions are logged into a CSV file (prediction_results.csv)
# - A dashboard visualizes trends and defect frequencies using charts

# 🔹 Features of the Streamlit App:
# - 🏠 Home            : Overview and guidance
# - 📤 Upload & Predict: Image classification and logging
# - 📊 Visualize       : Bar, pie, and line charts for defect trends
# - 🔄 Real-time CSV logging and auto-update charts

# 🔹 Benefits of the System:

# 🔹 Lightweight and Fast  
# - The model uses EfficientNetB0/B3 — a compact and high-accuracy CNN that provides fast inference on standard hardware.

# 🔹 Real-Time Image Prediction  
# - Users can upload an image of a solar panel and receive instant defect classification through a pre-trained Keras model.

# 🔹 Interactive Web Deployment (Streamlit)  
# - The application is deployed via Streamlit, allowing interactive usage without command-line or notebook interfaces.

# 🔹 Auto Logging of Predictions (CSV-based)  
# - Each prediction (image name, predicted class, date) is logged into a CSV file (`prediction_results.csv`) for traceability and visualization.

# 🔹 Visual Analytics Dashboard  
# - The "📊 Visualize" page uses Plotly to show interactive charts: 
#     → Bar chart for defect type frequency  
#     → Pie chart for category proportions  
#     → Line chart for daily prediction activity

# 🔹 Modular Page-Based Interface  
# - Streamlit sidebar provides navigation between clearly defined pages: 🏠 Home, 📤 Upload & Predict, 📊 Visualize.

# 🔹 Model Training with Fine-Tuning  
# - The model was built using transfer learning with EfficientNet, and further fine-tuned by setting `base_model.trainable = True`.

# 🔹 Performance Monitoring via Callbacks  
# - Callbacks such as `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` were used to prevent overfitting and optimize training.

# 🔹 High Model Accuracy & Reliability  
# - The model achieved a final test accuracy of 98.86% with strong precision, recall, and F1-score across all defect classes.

# 🔹 Clean Architecture and Reusability  
# - The codebase is modular, annotated, and organized for reusability — making future enhancements easy to integrate.


# 🔹 Final Notes:
# - The model performs reliably across all classes with near-perfect F1 scores.
# - The project showcases an end-to-end deep learning workflow: data → training → evaluation → deployment via Streamlit.
# - This framework is production-ready and can be scaled for real-world solar inspection systems with minimal adaptation.
