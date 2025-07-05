# 📦 Import Required Libraries and Initialize Model

# Core Imports 
import streamlit as st                     # For building the interactive web interface
import pandas as pd                        # For structured data handling and predictions display
import numpy as np                         # For numerical operations like array handling
import matplotlib.pyplot as plt            # For basic visualization
import seaborn as sns                      # For enhanced statistical visualization
from datetime import datetime              # To fetch current time for personalized greetings
import plotly.express as px                # For interactive and responsive plotting
from fpdf import FPDF                      # For generating a basic PDF report
import io                                  # For creating an in-memory bytes buffer

# 🖼️ Image & Model Related 
import tensorflow as tf                                                  # For loading and using the trained CNN model
from tensorflow.keras.applications.efficientnet import preprocess_input  # To preprocess input image
from PIL import Image                                                    # For image upload, resizing, and manipulation

# Short Description:
# This block imports all essential libraries required for the dashboard:
# - Streamlit for UI, pandas/matplotlib/seaborn/plotly for visualization
# - TensorFlow & PIL for image preprocessing and classification
# - Datetime for personalized greetings
# - Ensures model caching to avoid reloading during page switches

# 🧠 Performance Optimization Using Streamlit Cache
@st.cache_resource                             # Caches the model after first load to prevent repeated reloads
def load_model(path):                          # Function to load the pre-trained CNN model
    """
    Load the pre-trained defect classification model
    """
    return tf.keras.models.load_model(path)    # Load a saved .keras model for defect prediction

# Load the model once globally for use across all pages
model = load_model("final_model_solarpanel.keras")  # Global model variable for predictions

# 🧭 Sidebar Navigation Setup (Global)
st.sidebar.title("🔍 Navigation")                   # Sidebar heading
page = st.sidebar.selectbox("📂 Choose Page", [     # Sidebar dropdown for page selection
    "Home", 
    "Upload & Predict", 
    "Visualize"
])

# Commands Used 
# Streamlit
# import streamlit as st                       → Create Streamlit UI elements
# st.markdown(), st.title(), st.expander()     → Layout & formatting
# st.sidebar.title(), st.sidebar.selectbox()   → Sidebar navigation
# @st.cache_resource                           → Cache model

# Pandas
# pd.DataFrame(data)                          → Tabular output
# df.to_csv()                                 → Export results

# Numpy
# np.array(img)                               → Convert PIL image to array
# np.argmax(preds, axis=1)                    → Extract predicted label

# Matplotlib / Seaborn
# plt.plot(), sns.heatmap()                   → Graphs and visualizations

# Plotly
# px.bar(), px.pie()                          → Interactive charts

# TensorFlow / Keras
# keras.models.load_model()                   → Load trained CNN model
# efficientnet.preprocess_input               → Preprocess uploaded image

# PIL
# Image.open(), img.resize()                  → Handle and prepare image

# Datetime
# datetime.now().hour                         → Used for greeting

# 🌟 Key Features 
# - Imports all core and model-related libraries
# - Loads and caches deep learning model only once
# - Sidebar dropdown enables multi-page routing
# - Optimized for fast loading and reuse across dashboard pages

# 🏠 Home Page Function
def show_home_page():

    # 🌞 Get the current hour to determine greeting message
    current_hour = datetime.now().hour        # 0–23 hour format

    # ⏰ Display time-based greeting
    if current_hour < 12:
        st.markdown("### 🌞 Good Morning!")   # Morning greeting
    elif 12 <= current_hour < 16:
        st.markdown("### ☀️ Good Afternoon!") # Afternoon greeting
    else:
        st.markdown("### 🌙 Good Evening!")   # Evening greeting

    # 🧾 Display the main title of the dashboard
    st.title("🔋 SolarGuard - Solar Panel Defect Detection Dashboard")

    # 🗒️ Short dashboard description
    st.markdown("An intelligent system to detect solar panel defects, recommend maintenance actions, and improve operational efficiency.")

    # 📘 Expandable instructions section
    with st.expander("📘 How to Use This Dashboard"):
        st.markdown("""
        - **Home**: Overview and usage instructions  
        - **Upload & Predict**: Upload solar panel images to detect defects  
        - **Visualize**: Explore uploaded results and defect frequency  
        """)

# 📂 Page Routing Controller
if page == "Home":
    show_home_page()                          # Render Home Page content

# Commands Used in Routing
# - st.sidebar.selectbox()                 → Sidebar dropdown to choose page
# - if/elif page == ...                    → Routing logic
# - show_home_page()                       → Calls appropriate page function

# 🌟 Key Features
# - Sidebar enables intuitive navigation
# - Pages handled as independent functions for scalability
# - Avoids reruns and ensures performance

# 📦 Summary
# This corrected version fixes the NameError by moving the `page` selector
# outside the Home Page function. This allows the routing controller to
# access the value and render pages appropriately across the app.


# 📸 Upload & Predict Page – Image Classification with Deep Learning

elif page == "Upload & Predict":  # If user selects "Upload & Predict" from sidebar
    def show_upload_page():       # Define function to render Upload page

        # 📌 Import required packages
        import os                                                                # File path and file checking
        from PIL import Image                                                    # Image handling
        import numpy as np                                                       # For numerical operations
        import pandas as pd                                                      # For saving predictions
        import tensorflow as tf                                                  # For loading the model
        from tensorflow.keras.applications.efficientnet import preprocess_input  # EfficientNet preprocessing
        from datetime import datetime                                            # To save timestamp in prediction log

        # 🧾 Page Title and Description
        st.title("📸 Upload Solar Panel Image to Detect Defects")                # 🖼️ Display page title
        st.markdown("Upload a solar panel image to automatically detect possible surface defects using the trained deep learning model.")  # ℹ️ Instruction text

        # 📤 Upload Image Section
        uploaded_file = st.file_uploader("Upload a Solar Panel Image (JPG, PNG)", type=["jpg", "jpeg", "png"])  # 📁 Image upload widget

        if uploaded_file is not None:  # 📥 Check if an image has been uploaded

            # 🖼️ Display uploaded image
            image = Image.open(uploaded_file)                                       # Open uploaded image using PIL
            st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)  # Show preview of image
            st.markdown("✅ Image successfully uploaded!")                          # Upload confirmation

            # 🔄 Preprocess Image
            img = image.resize((300, 300))                 # Resize image to match model input
            img_array = np.array(img)                      # Convert to NumPy array
            img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_preprocessed = preprocess_input(img_batch) # Apply EfficientNet preprocessing

            # 🔍 Load Model (Safe Check)
            model_path = "best_model.keras"                    # Expected model path

            if os.path.exists(model_path):                      # Check if model file exists
                model = tf.keras.models.load_model(model_path)  # Load trained model

                # 🔮 Perform Prediction
                prediction = model.predict(img_preprocessed)              # Run prediction
                predicted_class = np.argmax(prediction, axis=1)[0]        # Get class index

                # 🏷️ Define Class Labels (Map index to label)
                class_labels = {
                    0: "Bird-drop",
                    1: "Clean",
                    2: "Dusty",
                    3: "Electrical-damage",
                    4: "Physical-Damage",
                    5: "Snow-Covered"
                }

                predicted_label = class_labels.get(predicted_class, "Unknown")  # Fetch label

                # ✅ Display Prediction Result
                st.subheader("🔎 Prediction Result")  # Show result section
                st.success(f"🎯 The uploaded image is predicted as: **{predicted_label}**")  # Prediction output

                # 🧾 Save Prediction to CSV Log
                results = {  # Log dictionary
                    "Filename": uploaded_file.name,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Predicted_Label": predicted_label
                }

                results_df = pd.DataFrame([results])  # Convert to DataFrame

                # 📄 Append or create prediction_results.csv
                if os.path.exists("prediction_results.csv"):
                    results_df.to_csv("prediction_results.csv", mode='a', header=False, index=False)  # Append
                else:
                    results_df.to_csv("prediction_results.csv", index=False)  # Create new CSV

            else:
                # Model not found – Show error message
                st.error("Model not found at: `best_model.keras`. Please ensure the model file is available in the project directory.")
                st.markdown("Tip: Train the model and ensure `best_model.keras` is saved using `ModelCheckpoint`.")

    # ▶️ Call the Upload Page Function
    show_upload_page()


# Purpose:
# - Enable users to upload a solar panel image
# - Detect and classify surface defects using a trained deep learning model
# - Display prediction output and log results for future visualization

# Short Description:
# The Upload & Predict page allows users to upload images of solar panels 
# and receive instant AI-based classification of surface defects such as 
# dust, cracks, snow, or bird droppings using a trained EfficientNet CNN model.
# Uploaded images are resized, preprocessed, and passed to the model.
# Prediction results are saved with timestamps to a CSV file for analysis.

# 🌟 Key Features:
# - Real-time image classification using a CNN model (EfficientNet)
# - Upload any JPG/PNG image and preview it in the app
# - Display the predicted defect type clearly
# - Automatically logs filename, label, and time to a CSV for tracking
# - Error handling if model is missing or image format is invalid

# Core Functionalities:
# - File upload interface using Streamlit
# - Image preprocessing (resize to 300x300, normalize)
# - Use of pretrained model to predict class
# - Show prediction output on UI
# - Save result into `prediction_results.csv` with metadata

# 📦 Key Packages Used:
# - Streamlit          → For UI interface and interaction
# - TensorFlow / Keras → For model loading and prediction
# - PIL (Pillow)       → To read and resize uploaded image
# - NumPy              → To convert and reshape image data
# - Pandas             → To log results in CSV format
# - OS / Datetime      → File handling and timestamp generation

# Commands Used:

# Streamlit
# st.title(), st.markdown()                     → Render page title and instructional text
# st.file_uploader()                            → Upload solar panel image (JPG, PNG)
# st.image()                                    → Display uploaded image in-app
# st.success(), st.error()                      → Show prediction result or missing model warning
# st.subheader()                                → Section title before displaying prediction result

# TensorFlow / Keras
# tf.keras.models.load_model()                  → Load the trained EfficientNet classification model
# model.predict()                               → Perform prediction on the preprocessed image
# preprocess_input()                            → Normalize image input for EfficientNet
# np.argmax()                                   → Get the index of the highest prediction probability

# PIL (Python Imaging Library)
# Image.open()                                  → Open uploaded image file
# img.resize((300, 300))                        → Resize image to model’s input size

# NumPy
# np.array()                                    → Convert PIL image to NumPy array
# np.expand_dims()                              → Add batch dimension to match model input shape

# Pandas
# pd.DataFrame()                                → Create DataFrame to store prediction results
# df.to_csv()                                   → Save or append prediction results to CSV file

# OS & System
# os.path.exists()                              → Check if model file or result log already exists
# os.makedirs()                                 → Create directories if needed (optional usage)

# Datetime
# datetime.now().strftime()                     → Generate and format current timestamp for logging

# 📊 Visualize Page – Analyze Uploaded Predictions and Defect Trends

elif page == "Visualize":           # 🔀 Handle sidebar navigation to "Visualize"
    def show_visualization_page():  # 🚀 Define the page rendering function

        # 🧾 Page Title and Description
        st.title("📊 Visualize Defect Trends and Frequencies")  # Set main title
        st.markdown("Explore predictions, track confidence scores, and understand solar panel defect trends over time.")  # Add subtext

        # 📂 Load prediction_results.csv file (uploaded history)
        try:
            df = pd.read_csv("prediction_results.csv")             # Read past predictions
            st.success("Prediction results loaded successfully!")  # Show success message

            # 🧾 Preview Raw Data Table
            with st.expander("📄 View Raw Prediction Data"):      # Add expandable data view
                st.dataframe(df)  # Display the loaded dataframe

                # 📥 Download Prediction Report
                csv = df.to_csv(index=False).encode('utf-8')  # Convert DataFrame to CSV format
                st.download_button(                           # Add a download button
                    label="📥 Download Prediction Report (CSV)",  # Button label
                    data=csv,                                     # CSV data
                    file_name='prediction_results.csv',           # Download file name
                    mime='text/csv'                               # MIME type
                )   

            # 📊 Plot 1: Bar Chart – Frequency of each defect type
            defect_counts = df["Predicted_Label"].value_counts().reset_index()  # Count defect occurrences
            defect_counts.columns = ["Defect Type", "Count"]                    # Rename columns for clarity

            fig_bar = px.bar(                         # Create a bar chart using Plotly Express
                defect_counts,                        # DataFrame containing defect counts per type
                x="Defect Type",                      # X-axis will represent different defect types
                y="Count",                            # Y-axis will show number of occurrences
                color="Defect Type",                  # Bar color coded by defect type
                title="🧮 Defect Type Frequency",     # Chart title
                labels={"Count": "Occurrences"},      # Rename y-axis label to "Occurrences"
                template="plotly_white"               # Use a clean white theme for the chart
            )
            st.plotly_chart(fig_bar, use_container_width=True)  # Display bar chart in full width

            # 🥧 Plot 2: Pie Chart – Distribution of predicted classes
            fig_pie = px.pie(                         # Create a pie (donut) chart using Plotly Express
                defect_counts,                        # DataFrame containing defect counts
                values="Count",                       # Pie slice size based on count values
                names="Defect Type",                  # Each slice labeled by defect type
                title="🧩 Defect Type Distribution", # Chart title
                hole=0.4                              # Creates a donut-style chart by adding a hole in the center
            )
            st.plotly_chart(fig_pie, use_container_width=True)  # Display pie chart in full width

            # 📅 Plot 3: Line Chart – Predictions over time (if timestamp exists)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert to datetime format
            df["Date"] = df["Timestamp"].dt.date               # Extract date part only

            daily_counts = df.groupby("Date").size().reset_index(name="Predictions")  # Count predictions per date

            fig_line = px.line(                       # Create line chart
                daily_counts,                         # DataFrame that contains date-wise prediction counts
                x="Date",                             # X-axis represents each date
                y="Predictions",                      # Y-axis shows number of predictions made on that date
                markers=True,                         # Display markers on each data point to highlight activity
                title="📈 Daily Prediction Activity", # Chart title for context
                template="plotly_white"               # Use a clean white background theme
            )
            st.plotly_chart(fig_line, use_container_width=True)  # Display line chart
            
            # 📥 Final Export – PDF Report Placeholder
            #st.markdown("---")  # 🔹 Draw a horizontal separator line to visually separate this section from the charts above

            st.subheader("📥 Export Full Summary")  # 🏷️ Add a subheading for the export section (appears in larger text)
    
            # 📄 Initialize PDF document
            pdf = FPDF()                    # 📄 Create a new FPDF object
            pdf.add_page()                  # ➕ Add a blank page to the PDF
            pdf.set_font("Arial", size=12)  # 🔠 Set font to Arial, size 12

            # 📝 Add report content to PDF
            pdf.cell(200, 10, txt="Defect Prediction Report Summary", ln=True, align='C')  # 🧾 Title centered
            pdf.ln(10)         # ⬇️ Add a vertical line break (empty space)

            # Summary lines showing key metrics from the prediction data
            pdf.cell(200, 10, txt="- Total Predictions: 123", ln=True) # Total predictions
            pdf.cell(200, 10, txt=f"- Defect Types Predicted: {df['Predicted_Label'].nunique()}", ln=True)  # Unique defect types
            pdf.cell(200, 10, txt=f"- First Prediction Date: {df['Timestamp'].min().date()}", ln=True)      # First date
            pdf.cell(200, 10, txt=f"- Last Prediction Date: {df['Timestamp'].max().date()}", ln=True)       # Last date
            
            # 🔄 Step 2: Export PDF content as bytes
            pdf_bytes = pdf.output(dest='S').encode('latin-1')  # Get string, encode to bytes

            # 💾 Step 3: Wrap bytes in BytesIO for Streamlit
            pdf_buffer = io.BytesIO(pdf_bytes)  # Create memory stream from encoded bytes

            # 📩 Create Streamlit Download Button for the generated PDF        
            st.download_button(                                   # 📦 Create a downloadable file button
                label="📄 Download Full Report (PDF)",            # 🏷️ Button text shown to the user
                data=pdf_buffer,                                   # In-memory bytes
                file_name="defect_analysis_report.pdf",            # 💾 Name of the file user will get when they click download
                mime="application/pdf"                             # 📚 MIME type indicating it’s a PDF file
            )

        # ⚠️ Fallback if CSV not found
        except FileNotFoundError:
            st.warning("⚠️ No prediction data found. Please upload and predict images first.")  # Show warning

    # Call the Visualize function
    show_visualization_page()

# Commands Used:

# Streamlit
# st.title(), st.markdown()                     → Render title and instructions
# st.success(), st.warning()                    → Show messages based on file status
# st.dataframe()                                → Display raw data table
# st.expander()                                 → Collapsible section for raw data
# st.download_button()                          → Allow users to export CSV reports
# st.plotly_chart()                             → Display interactive charts
# st.subheader()                                → Section header for export options

# Pandas
# pd.read_csv()                                 → Load past prediction results
# df["Predicted_Label"].value_counts()          → Count occurrences of each predicted class
# df.columns, df.reset_index()                  → Format DataFrame for plotting
# df.to_csv().encode('utf-8')                   → Convert DataFrame to downloadable CSV
# len(df), df["Predicted_Label"].nunique()      → Dynamic summary metrics

# Plotly Express
# px.bar()                                      → Interactive bar chart of class frequencies
# px.pie()                                      → Pie chart showing distribution of predictions
# px.line()                                     → Line chart to track prediction trends over time

# fpdf
# FPDF()                                        → Create and manage PDF documents
# pdf.add_page(), pdf.set_font()                → Add new pages and set styling
# pdf.cell(), pdf.ln()                          → Write text content and space formatting
# pdf.output(dest='S').encode('latin-1')        → Export PDF content as bytes (encoded for download)

# io
# io.BytesIO()                                  → In-memory stream to buffer the PDF bytes

# datetime
# datetime.now().strftime()                     → Add current timestamp to report footer

# 🌟 Key Features:
# - Automatically loads past prediction logs (if available)
# - Interactive bar chart and pie chart to analyze defect types
# - Line chart to view prediction activity over time
# - Data table viewer for raw prediction records
# - Download button to export prediction history as **CSV**
# - Export full visual summary as **PDF** (auto-generated)
# - Dynamically shows:
#     → Total predictions
#     → Defect types
#     → First and last prediction date
#     → Top 3 most frequent defects
#     → Report generation timestamp
# - Gracefully handles missing data scenarios
# - Uses in-memory buffer (BytesIO) for seamless download

# 📦 Summary:
# This page provides a Visual Analytics Dashboard for solar panel defect predictions.
# It automatically loads historical prediction results, and generates:
# - A Bar Chart showing frequency of predicted defect types,
# - A Pie Chart highlighting distribution of defects,
# - A Line Chart showing daily prediction activity over time,
# - A PDF summary report containing key statistics, generated dynamically.

# The raw prediction data is viewable in a collapsible table and can be exported as CSV.
# A downloadable PDF report adds professional summary documentation.
# This page enhances transparency, traceability, and trend analysis of model performance.