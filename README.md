# Wound Classification Using Deep Learning

## 🔍 Overview
This project focuses on classifying different types of wounds using deep learning techniques. It aims to aid medical diagnostics by providing automated wound classification based on image data. The project leverages state-of-the-art deep learning models and image-processing techniques to achieve high accuracy and efficiency.

## 🎯 Project Motivation
Wound classification is a critical component of medical diagnostics, aiding healthcare professionals in identifying and treating wounds effectively. Manual wound assessment can be time-consuming, subjective, and prone to human error, especially in resource-constrained environments. 

This project aims to address these challenges by leveraging deep learning techniques to automate wound classification, ensuring accuracy, consistency, and scalability. By building a robust and efficient classification system, this project seeks to assist medical practitioners in providing timely and precise care, ultimately improving patient outcomes and supporting the adoption of AI-driven healthcare solutions.

## 📂 Dataset
The project uses publicly available wound image datasets and custom-curated data for classification. The primary dataset used in this project can be accessed at: [Wound Classification Dataset](https://www.kaggle.com/datasets/ibrahimfateen/wound-classification)

The dataset contains a total of 2940 wound images categorized into 10 distinct classes. Below is an overview of the dataset distribution:

#### 🧮 Class Distribution
The dataset includes the following categories with their respective sample counts:
- **Pressure Wounds**: 602 images
- **Venous Wounds**: 494 images
- **Diabetic Wounds**: 462 images
- **Surgical Wounds**: 420 images
- **Bruises**: 242 images
- **Normal**: 200 images
- **Abrasions**: 164 images
- **Burns**: 134 images
- **Laceration**: 122 images
- **Cut**: 100 images

#### 🔎 Observations
1. **Class Imbalance**:
   - The dataset suffers from a significant class imbalance, as evidenced by the varying sample sizes across the classes. For example:
     - **Pressure Wounds** have the highest count (602 images).
     - **Cut** has the lowest count (100 images).
   - This imbalance could potentially bias the model towards classes with higher sample counts. Techniques like oversampling, undersampling, or data augmentation can be applied to mitigate this issue.

2. **Image Resolution and Aspect Ratio**:
   - The images in the dataset have varying resolutions and aspect ratios, which may introduce inconsistencies during model training.
   - To address this, preprocessing steps such as resizing and normalization are applied to standardize the images.

#### 🛠️ Preprocessing Strategies
To handle the above challenges, the following preprocessing techniques were used:
- **Image Resizing**: All images were resized to a uniform resolution to ensure consistency in input dimensions for the model.
- **Data Augmentation**:
  - Techniques like rotation, flipping, and shifting adjustments were applied to increase the diversity of training data, especially for underrepresented classes.

## 🧠 Model Architecture
The model is a custom Convolutional Neural Network (CNN) designed for wound classification, built using TensorFlow and Keras. It consists of four convolutional layers with ReLU activation, followed by batch normalization and max pooling layers to extract hierarchical features. A global average pooling layer reduces spatial dimensions before feeding into a dense layer with 128 neurons and L2 regularization, followed by a dropout layer for overfitting prevention. The final softmax layer outputs predictions for all wound classes. The model uses the Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss. It supports callbacks for model checkpointing, early stopping, and learning rate reduction, ensuring optimal training.

## 🤖 Model Training
- The model includes callbacks for:
  - **ModelCheckpoint**: Saves the best model based on validation performance.
  - **EarlyStopping**: Stops training early if the performance stops improving.
  - **ReduceLROnPlateau**: Reduces the learning rate when a metric has stopped improving.

## 🚀 Results
The initial model demonstrated moderate performance, achieving a validation accuracy of **89.80%** and a validation loss of **0.4103**. After enhancements to the architecture and training process, the improved model significantly outperformed the initial one, achieving a validation accuracy of **98.50%** and a reduced validation loss of **0.1061**. This improvement highlights the effectiveness of the modifications in boosting the model's accuracy and generalization capabilities.

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - Deep Learning: TensorFlow (GPU), Keras
  - Data Manipulation: NumPy, Pandas
  - Visualization: Matplotlib
  - Image Processing: OpenCV, PIL

## Development Environment
- **Code Editor**: Visual Studio Code (VS Code)
- **System Specifications**:
  - Processor: Intel® Core™ Ultra 9 185H
  - GPU: NVIDIA® GeForce RTX™ 4060 Laptop GPU
  - Memory: 32 GB LPDDR5X-7467MHz
  - Storage: 1 TB PCIe 4.0 SSD
- **Environment**: Python 3.8+ with TensorFlow GPU
- **Operating System**: Windows 11

## 📋 Instructions to Run the Project

1. **Prepare the Dataset**:
   - Place the dataset files in the `data/` directory.
   - Ensure the dataset path is correctly configured in the project files.

2. **Install Dependencies**:
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\\Scripts\\activate
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run Preprocessing**:
   - Open the `data_processing.ipynb` notebook in Jupyter Notebook or JupyterLab and execute all cells to preprocess the dataset.
   - Alternatively, execute the notebook programmatically:
     ```bash
     jupyter nbconvert --to notebook --execute data_processing.ipynb
     ```

4. **Train the Model**:
   - Open the `custom_cnn.ipynb` notebook in Jupyter Notebook or JupyterLab and execute all cells to train the model.
   - Alternatively, execute the notebook programmatically:
     ```bash
     jupyter nbconvert --to notebook --execute custom_cnn.ipynb
     ```

5. **Evaluate the Model**:
   - Open the `test_model.ipynb` notebook in Jupyter Notebook or JupyterLab and execute all cells to evaluate the model.
   - Alternatively, execute the notebook programmatically:
     ```bash
     jupyter nbconvert --to notebook --execute test_model.ipynb
     ```

6. **View Results**:
   - After evaluation, test the model performance using the `test_model.ipynb` notebook.

These instructions should help you get the project up and running. Let me know if further clarifications are needed!

## 📂 Project Structure
```
.
├── data/                        # Dataset files
│   ├── original                 # Unprocessed data
│   ├── processed                # Standerdized data (image resolution and aspect ratio)
│   ├── balanced                 # Every class has an equal number of samples
│   ├── train                    # Train dataset
│   ├── val                      # Validation dataset
│   ├── test                     # Test dataset
├── src/                         # Source code
│   ├── exploration.ipynb        # Dataset exploration
│   ├── data_processing.ipynb    # Data preprocessing script
│   ├── custom_cnn.ipynb         # Training script
├── models/                      # Saved Deep learning models
│   └── best_cnn_old.h5          # Initial CNN model
│   └── best_cnn.h5              # Improved CNN model
├── tests/                       # Test scripts
│   └── test_model.ipynb         # Test model on new data
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License file
```

## 🤝 Contributions
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact
For questions or collaborations, please contact:
- **LinkedIn**: [Fahim Ahamed](https://www.linkedin.com/in/f-a-tonmoy/)
- **Email**: f.a.tonmoy00@gmail.com
