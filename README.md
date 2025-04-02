# Plant Species Classification using ResNet50  

This repository contains a **Jupyter Notebook** for classifying plant species using **ResNet50**, a powerful deep learning model. The notebook uses **transfer learning** to achieve high accuracy while reducing training time.

---

##  Features  

- **Pretrained ResNet50 Model** – Leverages ImageNet-trained weights for efficient feature extraction.  
- **Dataset Preprocessing** – Includes resizing, normalization, and augmentation techniques.  
- **Fine-Tuning** – Adjusts specific layers of ResNet50 for better classification accuracy.  
- **Training & Evaluation** – Trains the model and analyzes performance using multiple metrics.  
- **Visualization** – Plots learning curves, confusion matrices, and feature maps.  
- **Inference Capability** – Allows users to test the model with new plant images.  

---

##  Detailed Explanation of Each Step  

### 1️ Load and Preprocess Data  

✔ **Load Dataset**  
   - The dataset consists of multiple plant species images.  
   - Each image is labeled according to its species.  
   - Data is loaded using **TensorFlow/Keras ImageDataGenerator** or a custom loader.  

✔ **Resize Images**  
   - ResNet50 requires images of size **224x224 pixels**.  
   - All images are resized to maintain consistency.  

✔ **Normalize Pixel Values**  
   - Each pixel is scaled between **0 and 1** (by dividing by 255).  
   - Normalization helps in faster convergence during training.  

✔ **Data Augmentation**  
   - Random transformations like **rotation, flipping, zooming, and brightness adjustment** are applied.  
   - Augmentation prevents overfitting and improves generalization.  

✔ **Split Dataset**  
   - The dataset is split into **Training, Validation, and Test sets**.  
   - Example:  
     - **80% Training Data**  
     - **10% Validation Data**  
     - **10% Test Data**  

---

### 2️ Set Up ResNet50 Model for Transfer Learning  

✔ **Load Pretrained ResNet50**  
   - Uses Keras' **ResNet50** model with pretrained weights from ImageNet.  
   - Pre-trained models save time and improve accuracy.  

✔ **Remove Fully Connected (FC) Layers**  
   - The original classification layers of ResNet50 are removed.  
   - A **new classification head** is added based on the number of plant species.  

✔ **Freeze Initial Layers**  
   - Freezing prevents the model from modifying pre-trained features.  
   - Typically, the first **80-90% of layers are frozen**, while the last few layers are fine-tuned.  

✔ **Add Custom Fully Connected Layers**  
   - Flatten layer is added.  
   - Dense layers are included with **ReLU activation** for feature learning.  
   - The final layer uses **Softmax activation** for multi-class classification.  

---

### 3️ Fine-Tune the Model  

✔ **Unfreeze Select Layers**  
   - Some deeper layers of ResNet50 are unfrozen for **fine-tuning**.  
   - This allows the model to learn dataset-specific features.  

✔ **Choose an Optimizer**  
   - Uses **Adam** or **SGD (Stochastic Gradient Descent)** with momentum.  
   - Learning rate is adjusted using **learning rate schedulers**.  

✔ **Prevent Overfitting**  
   - **Dropout Layers** are added to prevent overfitting.  
   - **Early Stopping** stops training when validation loss starts increasing.  

---

### 4️ Train and Evaluate Performance  

✔ **Training Process**  
   - The model is trained on **GPU (if available)** for faster computation.  
   - **Batch size and number of epochs** are chosen carefully.  

✔ **Loss Function**  
   - Uses **Categorical Crossentropy** as it’s a multi-class problem.  

✔ **Evaluation Metrics**  
   - **Accuracy**: Measures how well the model classifies plants.  
   - **Loss**: Tracks errors during training and validation.  
   - **Precision & Recall**: Helps in analyzing misclassifications.  

✔ **Validation Set Evaluation**  
   - Compares training vs. validation performance.  
   - Detects overfitting or underfitting issues.  

---

### 5️ Visualize Results  

✔ **Plot Training & Validation Accuracy**  
   - Helps in understanding if the model is improving or overfitting.  

✔ **Plot Training & Validation Loss**  
   - Should decrease over time; an increase indicates overfitting.  

✔ **Confusion Matrix**  
   - Shows the number of correctly/incorrectly classified species.  
   - Helps in analyzing **misclassifications**.  

✔ **Feature Map Visualization** (Optional)  
   - Displays intermediate feature representations learned by the model.  

---

### 6️ Predict on New Images  

✔ **Load a New Image**  
   - An unseen plant image is loaded for testing.  

✔ **Preprocess the Image**  
   - Resized to **224x224 pixels**.  
   - Normalized before feeding into the model.  

✔ **Run Through the Model**  
   - The trained model predicts the species based on learned features.  

✔ **Output the Prediction**  
   - The model returns the predicted species along with **confidence scores**.  

---
