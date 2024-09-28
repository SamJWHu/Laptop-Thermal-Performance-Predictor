# Laptop-Thermal-Performance-Predictor
Overview
The Laptop Thermal Performance Predictor is a machine learning project designed to analyze and predict the thermal behavior of various laptop models based on their hardware and cooling system specifications. By leveraging a comprehensive dataset and a Physics-Informed Neural Network (PINN), this project aims to provide insights into how different components influence a laptop's cooling efficiency, fan speed, temperature, and noise levels during high-performance tasks.

Features
Comprehensive Dataset: Includes detailed specifications of laptops from major brands such as Apple, HP, Dell, Lenovo, ASUS, and Acer.
Feature Engineering: Incorporates both hardware components (e.g., CPU/GPU TDP, chassis dimensions) and cooling system details (e.g., number of heat pipes, fan specifications).
Physics-Informed Neural Network (PINN): Integrates physical constraints into the neural network training process to enhance prediction accuracy and reliability.
Data Augmentation: Expands the dataset with realistic variations to improve model generalization and robustness.
Visualization Tools: Provides correlation matrices and box plots to analyze feature relationships and data distributions.
Dataset
The dataset comprises specifications of 23 laptop models, expanded with variations to increase dataset size and diversity. Key features include:

Input Features
cpu_tdp: CPU Thermal Design Power (W)
gpu_tdp: GPU Thermal Design Power (W)
total_tdp: Sum of CPU and GPU TDP (W)
cooling_capacity: Sum of heat pipes and cooling fans
chassis_thermal_conductivity: Thermal conductivity of the chassis material (W/m·K)
vent_area: Ventilation area (cm²)
fan_blade_count: Number of blades per fan
ambient_temp: Ambient temperature (°C)
heat_pipe_material: Thermal conductivity of heat pipe material (W/m·K)
fan_diameter: Diameter of cooling fans (mm)
Output Features
fan_speed_turbo: Fan speed in turbo mode (RPM)
temp_turbo: Surface temperature in turbo mode (°C)
fan_noise_turbo: Fan noise level in turbo mode (dB)
Model
The project utilizes a Physics-Informed Neural Network (PINN) built with TensorFlow and Keras. The model architecture includes multiple dense layers with ReLU activation, L2 regularization, and dropout to prevent overfitting. The loss function combines mean squared error (MSE) with physics-based constraints to ensure realistic predictions.

Training Parameters
Optimizer: Adam with a learning rate of 0.001
Batch Size: 16
Epochs: 500 (with early stopping based on validation loss)
Loss Function: Data loss (MSE) + Physics loss (weighted by alpha = 0.01)
Early Stopping: Monitors validation loss with a patience of 20 epochs
Usage

1. Clone the Repository
git clone https://github.com/SamJWHu/laptop-thermal-predictor.git
cd laptop-thermal-predictor

3. Install Dependencies
Ensure you have Python installed. Then, install the required packages:
pip install -r requirements.txt

5. Prepare the Dataset
Run the data preparation script to build and augment the dataset:
python prepare_data.py

6. Train the Model
Execute the training script to train the PINN model:
python train_model.py

7. Make Predictions
Use the prediction script to input new laptop specifications and receive thermal performance predictions:
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model and scalers
model = load_model('model.h5', custom_objects={'physics_loss': physics_loss})
input_scaler = StandardScaler()
output_scaler = StandardScaler()

# Fit scalers on training data (ensure this step is consistent with training)
# X_train and y_train should be your training datasets
input_scaler.fit(X_train[input_features])
output_scaler.fit(y_train[output_features])

# Define new laptop specifications
new_design = pd.DataFrame({
    'cpu_tdp': [35],
    'gpu_tdp': [85],
    'total_tdp': [35 + 85],
    'cooling_capacity': [3 + 2],
    'chassis_thermal_conductivity': [150],  # W/m·K
    'vent_area': [100],                     # cm²
    'fan_blade_count': [9],
    'ambient_temp': [25],                   # °C
    'heat_pipe_material': [401],            # W/m·K
    'fan_diameter': [60]                    # mm
})

# Normalize the input
new_design_scaled = input_scaler.transform(new_design[input_features])

# Predict and inverse transform the outputs
new_prediction_scaled = model.predict(new_design_scaled)
new_prediction = output_scaler.inverse_transform(new_prediction_scaled)

# Display the predictions
prediction_dict = dict(zip(output_features, new_prediction[0]))
for key, value in prediction_dict.items():
    print(f'{key}: {value:.2f}')

Visualization
The project includes scripts to visualize data distributions and feature correlations:

Correlation Matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = data[input_features + output_features].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

Box Plots
# Box plots for input features
data[input_features].plot(kind='box', subplots=True, layout=(2, 5), figsize=(15, 8))
plt.tight_layout()
plt.show()

# Box plots for output features
data[output_features].plot(kind='box', subplots=True, layout=(1, 3), figsize=(15, 4))
plt.tight_layout()
plt.show()


Contributing
Contributions are welcome! Please fork the repository and submit a pull request for enhancements or bug fixes.

License
This project is licensed under the MIT License.

Contact
For any inquiries or support, please contact r02522318@gmail.com
