# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 10:13:17 2024

@author: SamJWHu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Initialize an empty list to store laptop data
laptops = []


# Updated function to add a laptop entry with additional features
def add_laptop(brand, model, cpu_tdp, gpu_tdp, chassis_thickness, chassis_volume,
               num_heat_pipes, heat_pipe_length, num_cooling_fans, cooling_fan_thickness,
               fan_speed_silent, fan_speed_performance, fan_speed_turbo,
               airflow_rate_silent, airflow_rate_performance, airflow_rate_turbo,
               fan_noise_silent, fan_noise_performance, fan_noise_turbo,
               temp_silent, temp_performance, temp_turbo,
               chassis_thermal_conductivity, vent_area, fan_blade_count,
               ambient_temp, heat_pipe_material, fan_diameter):
    laptops.append({
        'brand': brand,
        'model': model,
        'cpu_tdp': cpu_tdp,
        'gpu_tdp': gpu_tdp,
        'chassis_thickness': chassis_thickness,
        'chassis_volume': chassis_volume,
        'num_heat_pipes': num_heat_pipes,
        'heat_pipe_length': heat_pipe_length,
        'num_cooling_fans': num_cooling_fans,
        'cooling_fan_thickness': cooling_fan_thickness,
        'fan_speed_silent': fan_speed_silent,
        'fan_speed_performance': fan_speed_performance,
        'fan_speed_turbo': fan_speed_turbo,
        'airflow_rate_silent': airflow_rate_silent,
        'airflow_rate_performance': airflow_rate_performance,
        'airflow_rate_turbo': airflow_rate_turbo,
        'fan_noise_silent': fan_noise_silent,
        'fan_noise_performance': fan_noise_performance,
        'fan_noise_turbo': fan_noise_turbo,
        'temp_silent': temp_silent,
        'temp_performance': temp_performance,
        'temp_turbo': temp_turbo,
        'chassis_thermal_conductivity': chassis_thermal_conductivity,
        'vent_area': vent_area,
        'fan_blade_count': fan_blade_count,
        'ambient_temp': ambient_temp,
        'heat_pipe_material': heat_pipe_material,
        'fan_diameter': fan_diameter
    })



add_laptop(
    'Apple', 'MacBook Air (M1)',
    cpu_tdp=10, gpu_tdp=0, chassis_thickness=16.1, chassis_volume=1.2,
    num_heat_pipes=1, heat_pipe_length=150, num_cooling_fans=0, cooling_fan_thickness=0,
    fan_speed_silent=0, fan_speed_performance=0, fan_speed_turbo=0,
    airflow_rate_silent=0, airflow_rate_performance=0, airflow_rate_turbo=0,
    fan_noise_silent=0, fan_noise_performance=0, fan_noise_turbo=0,
    temp_silent=35, temp_performance=40, temp_turbo=45,
    chassis_thermal_conductivity=205,  # Aluminum thermal conductivity in W/m·K
    vent_area=50,  # Estimated in cm²
    fan_blade_count=0,  # No fan
    ambient_temp=25,  # Standard room temperature
    heat_pipe_material=401,  # Copper thermal conductivity in W/m·K
    fan_diameter=0  # No fan
)

add_laptop(
    'Apple', 'MacBook Pro 13" (M1)',
    cpu_tdp=28, gpu_tdp=0, chassis_thickness=15.6, chassis_volume=1.4,
    num_heat_pipes=1, heat_pipe_length=150, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1500, fan_speed_performance=3000, fan_speed_turbo=4500,
    airflow_rate_silent=150, airflow_rate_performance=300, airflow_rate_turbo=450,
    fan_noise_silent=25, fan_noise_performance=35, fan_noise_turbo=45,
    temp_silent=40, temp_performance=45, temp_turbo=50,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=60,  # Estimated in cm²
    fan_blade_count=7,  # Typical laptop fan
    ambient_temp=25,
    heat_pipe_material=401,  # Copper
    fan_diameter=50  # Estimated in mm
)

add_laptop(
    'Apple', 'MacBook Pro 16"',
    cpu_tdp=45, gpu_tdp=35, chassis_thickness=16.2, chassis_volume=1.8,
    num_heat_pipes=2, heat_pipe_length=200, num_cooling_fans=2, cooling_fan_thickness=5,
    fan_speed_silent=2000, fan_speed_performance=3500, fan_speed_turbo=5000,
    airflow_rate_silent=200, airflow_rate_performance=350, airflow_rate_turbo=500,
    fan_noise_silent=28, fan_noise_performance=38, fan_noise_turbo=48,
    temp_silent=42, temp_performance=47, temp_turbo=52,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=80,  # Estimated in cm²
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,  # Copper
    fan_diameter=60  # Estimated in mm
)


add_laptop(
    'HP', 'Pavilion 15',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=17.9, chassis_volume=1.5,
    num_heat_pipes=1, heat_pipe_length=160, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1200, fan_speed_performance=2500, fan_speed_turbo=4000,
    airflow_rate_silent=120, airflow_rate_performance=250, airflow_rate_turbo=400,
    fan_noise_silent=22, fan_noise_performance=32, fan_noise_turbo=42,
    temp_silent=38, temp_performance=43, temp_turbo=48,
    chassis_thermal_conductivity=50,  # Plastic with some metal parts
    vent_area=70,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=205,  # Aluminum
    fan_diameter=50
)


add_laptop(
    'HP', 'Envy 13',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=14.9, chassis_volume=1.3,
    num_heat_pipes=1, heat_pipe_length=150, num_cooling_fans=1, cooling_fan_thickness=4,
    fan_speed_silent=1250, fan_speed_performance=2600, fan_speed_turbo=4100,
    airflow_rate_silent=125, airflow_rate_performance=260, airflow_rate_turbo=410,
    fan_noise_silent=23, fan_noise_performance=33, fan_noise_turbo=43,
    temp_silent=37, temp_performance=42, temp_turbo=47,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=60,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,  # Copper
    fan_diameter=45
)


add_laptop(
    'HP', 'Spectre x360',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=14.5, chassis_volume=1.2,
    num_heat_pipes=1, heat_pipe_length=145, num_cooling_fans=1, cooling_fan_thickness=4,
    fan_speed_silent=1300, fan_speed_performance=2700, fan_speed_turbo=4200,
    airflow_rate_silent=130, airflow_rate_performance=270, airflow_rate_turbo=420,
    fan_noise_silent=24, fan_noise_performance=34, fan_noise_turbo=44,
    temp_silent=36, temp_performance=41, temp_turbo=46,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=55,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,  # Copper
    fan_diameter=45
)


add_laptop(
    'HP', 'Omen 15',
    cpu_tdp=45, gpu_tdp=100, chassis_thickness=26.0, chassis_volume=2.5,
    num_heat_pipes=3, heat_pipe_length=250, num_cooling_fans=2, cooling_fan_thickness=8,
    fan_speed_silent=2000, fan_speed_performance=4000, fan_speed_turbo=5500,
    airflow_rate_silent=200, airflow_rate_performance=400, airflow_rate_turbo=550,
    fan_noise_silent=30, fan_noise_performance=40, fan_noise_turbo=50,
    temp_silent=45, temp_performance=50, temp_turbo=55,
    chassis_thermal_conductivity=150,  # Mix of plastic and metal
    vent_area=100,
    fan_blade_count=9,
    ambient_temp=25,
    heat_pipe_material=401,  # Copper
    fan_diameter=60
)


add_laptop(
    'Dell', 'XPS 13',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=14.8, chassis_volume=1.2,
    num_heat_pipes=1, heat_pipe_length=160, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1300, fan_speed_performance=2700, fan_speed_turbo=4200,
    airflow_rate_silent=130, airflow_rate_performance=270, airflow_rate_turbo=420,
    fan_noise_silent=23, fan_noise_performance=33, fan_noise_turbo=43,
    temp_silent=39, temp_performance=44, temp_turbo=49,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=60,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=45
)


add_laptop(
    'Dell', 'XPS 15',
    cpu_tdp=45, gpu_tdp=35, chassis_thickness=18.0, chassis_volume=1.8,
    num_heat_pipes=2, heat_pipe_length=180, num_cooling_fans=2, cooling_fan_thickness=5,
    fan_speed_silent=1600, fan_speed_performance=3100, fan_speed_turbo=4600,
    airflow_rate_silent=160, airflow_rate_performance=310, airflow_rate_turbo=460,
    fan_noise_silent=26, fan_noise_performance=36, fan_noise_turbo=46,
    temp_silent=41, temp_performance=46, temp_turbo=51,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=80,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=50
)



add_laptop(
    'Dell', 'Inspiron 15',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=20.0, chassis_volume=1.7,
    num_heat_pipes=1, heat_pipe_length=170, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1250, fan_speed_performance=2600, fan_speed_turbo=4100,
    airflow_rate_silent=125, airflow_rate_performance=260, airflow_rate_turbo=410,
    fan_noise_silent=23, fan_noise_performance=33, fan_noise_turbo=43,
    temp_silent=38, temp_performance=43, temp_turbo=48,
    chassis_thermal_conductivity=50,  # Plastic with metal parts
    vent_area=70,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=205,  # Aluminum
    fan_diameter=50
)



add_laptop(
    'Dell', 'G5 15',
    cpu_tdp=45, gpu_tdp=80, chassis_thickness=25.0, chassis_volume=2.4,
    num_heat_pipes=3, heat_pipe_length=240, num_cooling_fans=2, cooling_fan_thickness=8,
    fan_speed_silent=1900, fan_speed_performance=3900, fan_speed_turbo=5400,
    airflow_rate_silent=190, airflow_rate_performance=390, airflow_rate_turbo=540,
    fan_noise_silent=29, fan_noise_performance=39, fan_noise_turbo=49,
    temp_silent=44, temp_performance=49, temp_turbo=54,
    chassis_thermal_conductivity=150,  # Mix of plastic and metal
    vent_area=100,
    fan_blade_count=9,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=60
)


add_laptop(
    'Lenovo', 'ThinkPad X1 Carbon',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=14.9, chassis_volume=1.2,
    num_heat_pipes=1, heat_pipe_length=150, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1250, fan_speed_performance=2600, fan_speed_turbo=4100,
    airflow_rate_silent=125, airflow_rate_performance=260, airflow_rate_turbo=410,
    fan_noise_silent=22, fan_noise_performance=32, fan_noise_turbo=42,
    temp_silent=37, temp_performance=42, temp_turbo=47,
    chassis_thermal_conductivity=156,  # Magnesium alloy
    vent_area=60,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=45
)


add_laptop(
    'Lenovo', 'Yoga Slim 7',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=15.4, chassis_volume=1.3,
    num_heat_pipes=1, heat_pipe_length=155, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1275, fan_speed_performance=2650, fan_speed_turbo=4150,
    airflow_rate_silent=127, airflow_rate_performance=265, airflow_rate_turbo=415,
    fan_noise_silent=23, fan_noise_performance=33, fan_noise_turbo=43,
    temp_silent=38, temp_performance=43, temp_turbo=48,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=60,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=45
)


add_laptop(
    'Lenovo', 'Legion 5',
    cpu_tdp=45, gpu_tdp=115, chassis_thickness=26.0, chassis_volume=2.6,
    num_heat_pipes=4, heat_pipe_length=260, num_cooling_fans=2, cooling_fan_thickness=9,
    fan_speed_silent=2100, fan_speed_performance=4200, fan_speed_turbo=5700,
    airflow_rate_silent=210, airflow_rate_performance=420, airflow_rate_turbo=570,
    fan_noise_silent=31, fan_noise_performance=41, fan_noise_turbo=51,
    temp_silent=46, temp_performance=51, temp_turbo=56,
    chassis_thermal_conductivity=150,  # Mix of plastic and metal
    vent_area=110,
    fan_blade_count=9,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=60
)


add_laptop(
    'Lenovo', 'IdeaPad 5',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=17.9, chassis_volume=1.5,
    num_heat_pipes=1, heat_pipe_length=160, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1225, fan_speed_performance=2550, fan_speed_turbo=4050,
    airflow_rate_silent=122, airflow_rate_performance=255, airflow_rate_turbo=405,
    fan_noise_silent=22, fan_noise_performance=32, fan_noise_turbo=42,
    temp_silent=37, temp_performance=42, temp_turbo=47,
    chassis_thermal_conductivity=50,  # Plastic
    vent_area=70,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=205,
    fan_diameter=50
)



add_laptop(
    'ASUS', 'ZenBook 14',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=13.9, chassis_volume=1.1,
    num_heat_pipes=1, heat_pipe_length=145, num_cooling_fans=1, cooling_fan_thickness=4,
    fan_speed_silent=1250, fan_speed_performance=2600, fan_speed_turbo=4100,
    airflow_rate_silent=125, airflow_rate_performance=260, airflow_rate_turbo=410,
    fan_noise_silent=22, fan_noise_performance=32, fan_noise_turbo=42,
    temp_silent=36, temp_performance=41, temp_turbo=46,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=55,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=45
)


add_laptop(
    'ASUS', 'VivoBook S15',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=17.9, chassis_volume=1.5,
    num_heat_pipes=1, heat_pipe_length=160, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1275, fan_speed_performance=2650, fan_speed_turbo=4150,
    airflow_rate_silent=127, airflow_rate_performance=265, airflow_rate_turbo=415,
    fan_noise_silent=23, fan_noise_performance=33, fan_noise_turbo=43,
    temp_silent=37, temp_performance=42, temp_turbo=47,
    chassis_thermal_conductivity=50,  # Plastic
    vent_area=70,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=205,
    fan_diameter=50
)


add_laptop(
    'ASUS', 'ROG Zephyrus G14',
    cpu_tdp=35, gpu_tdp=65, chassis_thickness=19.9, chassis_volume=1.8,
    num_heat_pipes=3, heat_pipe_length=200, num_cooling_fans=2, cooling_fan_thickness=7,
    fan_speed_silent=1800, fan_speed_performance=3500, fan_speed_turbo=5000,
    airflow_rate_silent=180, airflow_rate_performance=350, airflow_rate_turbo=500,
    fan_noise_silent=28, fan_noise_performance=38, fan_noise_turbo=48,
    temp_silent=43, temp_performance=48, temp_turbo=53,
    chassis_thermal_conductivity=205,  # Aluminum-magnesium alloy
    vent_area=90,
    fan_blade_count=9,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=55
)


add_laptop(
    'ASUS', 'TUF Gaming A15',
    cpu_tdp=45, gpu_tdp=90, chassis_thickness=24.7, chassis_volume=2.3,
    num_heat_pipes=3, heat_pipe_length=240, num_cooling_fans=2, cooling_fan_thickness=8,
    fan_speed_silent=1950, fan_speed_performance=3950, fan_speed_turbo=5450,
    airflow_rate_silent=195, airflow_rate_performance=395, airflow_rate_turbo=545,
    fan_noise_silent=30, fan_noise_performance=40, fan_noise_turbo=50,
    temp_silent=44, temp_performance=49, temp_turbo=54,
    chassis_thermal_conductivity=150,  # Mix of plastic and metal
    vent_area=100,
    fan_blade_count=9,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=60
)


add_laptop(
    'Acer', 'Swift 3',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=15.9, chassis_volume=1.2,
    num_heat_pipes=1, heat_pipe_length=150, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1300, fan_speed_performance=2700, fan_speed_turbo=4200,
    airflow_rate_silent=130, airflow_rate_performance=270, airflow_rate_turbo=420,
    fan_noise_silent=23, fan_noise_performance=33, fan_noise_turbo=43,
    temp_silent=38, temp_performance=43, temp_turbo=48,
    chassis_thermal_conductivity=205,  # Aluminum
    vent_area=60,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=45
)


add_laptop(
    'Acer', 'Aspire 5',
    cpu_tdp=15, gpu_tdp=0, chassis_thickness=17.9, chassis_volume=1.6,
    num_heat_pipes=1, heat_pipe_length=160, num_cooling_fans=1, cooling_fan_thickness=5,
    fan_speed_silent=1250, fan_speed_performance=2600, fan_speed_turbo=4100,
    airflow_rate_silent=125, airflow_rate_performance=260, airflow_rate_turbo=410,
    fan_noise_silent=22, fan_noise_performance=32, fan_noise_turbo=42,
    temp_silent=38, temp_performance=43, temp_turbo=48,
    chassis_thermal_conductivity=50,  # Plastic
    vent_area=70,
    fan_blade_count=7,
    ambient_temp=25,
    heat_pipe_material=205,
    fan_diameter=50
)


add_laptop(
    'Acer', 'Predator Helios 300',
    cpu_tdp=45, gpu_tdp=90, chassis_thickness=26.8, chassis_volume=2.7,
    num_heat_pipes=3, heat_pipe_length=250, num_cooling_fans=2, cooling_fan_thickness=9,
    fan_speed_silent=2000, fan_speed_performance=4000, fan_speed_turbo=5500,
    airflow_rate_silent=200, airflow_rate_performance=400, airflow_rate_turbo=550,
    fan_noise_silent=30, fan_noise_performance=40, fan_noise_turbo=50,
    temp_silent=45, temp_performance=50, temp_turbo=55,
    chassis_thermal_conductivity=150,  # Mix of plastic and metal
    vent_area=110,
    fan_blade_count=9,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=60
)


add_laptop(
    'Acer', 'Nitro 5',
    cpu_tdp=45, gpu_tdp=80, chassis_thickness=25.9, chassis_volume=2.5,
    num_heat_pipes=3, heat_pipe_length=240, num_cooling_fans=2, cooling_fan_thickness=8,
    fan_speed_silent=1900, fan_speed_performance=3900, fan_speed_turbo=5400,
    airflow_rate_silent=190, airflow_rate_performance=390, airflow_rate_turbo=540,
    fan_noise_silent=29, fan_noise_performance=39, fan_noise_turbo=49,
    temp_silent=44, temp_performance=49, temp_turbo=54,
    chassis_thermal_conductivity=150,  # Mix of plastic and metal
    vent_area=100,
    fan_blade_count=9,
    ambient_temp=25,
    heat_pipe_material=401,
    fan_diameter=60
)



# Convert the list of dictionaries into a DataFrame
data = pd.DataFrame(laptops)

# Check the number of entries
print(f"Total entries in dataset: {len(data)}")



# Display basic statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())




# Generate variations to increase dataset size
for i in range(2):  # Create two variations of each model
    for laptop in laptops.copy():
        # Slightly vary some parameters
        cpu_tdp_variation = laptop['cpu_tdp'] * np.random.uniform(0.95, 1.05)
        gpu_tdp_variation = laptop['gpu_tdp'] * np.random.uniform(0.95, 1.05)
        chassis_thickness_variation = laptop['chassis_thickness'] * np.random.uniform(0.95, 1.05)
        chassis_thermal_conductivity_variation = laptop['chassis_thermal_conductivity'] * np.random.uniform(0.95, 1.05)
        vent_area_variation = laptop['vent_area'] * np.random.uniform(0.95, 1.05)
        fan_blade_count_variation = laptop['fan_blade_count']  # This is usually an integer; you might keep it the same
        ambient_temp_variation = laptop['ambient_temp']  # Assuming ambient temperature remains constant
        heat_pipe_material_variation = laptop['heat_pipe_material']  # Assuming material doesn't change
        fan_diameter_variation = laptop['fan_diameter'] * np.random.uniform(0.95, 1.05)
        # Create a new model name to avoid duplicates
        model_variation = laptop['model'] + f" Variant {i+1}"
        
        add_laptop(
            laptop['brand'],
            model_variation,
            cpu_tdp_variation,
            gpu_tdp_variation,
            chassis_thickness_variation,
            laptop['chassis_volume'],
            laptop['num_heat_pipes'],
            laptop['heat_pipe_length'],
            laptop['num_cooling_fans'],
            laptop['cooling_fan_thickness'],
            laptop['fan_speed_silent'],
            laptop['fan_speed_performance'],
            laptop['fan_speed_turbo'],
            laptop['airflow_rate_silent'],
            laptop['airflow_rate_performance'],
            laptop['airflow_rate_turbo'],
            laptop['fan_noise_silent'],
            laptop['fan_noise_performance'],
            laptop['fan_noise_turbo'],
            laptop['temp_silent'],
            laptop['temp_performance'],
            laptop['temp_turbo'],
            laptop['chassis_thermal_conductivity'],
            laptop['vent_area'],
            laptop['fan_blade_count'],
            laptop['ambient_temp'],
            laptop['heat_pipe_material'],
            laptop['fan_diameter']
        )





# Update the DataFrame
data = pd.DataFrame(laptops)

# Remove duplicates
data = data.drop_duplicates(subset=['brand', 'model'])

# Reset the index
data = data.reset_index(drop=True)

# Check the number of entries
print(f"Total entries in dataset after expansion: {len(data)}")



# Create new features
data['total_tdp'] = data['cpu_tdp'] + data['gpu_tdp']
data['cooling_capacity'] = data['num_heat_pipes'] + data['num_cooling_fans']

# Input features
input_features = [
    'cpu_tdp', 'gpu_tdp', 'total_tdp', 'cooling_capacity',
    'chassis_thermal_conductivity', 'vent_area', 'fan_blade_count',
    'ambient_temp', 'heat_pipe_material', 'fan_diameter'
]


# Output features
output_features = ['fan_speed_turbo', 'temp_turbo', 'fan_noise_turbo']








# Compute correlation matrix
corr_matrix = data[input_features + output_features].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# Histograms for input features
data[input_features].hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Box plots for input features
data[input_features].plot(kind='box', subplots=True, layout=(2, 5), figsize=(15, 8))
plt.tight_layout()
plt.show()


# Histograms for output features
data[output_features].hist(figsize=(6, 4))
plt.tight_layout()
plt.show()

# Box plots for output features
data[output_features].plot(kind='box', subplots=True, layout=(1, 3), figsize=(15, 4))
plt.tight_layout()
plt.show()




# Extract features and targets
X = data[input_features].values.astype('float32')
y = data[output_features].values.astype('float32')

# Proceed with splitting the data, normalization, and model training
# ...






# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



# Initialize scalers
input_scaler = StandardScaler()
output_scaler = StandardScaler()




# Fit and transform the data
X_train_scaled = input_scaler.fit_transform(X_train)
y_train_scaled = output_scaler.fit_transform(y_train)

X_val_scaled = input_scaler.transform(X_val)
y_val_scaled = output_scaler.transform(y_val)

X_test_scaled = input_scaler.transform(X_test)
y_test_scaled = output_scaler.transform(y_test)


print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train_scaled shape:", y_train_scaled.shape)



def create_pinn(input_dim, output_dim, num_neurons=128, num_layers=5, dropout_rate=0.2, l2_reg=1e-4):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(num_layers):
        x = layers.Dense(num_neurons, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(output_dim)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model





# Define model
input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]  # Should be 3
model = create_pinn(input_dim, output_dim, num_neurons=64, num_layers=4)

# Training loop (as before), ensuring that x_batch includes all input features

model.summary()


def physics_loss(y_pred, x):
    # Unpack predictions
    fan_speed_turbo_pred = y_pred[:, 0]
    temp_turbo_pred = y_pred[:, 1]
    fan_noise_turbo_pred = y_pred[:, 2]

    # Unpack inputs
    total_tdp = x[:, input_features.index('total_tdp')]
    cooling_capacity = x[:, input_features.index('cooling_capacity')]
    vent_area = x[:, input_features.index('vent_area')]
    chassis_thermal_conductivity = x[:, input_features.index('chassis_thermal_conductivity')]
    ambient_temp = x[:, input_features.index('ambient_temp')]
    fan_blade_count = x[:, input_features.index('fan_blade_count')]

    # Physics constraints
    constraint1 = fan_speed_turbo_pred - (total_tdp * 40) / (vent_area * cooling_capacity)
    constraint2 = temp_turbo_pred - (ambient_temp + total_tdp / chassis_thermal_conductivity)
    constraint3 = fan_noise_turbo_pred - (fan_speed_turbo_pred * 0.01 * fan_blade_count)

    # Combine constraints
    constraints = tf.stack([constraint1, constraint2, constraint3], axis=1)
    phys_loss = tf.reduce_mean(tf.square(constraints))
    return phys_loss





def total_loss(y_true, y_pred, x, alpha=0.01):
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    phys_loss = physics_loss(y_pred, x)
    total_loss_value = data_loss + alpha * phys_loss
    return total_loss_value




class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, x, alpha=0.1):
        super(CustomLoss, self).__init__()
        self.x = x
        self.alpha = alpha

    def call(self, y_true, y_pred):
        data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        phys_loss = physics_loss(y_pred, self.x)
        total_loss_value = data_loss + self.alpha * phys_loss
        return total_loss_value



# Prepare the datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_scaled))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val_scaled))
val_dataset = val_dataset.batch(batch_size)



optimizer = optimizers.Adam(learning_rate=0.001)
batch_size = 16
epochs = 500
train_losses = []
val_losses = []

alpha = 0.01  # Adjust alpha as needed

# Early stopping parameters
best_val_loss = np.inf
patience = 20
patience_counter = 0

for epoch in range(epochs):
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            data_loss = tf.reduce_mean(tf.square(y_batch - y_pred))
            phys_loss = physics_loss(y_pred, x_batch)
            loss_value = data_loss + alpha * phys_loss
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += loss_value.numpy()
    train_losses.append(epoch_loss / (step + 1))

    # Validation
    val_loss = 0
    for val_step, (x_val_batch, y_val_batch) in enumerate(val_dataset):
        y_val_pred = model(x_val_batch, training=False)
        data_loss = tf.reduce_mean(tf.square(y_val_batch - y_val_pred))
        phys_loss = physics_loss(y_val_pred, x_val_batch)
        val_loss_value = data_loss + alpha * phys_loss
        val_loss += val_loss_value.numpy()
    val_losses.append(val_loss / (val_step + 1))

    # Early stopping
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0
        # Save the best model weights
        best_weights = model.get_weights()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.set_weights(best_weights)
            break

    # Print progress
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}')



plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()




# Prepare test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test_scaled))
test_dataset = test_dataset.batch(batch_size)

test_loss = 0
for test_step, (x_test_batch, y_test_batch) in enumerate(test_dataset):
    y_test_pred = model(x_test_batch, training=False)
    data_loss = tf.reduce_mean(tf.square(y_test_batch - y_test_pred))
    phys_loss = physics_loss(y_test_pred, x_test_batch)
    loss_value = data_loss + alpha * phys_loss
    test_loss += loss_value.numpy()
test_loss = test_loss / (test_step + 1)
print(f'Test Loss: {test_loss:.4f}')




# Predict on test data
y_pred_test_scaled = model.predict(X_test_scaled)
y_pred_test = output_scaler.inverse_transform(y_pred_test_scaled)
y_test_inverse = output_scaler.inverse_transform(y_test_scaled)

# Fan Speed in Turbo Mode
plt.figure(figsize=(8, 6))
plt.scatter(y_test_inverse[:, 0], y_pred_test[:, 0], alpha=0.7)
plt.xlabel('Actual Fan Speed Turbo (RPM)')
plt.ylabel('Predicted Fan Speed Turbo (RPM)')
plt.title('Actual vs. Predicted Fan Speed in Turbo Mode')
plt.plot([min(y_test_inverse[:, 0]), max(y_test_inverse[:, 0])],
         [min(y_test_inverse[:, 0]), max(y_test_inverse[:, 0])], 'r--')
plt.show()

# Surface Temperature in Turbo Mode
plt.figure(figsize=(8, 6))
plt.scatter(y_test_inverse[:, 1], y_pred_test[:, 1], alpha=0.7)
plt.xlabel('Actual Surface Temperature Turbo (°C)')
plt.ylabel('Predicted Surface Temperature Turbo (°C)')
plt.title('Actual vs. Predicted Surface Temperature in Turbo Mode')
plt.plot([min(y_test_inverse[:, 1]), max(y_test_inverse[:, 1])],
         [min(y_test_inverse[:, 1]), max(y_test_inverse[:, 1])], 'r--')
plt.show()




# List of input features
input_features = [
    'cpu_tdp', 'gpu_tdp', 'total_tdp', 'cooling_capacity',
    'chassis_thermal_conductivity', 'vent_area', 'fan_blade_count',
    'ambient_temp', 'heat_pipe_material', 'fan_diameter'
]

# New design input with all required features
new_design = pd.DataFrame({
    'cpu_tdp': [35],
    'gpu_tdp': [85],
    'total_tdp': [35 + 85],            # 120
    'cooling_capacity': [3 + 2],       # 5
    'chassis_thermal_conductivity': [150],  # Mix of plastic and metal
    'vent_area': [100],                # Estimated in cm²
    'fan_blade_count': [9],            # Typical number of blades
    'ambient_temp': [25],              # Standard room temperature
    'heat_pipe_material': [401],       # Copper thermal conductivity
    'fan_diameter': [60]               # In mm
})

# Normalize the new input
new_design_scaled = input_scaler.transform(new_design[input_features])

# Predict outputs
new_prediction_scaled = model.predict(new_design_scaled)
new_prediction = output_scaler.inverse_transform(new_prediction_scaled)

# Display the predictions
prediction_dict = dict(zip(output_features, new_prediction[0]))
for key, value in prediction_dict.items():
    print(f'{key}: {value:.2f}')
