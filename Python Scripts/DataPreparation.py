import glob
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, base_file_path):
        # Initialize with a base file path, set a maximum number of files per class, and initialize variables for data
        self.max_files_per_class = 50
        self.base_file_path = base_file_path
        self.X = []
        self.y = []
        self.class_counts = {}
        self.text_files = self._get_text_files()
        
    def _get_text_files(self):
        # Get all directories in the base file path
        data = glob.glob(self.base_file_path + "/*")
        category_names = [os.path.basename(directory_path) for directory_path in data]
        text_files = {category: [] for category in category_names}
        
        # Map each category to its corresponding directories
        for directory_path in data:
            category = os.path.basename(directory_path)
            if category in text_files:
                text_files[category].append(directory_path)
        
        return text_files
    
    def _load_data(self):
        # Load data from each file into X and y arrays
        for label, (key, directory_paths) in enumerate(self.text_files.items(), start=0):
            self.class_counts[key] = 0
            for directory_path in directory_paths:
                files_in_directory = glob.glob(directory_path + "/*")
                for file_path in files_in_directory:
                    df = pd.read_csv(file_path, header=None)
                    """
                    To extract the fusion data df_array = df.iloc[1:51, :35].values 
                    To extact only FSR data change to df_array = df.iloc[1:51, :5].values
                    To extract only IMU data change to df_array = df.iloc[1:51, 5:35].valuest
                    """
                    df_array = df.iloc[1:51, :35].values  # Extract specific part of the dataframe
                    self.X.append(df_array)
                    self.y.append(label)
                    self.class_counts[key] += 1
                    # Stop if maximum files per class is reached
                    if self.class_counts[key] == self.max_files_per_class:
                        break
                print(f"Class {key}: {self.class_counts[key]} files")
    
    def _prepare_data(self):
        # Convert X and y to numpy arrays and ensure correct data types
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int32)
        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)
        
    def get_data_splits(self):
        # Load data, prepare it, and split it into training, validation, and test sets
        self._load_data()
        self._prepare_data()
        y_categorical = tf.keras.utils.to_categorical(self.y)
        X_train, x_tmp, y_train, y_tmp = train_test_split(self.X, y_categorical, test_size=0.3, random_state=42, stratify=self.y)
        X_val, X_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=42, stratify=np.argmax(y_tmp, axis=1))
        return X_train, y_train, X_val, y_val, X_test, y_test
