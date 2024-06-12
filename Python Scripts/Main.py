from DataPreparation import Dataset
from ElaborationClass import CNN_1D

dataset = Dataset(base_file_path="Dataset")
X_train, y_train, X_val, y_val, X_test, y_test = dataset.get_data_splits()

model = CNN_1D(X_train, y_train, X_val, y_val, X_test, y_test)
