from DataPreparation import Dataset
from ElaborationClass import CNN_1D

dataset = Dataset(base_file_path=r"C:\Users\mhamad.yackoub\OneDrive - unige.it\Desktop\VScode\Multimodal Glove\Csv Fusion")
X_train, y_train, X_val, y_val, X_test, y_test = dataset.get_data_splits()

model = CNN_1D(X_train, y_train, X_val, y_val, X_test, y_test)
