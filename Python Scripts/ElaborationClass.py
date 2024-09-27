# Import necessary libraries
import tensorflow as tf 
from tqdm import tqdm  

# Define a 1D Convolutional Neural Network (CNN) class
class CNN_1D():
    # Initialize the class with training, validation, and test datasets
    def __init__(self,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 X_test,
                 y_test):
        # Model configuration: filters, kernels, batch size, epochs, learning rate, and classes
        self.filters = [[8], [16]]  # Different filter sizes for Conv1D layers
        self.kernels = [2, 3, 4]  # Different kernel sizes for Conv1D layers
        self.batch_size = 64  # Number of samples per gradient update
        self.epochs = 300  # Number of training epochs
        self.learning_rate = 0.001  # Learning rate for the optimizer
        self.n_classes = 28  # Number of output classes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  # Optimizer configuration
        self.loss = 'categorical_crossentropy'  # Loss function for classification
        self.metrics = ['accuracy']  # Evaluation metric to track
        # Callbacks for early stopping to prevent overfitting
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
                         # Option to add learning rate scheduling (commented out)

        # Assign the training, validation, and testing data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # Automatically start training upon initialization
        self.train()

    # Internal method to create the CNN model with the specified filters and kernel sizes
    def __create_model(self, filters, kernel, n_classes, verbose=1):
        model = tf.keras.models.Sequential()  # Sequential model for a linear stack of layers
        for i, f in enumerate(filters):
            if i == 0:
                # First Conv1D layer with input shape specified
                model.add(tf.keras.layers.Conv1D(f, kernel, activation='relu', input_shape=(self.X_train.shape[1:]), padding='same'))
            else:
                # Additional Conv1D layers
                model.add(tf.keras.layers.Conv1D(f, kernel, activation='relu', padding='same'))
            model.add(tf.keras.layers.BatchNormalization())  # Batch normalization to stabilize learning
            model.add(tf.keras.layers.MaxPooling1D(2))  # Max pooling to reduce dimensionality
        model.add(tf.keras.layers.GlobalAveragePooling1D())  # Global average pooling layer
        # Output dense layer with softmax activation for classification
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        if verbose:
            model.summary()  # Print the model architecture if verbose is enabled
        return model

    # Train the CNN model using different combinations of filters and kernels
    def train(self):
        best_val_acc = 0  # Variable to track the best validation accuracy
        best_f = None  # To store the best filter configuration
        best_k = None  # To store the best kernel size

        # Save initial optimizer state to reuse across models
        initial_optimizer_config = self.optimizer.get_config()

        # Progress bar to display training progress across filter and kernel combinations
        with tqdm(total=len(self.filters) * len(self.kernels), desc="Training Progress", unit="iteration") as pbar:
            for f in self.filters:  # Iterate through each filter configuration
                for k in self.kernels:  # Iterate through each kernel size
                    # Create and compile the model
                    model = self.__create_model(f, k, self.n_classes, verbose=0)
                    optimizer = type(self.optimizer).from_config(initial_optimizer_config)  # Reset optimizer for each iteration
                    model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
                    
                    # Train the model on the training data and validate on validation data
                    history = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, 
                                        validation_data=(self.X_val, self.y_val), callbacks=self.callbacks, verbose=0)
                    
                    # If the current model's validation accuracy is the best, save it
                    if history.history['val_accuracy'][-1] > best_val_acc:
                        best_val_acc = history.history['val_accuracy'][-1]  # Update best validation accuracy
                        model.save(f"model_{f}_{k}.h5")  # Save the model
                        best_f = f  # Track best filter configuration
                        best_k = k  # Track best kernel size
                    pbar.update(1)  # Update the progress bar
        
        # Load the best model based on validation accuracy
        best_model = tf.keras.models.load_model(f"model_{best_f}_{best_k}.h5")

        # Evaluate the best model on training, validation, and testing datasets
        _, train_accuracy = best_model.evaluate(self.X_train, self.y_train, verbose=0)
        _, test_accuracy = best_model.evaluate(self.X_test, self.y_test, verbose=0)
        _, val_accuracy = best_model.evaluate(self.X_val, self.y_val, verbose=0)

        # Print the best filter/kernel configuration and accuracies
        print(f"best_f: {best_f}, best_k: {best_k}")
        print("Best model")
        best_model.summary()  # Print the best model architecture
        print("Train_accuracy:", train_accuracy, "Val accuracy:", val_accuracy, "Test Accuracy:", test_accuracy)
