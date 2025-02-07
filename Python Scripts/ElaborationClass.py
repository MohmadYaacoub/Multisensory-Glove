import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Define a 1D Convolutional Neural Network (CNN) class
class CNN1D:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Model parameters
        self.filters = [[8], [16]]  # Filter configurations for Conv1D layers
        self.kernels = [2, 3, 4]  # Kernel sizes for Conv1D layers
        self.batch_size = 64  # Batch size
        self.epochs = 300  # Training epochs
        self.learning_rate = 0.001  # Learning rate
        self.n_classes = 28  # Number of output classes
        
        # Optimizer, loss function, and evaluation metrics
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        
        # Early stopping callback
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        # Dataset assignments
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        # Train the model
        self.train()

    def __create_model(self, filters, kernel, n_classes, verbose=1):
        model = tf.keras.models.Sequential()
        for i, f in enumerate(filters):
            if i == 0:
                model.add(tf.keras.layers.Conv1D(f, kernel, activation='relu', input_shape=self.X_train.shape[1:], padding='same'))
            else:
                model.add(tf.keras.layers.Conv1D(f, kernel, activation='relu', padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        if verbose:
            model.summary()
        return model

    def train(self):
        best_val_acc, best_f, best_k = 0, None, None
        optimizer_config = self.optimizer.get_config()

        with tqdm(total=len(self.filters) * len(self.kernels), desc="Training Progress", unit="iteration") as pbar:
            for f in self.filters:
                for k in self.kernels:
                    model = self.__create_model(f, k, self.n_classes, verbose=0)
                    optimizer = type(self.optimizer).from_config(optimizer_config)
                    model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
                    
                    history = model.fit(
                        self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                        validation_data=(self.X_val, self.y_val), callbacks=self.callbacks, verbose=0
                    )
                    
                    val_acc = history.history['val_accuracy'][-1]
                    if val_acc > best_val_acc:
                        best_val_acc, best_f, best_k = val_acc, f, k
                        model.save(f"model_{f}_{k}.h5")
                    pbar.update(1)
        
        # Load and evaluate the best model
        best_model = tf.keras.models.load_model(f"model_{best_f}_{best_k}.h5")
        train_acc = best_model.evaluate(self.X_train, self.y_train, verbose=0)[1]
        val_acc = best_model.evaluate(self.X_val, self.y_val, verbose=0)[1]
        test_acc = best_model.evaluate(self.X_test, self.y_test, verbose=0)[1]
        
        print(f"Best Filters: {best_f}, Best Kernel: {best_k}")
        print("Best Model Architecture:")
        best_model.summary()
        print(f"Train Accuracy: {train_acc}, Validation Accuracy: {val_acc}, Test Accuracy: {test_acc}")

        # Post Training Quantization (PTQ)
        best_model = tf.keras.models.load_model(f"model_{best_f}_{best_k}.h5")

        def representative_dataset():
            for data in self.X_train.rebatch(1).take(150):
                yield [tf.dtypes.cast(data[0], tf.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8 
        converter.inference_output_type = tf.uint8

        tflite_quant_model = converter.convert()
        with open(f"model_{best_f}_{best_k}.tflite", 'wb') as f:
            f.write(tflite_quant_model)

        # Load and test quantized model
        interpreter = tf.lite.Interpreter(model_path=f"model_{best_f}_{best_k}.tflite")
        interpreter.allocate_tensors()

        def run_tflite_model(tflite_file, test_data_indices):
            interpreter = tf.lite.Interpreter(model_path=tflite_file)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            predictions = np.zeros((len(test_data_indices),), dtype=int)
            
            for i, test_idx in enumerate(test_data_indices):
                test_sample = self.X_test[test_idx]
                
                if input_details['dtype'] == np.int8:
                    input_scale, input_zero_point = input_details["quantization"]
                    test_sample = test_sample / input_scale + input_zero_point
                
                test_sample = np.expand_dims(test_sample, axis=0).astype(input_details["dtype"])
                interpreter.set_tensor(input_details["index"], test_sample)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details["index"])[0]
                predictions[i] = output.argmax()
            
            return predictions

        tflite_file = f"model_{best_f}_{best_k}.tflite"
        test_indices = range(self.X_test.shape[0])
        predictions = run_tflite_model(tflite_file, test_indices)
        pred_labels = tf.keras.utils.to_categorical(predictions, num_classes=28)

        accuracy = accuracy_score(self.y_test, pred_labels)
        print('Accuracy:', accuracy)
