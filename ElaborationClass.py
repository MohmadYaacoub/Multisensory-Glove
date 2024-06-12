import tensorflow as tf
from tqdm import tqdm
class CNN_1D():
    """
    CNN_1D: Class to train a 1D Convolutional Neural Network
    Attributes:
    filters: List of filters for the convolutional layers
    kernels: List of kernel sizes for the convolutional layers
    window: Window size for the input data
    n_folds: Number of folds for cross-validation
    channels: Number of channels in the input data
    batch_size: Batch size for training
    epochs: Number of epochs for training
    learning_rate: Learning rate for the optimizer
    optimizer: Optimizer for training
    loss: Loss function for training
    metrics: Metrics for training
    callbacks: Callbacks for training
    norm_dict: Dictionary with normalization parameters
    train: Method to train the model
    """
    def __init__(self,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 X_test,
                 y_test):
        self.filters = [[8], [16]]
        self.kernels = [2, 3, 4]
        self.batch_size = 64
        self.epochs = 300
        self.learning_rate = 0.001
        self.n_classes = 28
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
                         # tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch, lr: lr * 0.5 if epoch % 5 == 0 and epoch != 0 and lr>1e-4 else lr)]

        # Assign the training, validation, and testing data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        ########## Train ################
        self.train()

    def __create_model(self, filters, kernel, n_classes, verbose=1):
        model = tf.keras.models.Sequential()
        for i, f in enumerate(filters):
            if i == 0:
                model.add(tf.keras.layers.Conv1D(f, kernel, activation='relu', input_shape=(self.X_train.shape[1:]), padding='same'))
            else:
                model.add(tf.keras.layers.Conv1D(f, kernel, activation='relu', padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        if verbose:
            model.summary()
        return model
    def train(self):
        best_val_acc = 0
        best_f=None
        best_k= None
        initial_optimizer_config = self.optimizer.get_config()
        with tqdm(total=len(self.filters) * len(self.kernels), desc="Training Progress", unit="iteration") as pbar:
            for f in self.filters:
                for k in self.kernels:
                    model = self.__create_model(f, k, self.n_classes, verbose=0)
                    optimizer = type(self.optimizer).from_config(initial_optimizer_config)
                    model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
                    history = model.fit(self.X_train,self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_val,self.y_val),callbacks=self.callbacks, verbose=0)
                    if history.history['val_accuracy'][-1] > best_val_acc:
                        # print('best val acc:', history.history['val_accuracy'][-1])
                        best_val_acc = history.history['val_accuracy'][-1]
                        model.save(f"model_{f}_{k}.h5")
                        best_f = f
                        best_k = k
                    pbar.update(1)
        best_model = tf.keras.models.load_model(f"model_{best_f}_{best_k}.h5")
        # best_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        _, train_accuracy = best_model.evaluate(self.X_train,self.y_train, verbose=0)
        _, test_accuracy = best_model.evaluate(self.X_test,self.y_test, verbose=0)
        _, val_accuracy = best_model.evaluate(self.X_val,self.y_val, verbose=0)
        print(f"best_f: {best_f},best_k: {best_k}")
        print("Best model")
        best_model.summary()
        print("Train_accuracy:",train_accuracy,"Val accuracy:", val_accuracy, " Test Accuracy:", test_accuracy)