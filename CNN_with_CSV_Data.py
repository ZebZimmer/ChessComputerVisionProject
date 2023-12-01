import tensorflow as tf
from tensorflow.keras import layers, models


class ChessCNN_CSV:
    def __init__(self, input_dim_x, input_dim_y):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_dim_x, input_dim_y, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))

        # Output layer
        self.model.add(layers.Dense(12, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_images, train_labels, validation_images, validation_labels, epochs=31):
        self.model.fit(train_images, train_labels, epochs=epochs, validation_data=(validation_images, validation_labels))
        self.model.save(f"C:/Users/zebzi/Documents/School/Master_Year/CSCI 5525/Project/Models_Saved/CNN_with_CSV_Data_{epochs}epochs.h5")

    def test(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy: {test_acc}")
        print(f"Test loss: {test_loss}")