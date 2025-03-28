from keras.applications import InceptionResNetV2
from keras import layers, models
import keras
import pandas as pd

class Train_cnn:

      def __init__(self, input_shape, num_classes, model=False):

          self.input_shape = input_shape
          self.num_classes = num_classes

          self.base_model = InceptionResNetV2(input_shape=self.input_shape,
                               weights='imagenet',
                               classes=num_classes,
                               include_top=False)

          if model:
             self.m_model = keras.models.load_model('models/modelo_cnn.h5')
             self.model_train = models.Sequential([
                 self.m_model,
                 layers.Flatten(),
                 layers.Dense(1024, activation='relu'),
                 layers.Dropout(0.19),
                 layers.Dense(512, activation='relu'),
                 layers.Dropout(0.19),
                 layers.Dense(256, activation='relu'),
                 layers.Dropout(0.15),
                 layers.Dense(self.num_classes, activation='softmax')

             ])
          else:
              self.model_train = models.Sequential([
                  self.base_model,
                  layers.Flatten(),
                  layers.Dense(1024, activation='relu'),
                  layers.Dropout(0.19),
                  layers.Dense(512, activation='relu'),
                  layers.Dropout(0.19),
                  layers.Dense(256, activation='relu'),
                  layers.Dropout(0.15),
                  layers.Dense(self.num_classes, activation='softmax')

              ])



      def train_init(self, x_train, y_train, x_test, y_test, epochs, batch_size, save = True ):
          # Compile o modelo
          self.model_train.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

          history = self.model_train.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)
          if save:
                self.model_train.save(f"models/modelo_cnn1.h5")
          return history


      @staticmethod
      def train_metrics(history):

          results = {
              'Epoch': list(range(1, len(history.history['accuracy']) + 1)),
              'Accuracy': history.history['accuracy'],
              'Validation Accuracy': history.history['val_accuracy'],
              'Loss': history.history['loss'],
              'Validation Loss': history.history['val_loss'],

          }

          results_df = pd.DataFrame(results)

          return results_df

      def train_getModel(self):
          return  self.model_train