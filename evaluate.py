import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from keras.models import Model
import cv2
import random
from sklearn.manifold import TSNE
from tqdm import tqdm

class Evaluation:

      def __init__(self, model, x_test, y_test):

          self.model = model

          self.x_test = x_test
          self.y_test = y_test

      def evaluate_accuracy(self):

             loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
             print("Acurácia do modelo no conjunto de teste: {:.2f}%".format(accuracy * 100))

      def evaluate_history(history):

             plt.plot(history.history['loss'], label='Training Loss')
             plt.xlabel('Épocas')
             plt.ylabel('Perda')
             plt.legend()
             plt.show()

             plt.plot(history.history['accuracy'], label='Training Accuracy')
             plt.xlabel('Épocas')
             plt.ylabel('Acurácia')
             plt.legend()
             plt.show()

      def evaluate_confusion_matrix(self):


            # Fazer previsões com o modelo
            predicted_probs = self.model.predict(self.x_test)
            predicted_labels = np.argmax(predicted_probs, axis=1)

            # Calcular a matriz de confusão
            cm = confusion_matrix(self.y_test, predicted_labels)

            # Plotar a matriz de confusão usando Seaborn
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            # Salvar a matriz de confusão como uma imagem
            plt.savefig('confusion_matrix.png')

            # Fechar a figura (opcional)
            plt.close()

      def evaluate_scores(self):

          predicted_probs = self.model.predict(self.x_test)
          predicted_labels = np.argmax(predicted_probs, axis=1)

          # Calcular métricas de avaliação


          accuracy = accuracy_score(self.y_test, predicted_labels)
          precision = precision_score(self.y_test, predicted_labels, average='weighted')
          recall = recall_score(self.y_test, predicted_labels, average='weighted')
          f1 = f1_score(self.y_test, predicted_labels, average='weighted')

          print("Acurácia:", accuracy)
          print("Precisão:", precision)
          print("Recall:", recall)
          print("F1-Score:", f1)

          return accuracy, precision, recall, f1

      def evaluate_get_gradcam(image_array, model, last_conv_layer_name):
          grad_model = Model(
              [model.inputs],
              [model.get_layer(last_conv_layer_name).output, model.output]
          )

          with tf.GradientTape() as tape:
              conv_outputs, predictions = grad_model(image_array)
              top_prediction = tf.argmax(predictions[0])
              top_class_channel = predictions[:, top_prediction]

          grads = tape.gradient(top_class_channel, conv_outputs)
          pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

          conv_outputs = conv_outputs[0]
          heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
          heatmap = np.maximum(heatmap, 0)
          heatmap /= np.max(heatmap)

          return heatmap

      def evaluate_plot_heatmap(self, image, input_shape, last_conv_layer_name):

          image = tf.keras.preprocessing.image.load_img(image, target_size=input_shape[:2])

          image = tf.keras.preprocessing.image.img_to_array(image)
          image_array = np.expand_dims(image, axis=0)

          heatmap = self.evaluate_get_gradcam(image_array, self.model,last_conv_layer_name)

          # Upscale da heatmap para o tamanho da imagem original
          heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
          heatmap = np.uint8(255 * heatmap)
          heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

          # Combinação da imagem original com a heatmap
          superimposed_img = cv2.addWeighted(image.astype('uint8'), 0.7, heatmap, 0.3, 0)

          # Visualize a imagem original, heatmap e a imagem combinada
          plt.figure(figsize=(10, 6))
          plt.subplot(133)
          plt.imshow(superimposed_img / 255.0)  # Normalizar a imagem combinada
          plt.title('Superimposed')
          plt.show()
          plt.savefig('heatmap.png')


      def evaluate_t_sne(self, num_classes_total, dict_classes, input_shape,last_conv_layer_name):

          selected_classes = random.sample(range(len(dict_classes.keys()), num_classes_total))

          imagens = []
          rotulos = []

          for rotulo, arquivos in tqdm(dict_classes.items()):
              if int(rotulo) not in selected_classes:
                  continue

              for i, arquivo in enumerate(arquivos):
                  if ".bmp" not in arquivo:
                      continue

                  imagem = tf.keras.preprocessing.image.load_img(arquivo, target_size=input_shape[:2])
                  imagem = tf.keras.preprocessing.image.img_to_array(imagem)

                  imagens.append(imagem)
                  rotulos.append(int(rotulo))

                  if i >= 3 - 1:
                      break

          tf.convert_to_tensor(imagens), tf.convert_to_tensor(rotulos)


          rotulos -= 1
          imagens = imagens/ 255.0


          intermediate_layer_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(last_conv_layer_name).output)
          representations = intermediate_layer_model.predict(imagens)
          tsne = TSNE(n_components=2, perplexity=10, random_state=42)
          tsne_results = tsne.fit_transform(representations)


          patterned_colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'pink', 'yellow',
                              'lime', 'magenta', 'teal', 'indigo', 'gold', 'maroon', 'navy', 'olive',
                              'peru', 'orchid', 'skyblue', 'darkgreen']

          repeated_pattern = [color for color in patterned_colors for _ in range(3)]

          plt.figure(figsize=(10, 8))
          plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=repeated_pattern, cmap='viridis', alpha=0.7)
          plt.colorbar()
          plt.title('t-SNE Visualization of Image Representations')
          plt.xlabel('t-SNE Dimension 1')
          plt.ylabel('t-SNE Dimension 2')
          plt.show()
          plt.savefig('t_sne.png')


