from utils.SplitDataset import SplitDataset
from  tensorflow.keras.models import load_model

import tensorflow as tf
import numpy as np


def load_image(image,input_shape):

    imagem = tf.keras.preprocessing.image.load_img(image, target_size=input_shape[:2])
    imagem = tf.keras.preprocessing.image.img_to_array(imagem)

    return tf.convert_to_tensor(imagem)





def main():
  #SplitDataset.unzip_dataset("dataset_train_iris_v2.zip")
  #SplitDataset.copy_files('dataset_train')

  input_shape = (320, 240, 3)

  data_dict = SplitDataset.refactorDatabase()
  
  m_model = load_model('models/modelo_cnn.h5')

  predictions = m_model.predict(load_image('dataset_train/102/102_1_2.bmp'))

  predicted_class_index = np.argmax(predictions)

  print(predicted_class_index)


 # print(data_dict)

if __name__ == "__main__":
    main()
