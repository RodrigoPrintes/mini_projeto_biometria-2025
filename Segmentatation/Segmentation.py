import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from keras import layers, models

class Segmentation:
    def __init__(self, dict_images):
        self.dict_Segmentation = {}
        self.dict_Segmentation = self.save_SagmentationImaes(dict_images)

        self.augmented_imagens, self.rotulos = self.load_imagens(self.dict_Segmentation, augment=False)
        
        self.rotulos -= 1
        self.augmented_imagens = self.augmented_imagens / 255.0

    def get_parametos(self):

        return self.augmented_imagens, self.rotulos
    def load_imagens(dados_dict, augment=False):

        imagens = []
        rotulos = []

        # Configuração do gerador de aumento de dados
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        for rotulo, arquivos in tqdm(dados_dict.items()):

            for arquivo in arquivos:
                if ".bmp" not in arquivo:
                    continue
                try:
                    img = cv2.imread(arquivo)

                    imagem = tf.keras.preprocessing.image.load_img(arquivo, target_size=img[:2])
                    imagem = tf.keras.preprocessing.image.img_to_array(imagem)

                    imagens.append(imagem)
                    rotulos.append(int(rotulo))

                    if augment:
                        # Aplicar aumento de dados
                        augmented_image = datagen.random_transform(imagem)
                        imagens.append(augmented_image)
                        rotulos.append(int(rotulo))

                except Exception as e:

                    print("An exception occurred:", e, arquivo)


        return tf.convert_to_tensor(imagens), tf.convert_to_tensor(rotulos)

    def save_SagmentationImaes(self,data_dict):

        dict_test = {}
        dict_classes = {}

        for rotulo, arquivos in tqdm(data_dict.items()):

            count = 0
            for arquivo in arquivos:

                if ".bmp" not in arquivo:
                    continue
                try:
                    img = self.getImageSegmetation(arquivo)

                    if rotulo not in dict_classes:

                        dict_classes.setdefault(rotulo, [arquivo])
                        cv2.imwrite(arquivo, img)
                        count += 1
                    else:
                        dict_classes[rotulo].append(arquivo)

                except:

                    if rotulo not in dict_test:
                        dict_test.setdefault(rotulo, [f"{arquivo}-NOK"])
                    else:
                        dict_test[rotulo].append(f"{arquivo}-NOK")
        return dict_classes



    def daugman_normalizaiton(image, height, width, r_in, r_out, cx, cy):

        thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
        r_out = r_out + r_in
        # Create empty flatten image
        flat = np.zeros((height, width, 3), np.uint8)
        circle_x = int(image.shape[0] / 2)
        circle_y = int(image.shape[1] / 2)

        for i in range(width):
            for j in range(height):
                theta = thetas[i]  # value of theta coordinate
                r_pro = j / height  # value of r coordinate(normalized)

                # get coordinate of boundaries
                Xi = circle_x + r_in * np.cos(theta)
                Yi = circle_y + r_in * np.sin(theta)
                Xo = circle_x + r_out * np.cos(theta)
                Yo = circle_y + r_out * np.sin(theta)

                # the matched cartesian coordinates for the polar coordinates
                Xc = (1 - r_pro) * Xi + r_pro * Xo
                Yc = (1 - r_pro) * Yi + r_pro * Yo

                color = image[int(Xc)][int(Yc)]  # color of the pixel

                flat[j][i] = color
        return flat

    def getImageradios(img_path):
        img = cv2.imread(img_path, 0)
        img = cv2.medianBlur(img, 5)

        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)

        height, width = img.shape
        task = []
        mask = np.zeros((height, width), np.uint8)
        if circles is not None:

            circles2 = circles[0, :].astype(int)

            for (cx, cy, r) in circles2:
                cv2.circle(cimg, (cx, cy), r, (0, 0, 0))
                task = [cx, cy, r]
        return task

    def getImageSegmetation(self, img_path):

        cx, cy, radios = self.getImageradios(img_path)

        img = cv2.imread(img_path)

        image_nor = self.daugman_normalizaiton(img, 60, img.shape[1], radios, int(1.5 * radios), cx, cy)

        return image_nor

    @staticmethod
    def seg_train(model):

        model_teste = models.Sequential([
            model,
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.19),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.19),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(99, activation='softmax')

        ])

        # Compile o modelo
        model_teste.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        return model_teste
