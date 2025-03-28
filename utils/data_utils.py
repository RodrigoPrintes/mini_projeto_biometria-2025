import tensorflow as tf
from tqdm import tqdm



def load_imagens(dados_dict, input_shape, augment=False):
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

            imagem = tf.keras.preprocessing.image.load_img(arquivo, target_size=input_shape[:2])
            imagem = tf.keras.preprocessing.image.img_to_array(imagem)

            imagens.append(imagem)
            rotulos.append(int(rotulo))

            if augment:
                # Aplicar aumento de dados
                augmented_image = datagen.random_transform(imagem)
                imagens.append(augmented_image)
                rotulos.append(int(rotulo))

    return tf.convert_to_tensor(imagens), tf.convert_to_tensor(rotulos)


