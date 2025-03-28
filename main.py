from utils.SplitDataset import SplitDataset
from utils.data_utils import load_imagens
import numpy as np
from sklearn.model_selection import train_test_split
from  train import  Train_cnn
from evaluate import  Evaluation
from Segmentatation.Segmentation import  Segmentation
import keras

def main():
    #SplitDataset.download_dataset()
    #SplitDataset.unzip_dataset()

    data_dict = SplitDataset.refactorDatabase()
    input_shape = (320, 240, 3)
    # Carregar imagens com aumento de dados
    augmented_imagens, rotulos = load_imagens(data_dict, augment=False, input_shape=input_shape)

    # Hiperparâmetros

    num_classes = len(data_dict.keys())

    rotulos -= 1
    augmented_imagens = augmented_imagens / 255.0

    imagens_np = augmented_imagens.numpy()
    rotulos_np = rotulos.numpy()

    num_classes = len(data_dict.keys())

    x_train, x_test, y_train, y_test = train_test_split(imagens_np, rotulos_np, test_size=0.2, random_state=42)

    train = Train_cnn(input_shape=input_shape,num_classes=num_classes, model=True)

    history = train.train_init(x_train, x_test, y_train, y_test)

    data_train = train.train_metrics(history)
    print(data_train)

    model = train.train_getModel()

    evaluate = Evaluation(model, x_test, y_test)
    evaluate.evaluate_accuracy()
    evaluate.evaluate_history(history)
    evaluate.evaluate_scores()
    evaluate.evaluate_get_gradcam(model, 'dense_3')
    evaluate.evaluate_t_sne(num_classes= num_classes,dict_classes=data_dict,input_shape =input_shape,last_conv_layer_name='dense 3')
    evaluate.evaluate_confusion_matrix()


    segmentation = Segmentation(data_dict)

    images, rotulos = segmentation.get_parametos()
    imagens_np = images.numpy()
    rotulos_np = rotulos.numpy()

    # Fazer o split dos dados em conjunto de treinamento e teste
    sx_train, sx_test, sy_train, sy_test = train_test_split(imagens_np, rotulos_np, test_size=0.2, random_state=42)

    smodel = keras.models.load_model('models/modelo_cnn.h5')


    s_train_model = segmentation.seg_train(smodel)

    shistory = model.fit(sx_train, sy_train, epochs=120, validation_data=(sx_test, sy_test), batch_size=128)

    eval_seg = Evaluation(model, x_test, y_test)
    eval_seg.evaluate_accuracy()
    eval_seg.evaluate_history(shistory)
    eval_seg.evaluate_scores()

    evaluate.evaluate_confusion_matrix()


if __name__ == "__main__":
    main()
