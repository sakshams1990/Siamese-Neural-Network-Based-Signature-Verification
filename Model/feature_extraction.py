import keras
import numpy as np
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import vgg19
from keras.preprocessing import image

from Utils import Utils

# ===========================================================================================
vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False)
vgg19_model = vgg19.VGG19(weights='imagenet', include_top=False)
resnet50_model = resnet50.ResNet50(weights='imagenet', include_top=False)

vgg16_preprocess_input = vgg16.preprocess_input
vgg19_preprocess_input = vgg19.preprocess_input
resnet50_preprocess_input = resnet50.preprocess_input


# ===========================================================================================

def get_image_embeddings(image_data, pre_trained_model, pre_trained_preprocess_input):
    img_array = image.img_to_array(image_data)
    img = pre_trained_preprocess_input(img_array)
    img = np.expand_dims(img, axis=0)
    model = keras.models.Sequential()
    model.add(pre_trained_model)
    model.add(keras.layers.Flatten())
    image_embedding = model.predict(img)
    return image_embedding


def get_cosine_similarity(image1, image2, pre_trained_model_option):
    image1_feature = None
    image2_feature = None
    cosine_similarity = 0.00
    if pre_trained_model_option == 'VGG16':
        image1_feature = get_image_embeddings(image1, vgg16_model, vgg16_preprocess_input)
        image2_feature = get_image_embeddings(image2, vgg16_model, vgg16_preprocess_input)
    elif pre_trained_model_option == 'VGG19':
        image1_feature = get_image_embeddings(image1, vgg19_model, vgg19_preprocess_input)
        image2_feature = get_image_embeddings(image2, vgg19_model, vgg19_preprocess_input)
    elif pre_trained_model_option == 'ResNet50':
        image1_feature = get_image_embeddings(image1, resnet50_model, resnet50_preprocess_input)
        image2_feature = get_image_embeddings(image2, resnet50_model, resnet50_preprocess_input)

    if image1_feature is not None and image2_feature is not None:
        cosine_similarity = Utils.cosine_similarity_score(image1_feature, image2_feature)
        cosine_similarity = round(cosine_similarity * 100, 2)
    return cosine_similarity


if __name__ == '__main__':
    pass
