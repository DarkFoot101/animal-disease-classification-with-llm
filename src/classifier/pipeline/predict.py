import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from classifier.utils.common import load_model

class PredictionPipeline:
    def __init__(self, model_path, target_size=(224, 224)):
        self.model = load_model(model_path)
        self.target_size = target_size

    def predict(self, img_path):
        imagename = img_path
        test_image = image.load_img(imagename, target_size= (224, 224))
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(self.model.predict(test_image), axis= 1)
        print(result)

        if result[0] == 0:
            prediction = 'You are suffering from Giloma, get immediate treatment'
            return [{"image" : prediction}]
        elif result[0] == 1:
            prediction = 'You are Extremely Healthy, go ahead!'
            return [{"image" : prediction}]
        elif result[0] == 2:
            prediction = 'You are suffering from Meningioma, get immediate treatment'
            return [{"image" : prediction}]
        elif result[0] == 3:
            prediction = 'You are suffering from Pituitary Tumor, get immediate treatment'
            return [{"image" : prediction}]
        else:
            prediction = 'Error in prediction'
            return [{"image" : prediction}]