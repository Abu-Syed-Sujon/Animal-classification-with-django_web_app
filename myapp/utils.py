'myapp/utils.py'
from io import BytesIO
from keras.models import load_model
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_file):
    'Use BytesIO to read the content of the in-memory file'
    image_data = BytesIO(image_file.read())
    img = Image.open(image_data)
    img = img.resize((224, 224))  # adjust to your model's input size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_cat_or_dog(image_file):
    'predict image'
    model_path = 'myapp/Resnet152v2-01-0.96.hdf5'  # replace with your model path

    # Check if the model path is a string (file path) or a model object
    if isinstance(model_path, str):
        # If it's a string, load the model from the file path
        model = load_model(model_path)
    else:
        # If it's an already loaded model, use it directly
        model = model_path
    
    preprocessed_image = load_and_preprocess_image(image_file)
    prediction = model.predict(preprocessed_image)

    # Assuming 4 output classes: Cat, Dog, Cow, Butterfly
    class_labels = ['Butterfly','Cat','Cow', 'Dog' ]

    # Find the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    # Output raw probabilities for debugging
    print(f'Raw Probabilities: {prediction[0]}')

    # Check if the highest probability is above a threshold
    threshold = 0.5  # Adjust the threshold as needed
    if prediction[0][predicted_class_index] > threshold:
        return predicted_class
    else:
        print(f'Low Confidence Prediction: {predicted_class}')
        return 'Unknown'  # or any default class or label for low-confidence predictions
