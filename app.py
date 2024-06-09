import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

CLASES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
k = 2

# Define the prediction function which takes an image and returns the prediction.
def prediccion(file):
    imag = Image.fromarray(file)
    imag.save("imagen.jpg")

    img = image.load_img("./imagen.jpg", target_size=(28*k, 28*k))

    # Load the model outside of the prediction function.
    model = load_model("model_sign_language.h5")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predicted_batch = model.predict(img_array)
    predicted_class = np.argmax(predicted_batch)

    predicted_label = CLASES[predicted_class]

    return f"El número es: {predicted_label}"

demo = gr.Interface(
    fn=prediccion,
    inputs=[gr.Image()],
    outputs=[gr.Textbox()],
    title="Sign language number detector",
    description="This application uses machine learning models to detect sign language numbers in images.",
    examples=["three.jpg", "six.jpg", "eight.jpg", "nine.jpg"])

if __name__ == "__main__":
    demo.launch()