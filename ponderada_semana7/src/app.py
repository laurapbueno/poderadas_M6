from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Carregar os modelos
model_convolucional = tf.keras.models.load_model('modelos/pesos.h5')
model_linear = tf.keras.models.load_model('modelos/pesos_linear.h5')

@app.route('/input', methods=['GET', 'POST'])
def index():
    previsao_conv = None
    previsao_linear = None

    if request.method == 'POST':
        # Receber a imagem do formulário
        imagem = request.files['imagem']
        img = cv2.imdecode(np.frombuffer(imagem.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img / img.max()
        img = img.reshape(1, 28, 28, 1)

        # Previsão com o modelo convolucional
        previsao_conv = np.argmax(model_convolucional.predict(img))

        # Previsão com o modelo linear
        img_linear = img.reshape(1, 28 * 28)  # Redimensionar para modelo linear
        previsao_linear = np.argmax(model_linear.predict(img_linear))

    return render_template('index.html', previsao_conv=previsao_conv, previsao_linear=previsao_linear)

if __name__ == '__main__':
    app.run(debug=True)
