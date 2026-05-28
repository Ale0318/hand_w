import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# =========================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================

st.set_page_config(
    page_title='DigitVision AI',
    layout='wide'
)

# =========================================
# CSS PERSONALIZADO
# =========================================

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #EAF2FF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #DCE8FF;
}

/* Título */
h1 {
    color: #1B2A41;
    text-align: center;
}

/* Subtítulo */
h3 {
    text-align: center;
    color: #324A5F;
}

/* Botones */
.stButton>button {
    background-color: #4F8CFF;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #2563EB;
    color: white;
}

/* Slider */
.stSlider {
    color: #2563EB;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# FUNCIÓN PREDICCIÓN
# =========================================

def predictDigit(image):

    model = tf.keras.models.load_model(
        "model/handwritten.h5"
    )

    image = ImageOps.grayscale(image)

    img = image.resize((28,28))

    img = np.array(
        img,
        dtype='float32'
    )

    img = img / 255

    plt.imshow(img)
    plt.show()

    img = img.reshape((1,28,28,1))

    pred = model.predict(img)

    result = np.argmax(pred[0])

    return result

# =========================================
# TÍTULO
# =========================================

st.title('DigitVision AI')

st.subheader(
    "Dibuja el dígito en el panel y presiona 'Predecir'"
)

# =========================================
# CONFIGURACIÓN DEL CANVAS
# =========================================

drawing_mode = "freedraw"

stroke_width = st.slider(
    'Selecciona el ancho de línea',
    1,
    30,
    15
)

stroke_color = '#000000'

bg_color = '#FFFFFF'

# =========================================
# CANVAS
# =========================================

canvas_result = st_canvas(

    fill_color="rgba(255, 165, 0, 0.3)",

    stroke_width=stroke_width,

    stroke_color=stroke_color,

    background_color=bg_color,

    height=250,

    width=250,

    key="canvas",
)

# =========================================
# BOTÓN PREDICCIÓN
# =========================================

if st.button('Predecir'):

    if canvas_result.image_data is not None:

        input_numpy_array = np.array(
            canvas_result.image_data
        )

        input_image = Image.fromarray(
            input_numpy_array.astype('uint8'),
            'RGBA'
        )

        input_image.save('prediction/img.png')

        img = Image.open(
            "prediction/img.png"
        )

        res = predictDigit(img)

        st.success(
            'El dígito es: ' + str(res)
        )

    else:

        st.warning(
            'Por favor dibuja un dígito.'
        )

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("Acerca de")

st.sidebar.markdown("""
Esta aplicación evalúa la capacidad de una Red Neuronal Artificial para reconocer dígitos escritos a mano.

### Tecnologías utilizadas:
- TensorFlow
- Streamlit
- NumPy
- PIL
- Canvas interactivo
""")
