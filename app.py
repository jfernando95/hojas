
from streamlit.components.v1 import html as st_html
import yaml
from yaml.loader import SafeLoader

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize

import streamlit_authenticator as stauth  # pip install streamlit-authenticator

st.set_page_config(
    page_title="Deteccion Tizon Tardio",
    page_icon="🍃",
    layout="wide",
)

# Ruta del modelo preentrenado
MODEL_PATH = 'models/model.h5'
width_shape = 224
height_shape = 224
names = ['Tomate__Sano', 'Tomate__Tizón_Tardío']

# cargar credenciales de usuario
with open('login/credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
)

#configurar y gestionar la autenticación de usuarios
nameU, authentication_status, username = authenticator.login('Login', 'main')

def registro(): #registro de usuario
    st.markdown("# Registrarse 📒")
    try:
        if authenticator.register_user('Register user',  preauthorization=False):
            st.success('User registered successfully')
            #actualizar archivo
            with open('login/credentials.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(e)

if authentication_status == False:
    st.error("Usuario o Contraseña incorrecto")
    #mostrar opcion de registro
    page_names_to_funcs = {
        "Login": lambda: None,
        "Registrarse": registro,
    }
    selected_page = st.selectbox("Registrarse:", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if authentication_status == None:
    st.warning("Ingresa tu usuario y contraseña")
    # mostrar opcion de registro
    page_names_to_funcs = {
        "Login": lambda: None,
        "Registrarse": registro,
    }
    selected_page = st.selectbox("Registrarse:", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if authentication_status: #Logueado correctamente

    # ---- SIDEBAR ----
    authenticator.logout("Salir", "sidebar")
    st.sidebar.write(f"<h1 style=\"color: #EBF5FB; \">Bienvenido {nameU}</h2>", unsafe_allow_html=True)
    imagen='assets/images/tomate.png'
    st.sidebar.image(imagen, caption="", use_column_width=True)

def model_prediction(img, model):
    img_resize = resize(img, (width_shape, height_shape))
    x = preprocess_input(img_resize * 255)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

def main():
    # Presentamos images y textos
    st.image("assets/images/unl_header.png", use_column_width=True)
    st.title("Detección de Enfermedades en Plantas de Tomate")
    st.write("<p>Detecta señales de tizón en las plantas de tomate a través del análisis de sus hojas.</p>",
             unsafe_allow_html=True)
    st.markdown("""---""")

    #Carga y prediccion con el Modelo
    model = None

    if model is None:
        model = load_model(MODEL_PATH)

    uploaded_images = st.file_uploader("Cargar imágenes:", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_images:
        image_list = [np.array(Image.open(image)) for image in uploaded_images]

    if st.button("Realizar Predicciones"):
        if uploaded_images:
            num_columns = 3  # Numero de columnas
            num_images = len(image_list)
            for i in range(0, num_images, num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    idx = i + j
                    if idx < num_images:
                        image = image_list[idx]
                        prediction = model_prediction(image, model)
                        class_name = names[np.argmax(prediction)]
                        confidence = np.max(prediction)
                        
                        if class_name == 'Tomate__Sano':
                            cols[j].write(f"Imagen {idx+1}: No es Tizón Tardío", use_container_width=True)
                        elif class_name == 'Tomate__Tizón_Tardío':
                            cols[j].write(f"Imagen {idx+1}: Sí es Tizón Tardío", use_container_width=True)
                        
                        cols[j].image(image, caption=f"Imagen {idx+1}: Clase - {class_name} -- Nivel de Confianza: {confidence:.2%} ", use_column_width=True)
                    




if authentication_status:
    # Cargar el archivo CSS
    st.markdown('<style>{}</style>'.format(open("css/estilo.css").read()), unsafe_allow_html=True)

    main()

