import streamlit as st
import cv2
import numpy as np
import pickle
import streamlit.components.v1 as components
from model import EdgeNet


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="EdgeNet CNN",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# Load CSS
# --------------------------------------------------

with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )



with open("snow.html", "r", encoding="utf-8") as f:
    snow = f.read()

components.html(
    snow,
    height=0,
)
# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>❄️ EdgeNet CNN</h1>
        <p>Edge Detection Built Completely From Scratch Using NumPy</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = EdgeNet()

    with open("best_model.pkl", "rb") as f:
        weights = pickle.load(f)

    for i, layer in enumerate(model.layers):

        if hasattr(layer, "W"):
            layer.W = weights[f"W{i}"]

        if hasattr(layer, "b"):
            layer.b = weights[f"b{i}"]

    return model


model = load_model()


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original = image.copy()

    resized = cv2.resize(
        image,
        (128, 128)
    )

    x = resized.astype(np.float32) / 255.0

    x = x.transpose(2, 0, 1)

    x = np.expand_dims(
        x,
        axis=0
    )

    with st.spinner("Detecting Edges..."):

        pred = model.forward(x)

    edge = pred[0, 0]

    edge = (edge * 255).astype(np.uint8)

    edge = cv2.resize(
        edge,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "<h3 style='text-align:center;'>Original Image</h3>",
            unsafe_allow_html=True
        )

        st.image(
            original,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "<h3 style='text-align:center;'>Edge Prediction</h3>",
            unsafe_allow_html=True
        )

        st.image(
            edge,
            use_container_width=True
        )

    _, buffer = cv2.imencode(".png", edge)

    st.download_button(
        "⬇ Download Edge Image",
        buffer.tobytes(),
        file_name="edge_prediction.png",
        mime="image/png"
    )