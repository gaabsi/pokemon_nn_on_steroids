import os

import streamlit as st

from evaluation import PredictEval

st.title("Pokemon NN on Steroids")


@st.cache_resource
def load_model():
    BASE_DIR = os.path.expanduser("~/pokemon_nn_on_steroids")
    checkpoint_path = os.path.join(BASE_DIR, "models/version_0/version_0.ckpt")
    feature_extractor = "facebook/deit-base-distilled-patch16-224"
    dir_classes = os.path.join(BASE_DIR, "data/split_data/train")

    return PredictEval(
        checkpoint_path=checkpoint_path,
        feature_extractor=feature_extractor,
        dir_classes=dir_classes,
    )


uploaded_file = st.file_uploader("Upload une image", type=["jpg", "png", "webp"])

if uploaded_file:
    st.image(uploaded_file)

    if st.button("Prédire"):
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        predictor = load_model()
        nom_pred, _, conf, _ = predictor.predict(temp_path)

        st.success(f"**Pokémon:** {nom_pred}")
        st.info(f"**Confiance:** {conf*100:.1f}%")

        os.remove(temp_path)
