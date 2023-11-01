import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


model_path = "best_model.h5"
model = load_model(model_path)

name_map = dict(
    cbb="Bacterial Blight",
    cbsd="Brown Streak Disease",
    cgm="Green Mite",
    cmd="Mosaic Disease",
    healthy="Healthy",
)


class_labels = list(name_map.values())

st.title("Cassava Classification App")
st.markdown(
    "Using pretrained machine learning algorithms 🧠🤖 to  identify and classify various diseases affecting cassava through  their leaves. 🍀🔍"
)
# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img = image.load_img(
        uploaded_image, target_size=(224, 224)
    )  # Adjust target size based on your model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make predictions
    predictions = model.predict(img)

    # Display the predicted class and probabilities
    st.subheader("Prediction:")
    predicted_class_index = np.argmax(predictions)
    st.success(
        f"Predicted Disease ======>  '{class_labels[predicted_class_index].upper()}'"
    )

    class_probs = [i.round(4) for i in predictions[0]]
    print(class_probs)
    prob_dict = dict(zip(class_labels, class_probs))
    st.info(f"Probability Distribution ====>  {prob_dict}")
