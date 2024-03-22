import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
from io import BytesIO



mp_face_detection = mp.solutions.face_detection

def blurred_faces(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)
            img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w], (150, 150))
    return img

st.header("Face Anonymizer")
st.divider()
st.text("Upload Any Image With Face And It will Detect The Image And Blur The Face")
st.subheader("Upload An Image")
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])




if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Original Image", use_column_width=True)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Convert uploaded image to numpy array
        img_array = np.array(image)
        anonymized_image = blurred_faces(img_array, face_detection=face_detection)

    st.image(anonymized_image, caption="Anonymized Image", use_column_width=True)

    # Function to allow users to download the anonymized image
    def download_anonymized_image(anonymized_image):
        img_pil = Image.fromarray(anonymized_image)
        img_io = io.BytesIO()
        img_pil.save(img_io, format='PNG')
        img_io.seek(0)
        return img_io.getvalue()

    if st.button('Download Anonymized Image'):
        anonymized_img_data = download_anonymized_image(anonymized_image)
        st.download_button(label='Click here to download', data=anonymized_img_data, file_name='anonymized_image.png', mime='image/png')

