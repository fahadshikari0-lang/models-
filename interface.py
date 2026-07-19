import streamlit as st
import requests
from PIL import Image
import os
import time

st.set_page_config(
    page_title="YOLOv11 Object Detection",
    page_icon="📦",
    layout="wide"
)

st.title("📦 YOLOv11 Object Detection")

uploaded_file = st.file_uploader(
    "Upload an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    file_type = uploaded_file.type

    # ================= IMAGE =================
    if file_type.startswith("image"):

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Detect Objects"):

            with st.spinner("Detecting objects..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:8003/detect-image",
                    files=files
                )

            if response.status_code == 200:

                data = response.json()

                objects = data["objects"]

                if len(objects) == 0:
                    st.warning("No objects detected.")
                else:
                    st.success(f"{len(objects)} object(s) detected")

                    for obj in objects:
                        st.write(f"### {obj['label']}")
                        st.write(f"Confidence: {obj['confidence']:.2f}")
                        st.write(f"Bounding Box: {obj['box']}")
                        st.divider()

            else:
                st.error("Image Detection API Error")

    # ================= VIDEO =================
    elif file_type.startswith("video"):

        st.video(uploaded_file)

        if st.button("Detect Video"):

            with st.spinner("Processing video..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:8006/detect-video",
                    files=files
                )

            if response.status_code == 200:

                # Har baar unique filename
                output_path = f"detected_video_{int(time.time())}.mp4"

                with open(output_path, "wb") as f:
                    f.write(response.content)

                st.success("Detection Completed!")

                st.subheader("Detected Video")

                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇ Download Detected Video",
                        f,
                        file_name=os.path.basename(output_path),
                        mime="video/mp4"
                    )

            else:
                st.error("Video Detection API Error")

else:
    st.info("Please upload an image or video.")