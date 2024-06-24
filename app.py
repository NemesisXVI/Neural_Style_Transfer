import streamlit as st
from PIL import Image
import numpy as np
import io
from model import run,load_image



# Streamlit app layout
st.title("Neural Style Transfer App")

st.sidebar.markdown("""
## Instructions:
1. Upload the content image.
2. Upload the style image.
3. Enter no. of epochs to run model for.
4. Click 'Generate' to create the stylized image.
5. Download the generated image.
""")

st.sidebar.header("Upload Images")
# Creating file uploader in sidebar
content_image_file = st.sidebar.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png"])
style_image_file = st.sidebar.file_uploader("Choose a style image...", type=["jpg", "jpeg", "png"])

if content_image_file is not None and style_image_file is not None:

    content_image = Image.open(content_image_file)
    style_image = Image.open(style_image_file)

    # Displaying style and content image in sidebar
    st.sidebar.image(content_image, caption='Content Image', use_column_width=True)
    st.sidebar.image(style_image, caption='Style Image', use_column_width=True)

    # Taking epochs as input
    epochs = st.sidebar.number_input("Number of epochs:", min_value=1, max_value=100000, value=10, step=1)

    # Calling run function from model.py to generate the generated_image.
    if st.sidebar.button('Generate'):
        with st.spinner('Generating Image...'):
            content_image = load_image(content_image_file)
            style_image = load_image(style_image_file)

            generated_image = run(content_image, style_image, epochs)

        # Displaying generated image.
        st.image(generated_image, caption='Generated Image', use_column_width=True)

        buf = io.BytesIO()
        generated_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Creating the download button.
        st.download_button(label="Download Image",
                           data=byte_im,
                           file_name="generated_image.png",
                           mime="image/png")


