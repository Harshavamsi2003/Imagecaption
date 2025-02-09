import streamlit as st
import numpy as np
import torch
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from src.utils import seed_everything, Tokenizer, read_json, ImageCaptionGenerator
from src.utils.model import Transformer
from time import sleep

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 3rem;
            padding-right: 50%;
        }
    </style>
""", unsafe_allow_html=True)

# Global configuration
class GLOBAL:
    IMG_SIZE = 384
    SEED = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(ROOT_DIR, 'model')
    CONFIG_FILE = os.path.join(MODEL_DIR, 'config.json')
    WEIGHTS_FILE = os.path.join(MODEL_DIR, 'weights.pt')
    TOKENIZER_FILE = os.path.join(MODEL_DIR, 'vocab.pkl')

# Print the paths for debugging
print("Root directory:", GLOBAL.ROOT_DIR)
print("Model directory:", GLOBAL.MODEL_DIR)
print("Config file path:", GLOBAL.CONFIG_FILE)
print("Weights file path:", GLOBAL.WEIGHTS_FILE)
print("Tokenizer file path:", GLOBAL.TOKENIZER_FILE)

# Seed everything for reproducibility
seed_everything(GLOBAL.SEED)

# Load the model, tokenizer, and preprocessor
@st.cache_resource
def load(config):
    try:
        # Define the image preprocessor
        preprocessor = Compose([
            Resize((GLOBAL.IMG_SIZE, GLOBAL.IMG_SIZE)),
            ToTensor()
        ])

        # Load the tokenizer
        print("Loading tokenizer...")
        tokenizer: Tokenizer = Tokenizer.load(GLOBAL.TOKENIZER_FILE)
        print("Tokenizer loaded successfully. Vocabulary size:", len(tokenizer.vocab))

        # Initialize the model
        print("Initializing model...")
        model = Transformer(
            **config,
            vocab_size=len(tokenizer.vocab),
            device=GLOBAL.DEVICE,
            pad_idx=tokenizer.vocab.pad_idx
        ).to(GLOBAL.DEVICE)

        # Load the model weights
        print("Loading model weights...")
        model.load_state_dict(torch.load(GLOBAL.WEIGHTS_FILE, map_location=GLOBAL.DEVICE))
        print("Model weights loaded successfully.")

        return preprocessor, tokenizer, model

    except Exception as e:
        st.error(f"Error during loading: {e}")
        raise e

# Load the configuration file
try:
    config = read_json(GLOBAL.CONFIG_FILE)
    print("Config loaded successfully.")
except Exception as e:
    st.error(f"Error loading config file: {e}")
    st.stop()

# Load the preprocessor, tokenizer, and model
try:
    preprocessor, tokenizer, model = load(config)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Streamlit UI
with st.sidebar:
    st.markdown("### Upload your image from here")
    uploaded_file = st.file_uploader(label="", type=["png", "jpg", "jpeg"])

st.markdown("# Automatic Caption Generator")

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        np_image = np.array(image)
        st.image(np_image)

        # Button to generate caption
        predict_button = st.button("Generate caption")

        if predict_button:
            # Initialize the caption generator
            generator = ImageCaptionGenerator(
                model=model,
                tokenizer=tokenizer,
                img=image,
                max_len=int(config['max_len']),
                device=GLOBAL.DEVICE,
                preprocessor=preprocessor
            )

            # Generate and display the caption word by word
            t = st.empty()
            caption = []

            for word in generator:
                caption.append(word)
                caption[0] = caption[0].title()  # Capitalize the first word
                t.markdown("%s" % ' '.join(caption))
                sleep(0.05)

            # Display the final caption with a period
            t.markdown("%s" % ' '.join(caption) + '.')

    except Exception as e:
        st.error(f"Error generating caption: {e}")

else:
    st.info("No image is uploaded.")
