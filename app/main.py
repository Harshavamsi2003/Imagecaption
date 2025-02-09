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

# Seed everything for reproducibility
seed_everything(8)  # Fixed seed for reproducibility

# Define paths
MODEL_DIR = os.path.join("app", "model")  # Path to the model folder
CONFIG_FILE = os.path.join(MODEL_DIR, "config.json")  # Path to config file
WEIGHTS_FILE = os.path.join(MODEL_DIR, "weights.pt")  # Path to weights file
TOKENIZER_FILE = os.path.join(MODEL_DIR, "vocab.pkl")  # Path to tokenizer file

# Print paths for debugging
print("Config file path:", CONFIG_FILE)
print("Weights file path:", WEIGHTS_FILE)
print("Tokenizer file path:", TOKENIZER_FILE)

# Load the model, tokenizer, and preprocessor
@st.cache_resource
def load(config):
    try:
        # Define the image preprocessor
        preprocessor = Compose([
            Resize((384, 384)),  # Fixed image size
            ToTensor()
        ])

        # Load the tokenizer
        print("Loading tokenizer...")
        tokenizer: Tokenizer = Tokenizer.load(TOKENIZER_FILE)
        print("Tokenizer loaded successfully. Vocabulary size:", len(tokenizer.vocab))

        # Initialize the model
        print("Initializing model...")
        model = Transformer(
            **config,
            vocab_size=len(tokenizer.vocab),
            device="cuda" if torch.cuda.is_available() else "cpu",
            pad_idx=tokenizer.vocab.pad_idx
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model weights with weights_only=False
        print("Loading model weights...")
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False))
        print("Model weights loaded successfully.")

        return preprocessor, tokenizer, model

    except Exception as e:
        st.error(f"Error during loading: {e}")
        raise e

# Load the configuration file
try:
    config = read_json(CONFIG_FILE)
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
                device="cuda" if torch.cuda.is_available() else "cpu",
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
