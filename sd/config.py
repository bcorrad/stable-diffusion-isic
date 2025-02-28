import os, time, torch
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Initialize experiment folder 
EXPERIMENT_FOLDER = f"{time.strftime('%Y%m%d-%H%M%S')}/features"
os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

## DATABASE
ISIC_DB_FOLDER = "/repo/bonechi/Datasets/Dataset_Nevi/ISIC_Cleaned/Cleaned_Balanced/Solo_Nei_e_Melanomi"
ISIC_DB_IMAGES = "train_balanced/balanced/images"
ISIC_DB_METADATA_CSV = "/repo/bonechi/Datasets/Dataset_Nevi/ISIC_Cleaned/Cleaned_Balanced/Solo_Nei_e_Melanomi/train_balanced/training_cleaned.csv"

## CONFIGURATION
DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = False
if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")


## MODELS
SDM_ROOT = "/repo/cv_202/stable-diffusion-v1-5/"
CLIP_ROOT = "/repo/cv_202/clip-vit-base-patch32/"


## TEXT TO IMAGE 
# PROMPT = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution." # demo prompt (uncomment to enable)
# PROMPT = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution." # demo prompt (uncomment to enable)
PROMPT = ""
UNCOND_PROMPT = ""  # Also known as negative prompt


## CONDITIONAL IMAGE GENERATION
DO_CFG = True   
CFG_SCALE = 8   # Parameter to control the strength of the conditional information in the image generation process (higher values means more influence of the prompt) (min=1, max=14)


## IMAGE TO IMAGE GENERATION
I2I_GENERATION = True       # Enable or disable the image to image generation (if False, no input images will be used -- generate images from noise)
INPUT_IMAGE_PATHS = None    # If I2I_GENERATION is enabled and INPUT_IMAGE_PATHS None, the ISIC dataset will be used as input images
# ... OR you can Specify the path of the image(s) to be used as input for the image to image generation
# INPUT_IMAGE_PATHS = [os.path.join(PROJECT_ROOT, "images", "dog.jpg"), os.path.join(PROJECT_ROOT, "images", "cat.jpg")]  
# ... OR you can Read all the images given a folder (both .png and .jpg)
# INPUT_IMAGE_PATHS = list(Path(os.path.join(PROJECT_ROOT, "images")).rglob("*.jpg")) + list(Path(os.path.join(PROJECT_ROOT, "images")).rglob("*.png"))
# Number of dataset images to be given to SDM 
NUMBER_OF_IMAGES = 10    

# STRENGTH parameter: Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
STRENGTH = 0.9

## SAMPLER
SAMPLER = "ddpm"
NUM_INFERENCE_STEPS = 2         # no noise added
SEED = 42