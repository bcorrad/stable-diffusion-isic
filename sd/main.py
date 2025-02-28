import model_loader
import pipeline
import config
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import os
from config import DEVICE, SDM_ROOT, EXPERIMENT_FOLDER, PROMPT, UNCOND_PROMPT, DO_CFG, CFG_SCALE, STRENGTH, SAMPLER, NUM_INFERENCE_STEPS, SEED
from config import ISIC_DB_FOLDER, ISIC_DB_IMAGES, ISIC_DB_METADATA_CSV
import logging 
  
# LOGGER OBJECT CONFIGURATION
logger = logging.getLogger() 

# configuring the logger to display log message 
# along with log level and time  
logging.basicConfig(filename="message.log", 
                    format='%(asctime)s: %(levelname)s: %(message)s', 
                    level=logging.INFO) 

tokenizer = CLIPTokenizer(os.path.join(SDM_ROOT, "data/tokenizer_vocab.json"), merges_file=os.path.join(SDM_ROOT, "data/tokenizer_merges.txt"))
model_file = os.path.join(SDM_ROOT, "data/v1-5-pruned-emaonly.ckpt")
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

import pandas as pd

## DATASET PREPARATION
if config.I2I_GENERATION: 
    if config.INPUT_IMAGE_PATHS is not None:
        # Use the input images specified in the config file
        from config import INPUT_IMAGE_PATHS
        if type(INPUT_IMAGE_PATHS) is str:
            input_images_paths = [INPUT_IMAGE_PATHS]
        elif type(INPUT_IMAGE_PATHS) is list:
            input_images_paths = INPUT_IMAGE_PATHS
        else:
            raise ValueError("INPUT_IMAGE_PATHS must be a string or a list of strings")
    elif config.INPUT_IMAGE_PATHS is None:
        logging.info("No input images specified in the config file, so the ISIC dataset will be used as input images.")
        training_df = pd.read_csv(ISIC_DB_METADATA_CSV)
        # Columns in the dataframe related to the image name and target
        targets_col = [col for col in training_df.columns if "target" in col]
        names_col = [col for col in training_df.columns if "image" in col or "name" in col]
        cols_subset = names_col + targets_col
        # Save image_id and target columns from training_df to a dictionary
        training_dictionary = training_df[cols_subset].to_dict(orient="records")
        # Create a list of image paths from the training_dictionary adding the ISIC_DB_FOLDER to the image_id
        input_images_paths = [os.path.join(ISIC_DB_FOLDER, ISIC_DB_IMAGES, f"{image[names_col[0]]}.jpg") for image in training_dictionary]
    else:
        raise ValueError("INPUT_IMAGE_PATHS must be a string or a list of strings")
    # Select a subset of the input images if the NUMBER_OF_IMAGES is specified in the config file
    if config.NUMBER_OF_IMAGES is not None:
        input_images_paths = input_images_paths[:config.NUMBER_OF_IMAGES]
else:
    print("Image to image generation is disabled in the config file, so no input images will be used.")
    logging.info("Image to image generation is disabled in the config file, so no input images will be used.")
    config.INPUT_IMAGE_PATHS = None

for image_idx, image_path in enumerate(input_images_paths):
    
    print(f"Processing image {image_idx+1}/{len(input_images_paths)}: {image_path}")
    logging.info(f"Processing image {image_idx+1}/{len(input_images_paths)}: {image_path}")
    
    # Read the target of the image from the dictionary (1: Melanoma, 0: Nevi)
    image_target = [image for image in training_dictionary if image[names_col[0]] == Path(image_path).stem][0][targets_col[0]]
    
    # Create a folder to store the output images in the experiment folder
    IMAGE_FOLDER = os.path.join(EXPERIMENT_FOLDER, f"{Path(image_path).stem}_cl_{image_target}")
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    
    # Load the input image
    input_image = Image.open(image_path)

    # Generate an output image from the input image via SDM
    output_image = pipeline.generate(
        prompt=PROMPT,
        uncond_prompt=UNCOND_PROMPT,
        input_image=input_image,
        input_image_path=f"{Path(image_path).stem}_cl_{image_target}",
        strength=STRENGTH,
        do_cfg=DO_CFG,
        cfg_scale=CFG_SCALE,
        sampler_name=SAMPLER,
        n_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # Combine the input image and the output image into a single image.
    Image.fromarray(output_image)
    
    # Save the output image to a file.
    output_image_path = os.path.join(IMAGE_FOLDER, Path(image_path).stem + ".png")
    Image.fromarray(output_image).save(output_image_path)
