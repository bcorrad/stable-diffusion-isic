import model_loader
import pipeline
import config
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import os
from config import DEVICE, SDM_ROOT, EXPERIMENT_FOLDER, PROMPT, UNCOND_PROMPT, DO_CFG, CFG_SCALE, STRENGTH, SAMPLER, NUM_INFERENCE_STEPS, SEED
import logging 
  
# LOGGER OBJECT CONFIGURATIONogger object 
logger = logging.getLogger() 

# configuring the logger to display log message 
# along with log level and time  
logging.basicConfig(filename="message.log", 
                    format='%(asctime)s: %(levelname)s: %(message)s', 
                    level=logging.INFO) 

tokenizer = CLIPTokenizer(os.path.join(SDM_ROOT, "data/tokenizer_vocab.json"), merges_file=os.path.join(SDM_ROOT, "data/tokenizer_merges.txt"))
model_file = os.path.join(SDM_ROOT, "data/v1-5-pruned-emaonly.ckpt")
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

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
        # Read ISIC dataset from /repo/bonechi/Datasets/Dataset_Nevi/ISIC_Cleaned/Cleaned_Balanced/Solo_Nei_e_Melanomi as a list of images
        from config import ISIC_DB_FOLDER
        input_images_paths = list(Path(ISIC_DB_FOLDER).rglob("*.jpg"))
    else:
        raise ValueError("INPUT_IMAGE_PATHS must be a string or a list of strings")
    # Select a subset of the input images if the NUMBER_OF_IMAGES is specified in the config file
    if config.NUMBER_OF_IMAGES is not None:
        input_images_paths = input_images_paths[:config.NUMBER_OF_IMAGES]
else:
    logging.info("Image to image generation is disabled in the config file, so no input images will be used.")
    config.INPUT_IMAGE_PATHS = None

for image_path in input_images_paths:
    
    # Create a folder to store the output images in the experiment folder
    IMAGE_FOLDER = os.path.join(EXPERIMENT_FOLDER, image_path.stem)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    
    # Load the input image
    input_image = Image.open(image_path)

    # Generate an output image from the input image via SDM
    output_image = pipeline.generate(
        prompt=PROMPT,
        uncond_prompt=UNCOND_PROMPT,
        input_image=input_image,
        input_image_path=image_path,
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
    output_image_path = os.path.join(IMAGE_FOLDER, image_path.stem + ".png")
    Image.fromarray(output_image).save(output_image_path)
