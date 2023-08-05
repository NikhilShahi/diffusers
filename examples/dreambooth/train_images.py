import argparse
import os
import subprocess
from dotenv import load_dotenv
from accelerate.utils import write_basic_config

load_dotenv()

DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"

def train_images(data_dir: str, class_dir: str = "default", output_dir: str = "default", model_name: str = DEFAULT_MODEL_NAME, num_class_images: int = 200, max_train_steps: int = None, hub_token: str = None, token_name: str = None, class_name: str = "man", cli: bool = False):
    write_basic_config()

    if token_name is None:
        raise Exception("No token name specified")

    MODEL_NAME = model_name
    INSTANCE_DIR = os.path.join("content", "data", data_dir)
    CLASS_DIR = os.path.join("content", "class", class_dir)
    OUTPUT_DIR = os.path.join("output", output_dir)

    if not os.path.exists(INSTANCE_DIR):
        os.makedirs(INSTANCE_DIR)
    if not os.path.exists(CLASS_DIR):
        os.makedirs(CLASS_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


    files = os.listdir(INSTANCE_DIR)
    number_of_files = len([f for f in files if os.path.isfile(os.path.join(INSTANCE_DIR, f))])
    if number_of_files == 0:
        raise Exception("No images found in the data directory")
    if max_train_steps is None:
        max_train_steps = number_of_files * 100
    if hub_token is None:
        token = os.environ.get("HUB_TOKEN")
        if token is None:
            raise Exception("No Hub token found")
        hub_token = token
    
    class_files = os.listdir(CLASS_DIR)
    for filename in class_files:
        file_path = os.path.join(CLASS_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    train_command = [
        "accelerate", "launch", "train_dreambooth.py",
        "--pretrained_model_name_or_path=" + MODEL_NAME,
        "--instance_data_dir=" + INSTANCE_DIR,
        "--class_data_dir=" + CLASS_DIR,
        "--output_dir=" + OUTPUT_DIR,
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--instance_prompt=" + f"a photo of {token_name} {class_name}",
        "--class_prompt=" + f"a photo of {class_name}",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=2",
        "--learning_rate=1e-6",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--num_class_images=" + str(num_class_images),
        "--max_train_steps=" + str(max_train_steps),
        "--hub_token=" + hub_token,
        "--train_text_encoder"
    ]
    if cli:
        with subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
    else:
        result = subprocess.run(train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f"Error executing train command: {result.stderr.decode()}")
    
    generate_ckpt(output_dir)
    
        
def generate_ckpt(output_dir: str):
    MODEL_PATH = os.path.join("output", output_dir)
    CHECKPONT_PATH = os.path.join("output", output_dir, f"{output_dir}.ckpt")
    generate_script_path = os.path.join("..", "..", "scripts", "convert_diffusers_to_original_stable_diffusion.py")
    generate_command = [
        "python", generate_script_path,
        "--model_path=" + MODEL_PATH,
        "--checkpoint_path=" + CHECKPONT_PATH
    ]
    result = subprocess.run(generate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(result.stdout.decode())
    else:
        raise Exception(f"Encountered error while generating ckpt file: {result.stderr.decode()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="Name of directory where images are stored")
    parser.add_argument("--class_dir", default="default", type=str, required=True, help="Name of directory to store class images in")
    parser.add_argument("--output_dir", default="default", type=str, required=True, help="Name of directory to store output in")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, type=str, help="Name of base model")
    parser.add_argument("--num_class_images", default=200, type=int)
    parser.add_argument("--max_train_steps", default=None, type=int)
    parser.add_argument("--hub_token", default=None, type=str)
    parser.add_argument("--token_name", default=None, type=str, required=True)
    parser.add_argument("--class_name", default="man", type=str, required=True)

    args = parser.parse_args()

    train_images(data_dir=args.data_dir, class_dir=args.class_dir, output_dir=args.output_dir, model_name=args.model_name, num_class_images=args.num_class_images, max_train_steps=args.max_train_steps, hub_token=args.hub_token, token_name=args.token_name, class_name=args.class_name, cli=True)

