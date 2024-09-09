import os
import zipfile
from flask import Flask, request, render_template, jsonify
import replicate
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from replicate.exceptions import ReplicateError
from config import FLUX_TRAINER_VERSION, STEPS, TRIGGER_WORD, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, MODEL

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set Replicate API token
replicate.api_token = os.getenv("REPLICATE_API_TOKEN")

# currently does not work as expected
def convert_to_png(image_file):
    try:
        img = Image.open(image_file)
        img_io = BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        return img_io, f"{os.path.splitext(image_file.filename)[0]}.png"
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("images")
        if uploaded_files:
            zip_path = "uploads/images.zip"
            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                for file in uploaded_files:
                    converted_file, filename = convert_to_png(file)
                    if converted_file:
                        zip_file.writestr(filename, converted_file.getvalue())
            
            return jsonify({"message": "Images uploaded, processed, and zipped successfully."})
    return render_template('upload.html')

@app.route('/fine-tune', methods=['POST'])
def start_fine_tuning():
    username = request.json.get('username')
    if username:
        zip_path = "uploads/images.zip"
        if os.path.exists(zip_path):
            base_model_name = f"flux-face-{username}"
            model = create_model(base_model_name)
            training = fine_tune_model(zip_path, model)
            return jsonify({
                "message": "Fine-tuning started",
                "training_url": f"https://replicate.com/p/{training.id}",
                "model_name": model.name,
                "training_id": training.id
            })
    return jsonify({"error": "Invalid username or zip file not found."}), 400

def create_model(base_model_name):
    owner = os.getenv("REPLICATE_ACCOUNT_NAME")
    model_name = base_model_name
    suffix = 1

    while True:
        try:
            existing_model = replicate.models.get(f"{owner}/{model_name}")
            if existing_model:
                suffix += 1
                model_name = f"{base_model_name}-{suffix}"
        except ReplicateError as e:
            if e.status == 404:
                # Model not found, we can create it
                break
            else:
                # Some other error occurred
                raise e

    model = replicate.models.create(
        owner=owner,
        name=model_name,
        visibility="public",
        hardware="gpu-t4", # Replicate will override this for fine-tuned models
        description="A fine-tuned FLUX.1 model"
    )

    print(f"Model created: {model.name}")
    print(f"Model URL: https://replicate.com/{model.owner}/{model.name}")

    return model

def fine_tune_model(zip_path, model):
    training = replicate.trainings.create(
        destination=f"{model.owner}/{model.name}",
        version=f"ostris/flux-dev-lora-trainer:{FLUX_TRAINER_VERSION}",
        input={
            "input_images": open(zip_path, "rb"),
            "steps": STEPS,
            "trigger_word": TRIGGER_WORD,
            "hf_repo_id": os.getenv("HF_REPO_ID"),
            "hf_token": os.getenv("HF_TOKEN"),
        }
    )

    print(f"Training started: {training.status}")
    print(f"Training URL: https://replicate.com/p/{training.id}")

    return training


@app.route('/generate', methods=['POST'])
def generate_image():
    model_name = request.json.get('model_name')
    model_version = request.json.get('model_version')
    prompt = request.json.get('prompt')
    
    if not model_name or not model_version or not prompt:
        return jsonify({"error": "Model name, model version, and prompt are required."}), 400

    try:
        output = replicate.run(
            f"{os.getenv('REPLICATE_ACCOUNT_NAME')}/{model_name}:{model_version}",
            input={
                "prompt": prompt,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "model": MODEL,
                "num_outputs": 1,
                "output_format": "png"
            }
        )
        
        if output and len(output) > 0:
            return jsonify({"image_url": output[0]})
        else:
            return jsonify({"error": "No image generated."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/training-status/<training_id>', methods=['GET'])
def training_status(training_id):
    try:
        training = replicate.trainings.get(training_id)
        if training.status == 'succeeded':
            model = replicate.models.get(training.destination)
            print(training.status)
            print(model.latest_version)
            print(type(model.latest_version))
            return jsonify({
                "status": training.status,
                "model_version": model.latest_version.id
            })
        else:
            return jsonify({"status": training.status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
