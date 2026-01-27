import os
import shutil
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.linear_model import LogisticRegression
from pocketbase import PocketBase
from pocketbase.client import FileUpload
from dagster import op, job, sensor, RunRequest, SkipReason, DefaultSensorStatus, Definitions

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION (Move these to Env Vars in production) ---
POCKETBASE_URL = "http://127.0.0.1:8090" 
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL") # "your_admin_email@example.com"
ADMIN_PASS = os.getenv("ADMIN_PASS") # "your_admin_password"
DINOV3_MODEL_PATH = "../assets/dinov3_feature_extractor.onnx" 

# Constants
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- ML HELPERS ---
def preprocess_image(img_pil, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    w, h = img_pil.size
    h_patches = image_size // patch_size
    w_patches = int((w * image_size) / (h * patch_size))
    target_size = (h_patches * patch_size, w_patches * patch_size)
    resized_img = img_pil.resize((target_size[1], target_size[0]), Image.Resampling.BICUBIC)
    img_np = np.array(resized_img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized_img = (img_np - mean) / std
    return normalized_img.transpose(2, 0, 1)[np.newaxis, :, :]

def resize_mask(mask_pil, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    w, h = mask_pil.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    target_size = (h_patches * patch_size, w_patches * patch_size)
    return TF.to_tensor(TF.resize(mask_pil, target_size))

# --- DAGSTER OP (The Heavy Lifting) ---

@op(config_schema={"record_id": str})
def train_classifier_op(context):
    """
    Downloads images, trains the classifier, and uploads the ONNX.
    """
    record_id = context.op_config["record_id"]
    context.log.info(f"🚀 Starting training for record: {record_id}")

    # Connect to PocketBase
    pb = PocketBase(POCKETBASE_URL)
    pb.admins.auth_with_password(ADMIN_EMAIL, ADMIN_PASS)
    
    # Fetch record details
    record = pb.collection('datasets').get_one(record_id)
    
    # Update status to 'training'
    pb.collection('datasets').update(record_id, {"status": "training"})

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download Images
            image_files = []
            for filename in record.images:
                file_url = pb.get_file_url(record, filename)
                import requests
                r = requests.get(file_url)
                local_path = os.path.join(temp_dir, filename)
                with open(local_path, 'wb') as f:
                    f.write(r.content)
                image_files.append(local_path)
            
            context.log.info(f"Downloaded {len(image_files)} images.")

            # Extract Features (DINOv3) & Train

            ort_sess = ort.InferenceSession(DINOV3_MODEL_PATH)
            input_name = ort_sess.get_inputs()[0].name
            
            xs = []
            ys = []
            
            patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
            patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))
            
            for img_path in image_files:
                with Image.open(img_path) as img:
                    if img.mode != 'RGBA': continue
                    img.load()
                    rgb_img = img.convert("RGB")
                    mask_img = img.split()[-1]
                    
                    # Ground Truth
                    mask_tensor = resize_mask(mask_img)
                    mask_quantized = patch_quant_filter(mask_tensor).squeeze().view(-1).detach().numpy()
                    ys.append(mask_quantized)
                    
                    # Input
                    img_input = preprocess_image(rgb_img)
                    
                    # Inference
                    feats = ort_sess.run(None, {input_name: img_input})[0]
                    feats = feats.squeeze(0) # [Num_Patches, Embed_Dim]
                    xs.append(feats)

            # Concat
            xs = np.concatenate(xs, axis=0)
            ys = np.concatenate(ys, axis=0)
            
            # Filter (Keep only clear foreground/background)
            idx = (ys < 0.01) | (ys > 0.99)
            xs_clean = xs[idx]
            ys_clean = ys[idx]

            # Train Classifier
            print("   Training Logistic Regression...")
            # Fixed C=1.0 is usually fine for few-shot, or use simplified search
            clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
            clf.fit(xs_clean, (ys_clean > 0).astype(int))
            
            # Export using skl2onnx (Cleaner and lighter than `torch.onnx.export`)
            print("   Exporting Classifier via skl2onnx...")
    
            # feature_dim is the size of the DINOv3 embeddings
            feature_dim = xs_clean.shape[1]
            # Define the input type and shape: 
            # [None, feature_dim] allows for a dynamic number of patches 
            initial_type = [('patch_features', FloatTensorType([None, feature_dim]))]

            # Convert the scikit-learn model to ONNX
            options = {type(clf): {'zipmap': False}}
            onx = to_onnx(
                clf, 
                initial_types=[('patch_features', FloatTensorType([None, feature_dim]))],
                options=options, # This forces a simple Tensor output
                target_opset=17
            )

            output_path = os.path.join(temp_dir, "classifier.onnx")
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())

            print(f"✅ Model saved to {output_path}")
            # Upload Result
            context.log.info("Uploading classifier to PocketBase...")
            pb.collection('datasets').update(
                record_id,
                {
                    "status": "ready",
                    "classifier_file": FileUpload(
                        ("classifier.onnx", open(output_path, "rb"))
                    )
                }
            )
            context.log.info(f"✅ Job Complete for {record_id}")

    except Exception as e:
        context.log.error(f"❌ Training failed: {e}")
        pb.collection('datasets').update(record_id, {"status": "failed"})
        raise e

# --- DAGSTER JOB ---

@job
def training_job():
    train_classifier_op()

# --- DAGSTER SENSOR (The Polling Logic) ---

@sensor(job=training_job, minimum_interval_seconds=10, default_status=DefaultSensorStatus.RUNNING)
def pocketbase_sensor(context):
    """
    Polls PocketBase for datasets with status='pending'.
    Triggers a run for each one.
    """
    pb = PocketBase(POCKETBASE_URL)
    try:
        pb.admins.auth_with_password(ADMIN_EMAIL, ADMIN_PASS)
        
        # Filters and sorts must be inside 'query_params'
        records = pb.collection('datasets').get_list(
            page=1, 
            per_page=10, 
            query_params={
                "filter": 'status = "pending"',
                "sort": 'created'
            }
        )
        
        if not records.items:
            yield SkipReason("No pending datasets found.")
            return

        for record in records.items:
            # Create a unique run key using the record ID
            run_key = f"train_{record.id}"
            
            yield RunRequest(
                run_key=run_key,
                run_config={
                    "ops": {
                        "train_classifier_op": {
                            "config": {"record_id": record.id}
                        }
                    }
                }
            )
            
    except Exception as e:
        context.log.error(f"Failed to connect to PocketBase: {e}")
        # Use Yield SkipReason to avoid crashing the sensor daemon
        yield SkipReason(f"Connection error: {e}")

# --- DAGSTER DEFINITIONS ---
defs = Definitions(
    jobs=[training_job],
    sensors=[pocketbase_sensor],
)