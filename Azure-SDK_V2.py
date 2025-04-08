project-root/
│
├── config.json                    # Azure ML Workspace config
├── requirements.txt               # Python dependencies
├── scripts/
│   ├── train.py                   # TensorFlow training script (SDK v2)
│   └── score.py                   # Inference script for deployment
│
├── data/
│   └── sample_images/            # Example data folder (optional)
│
├── register_model.py             # Register trained model (SDK v2)
├── submit_training.py            # Submits training job to Azure ML (SDK v2)
├── deploy_model.py               # Deploys model as web service (SDK v2)
└── utils/
    └── __init__.py               # (optional) shared helper functions

# -------------------------------
# File: requirements.txt
azure-ai-ml
azure-identity
pandas
numpy
tensorflow
scikit-learn
Pillow
joblib

# -------------------------------
# File: scripts/train.py
import argparse
import os
import tensorflow as tf
import numpy as np
from joblib import dump
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Run
from azure.identity import DefaultAzureCredential

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True)
args = parser.parse_args()

img_height, img_width = 180, 180
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_path,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=5)

os.makedirs('outputs', exist_ok=True)
model.save('outputs/tf_defect_model')
dump(class_names, 'outputs/class_names.joblib')

# -------------------------------
# File: scripts/score.py
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from joblib import load
import os

model = None
class_names = None
img_height, img_width = 180, 180

def init():
    global model, class_names
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "tf_defect_model")
    model = tf.keras.models.load_model(model_path)
    try:
        class_names = load(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "class_names.joblib"))
    except Exception:
        class_names = ['class_0', 'class_1']

def run(raw_data):
    image = Image.open(io.BytesIO(raw_data)).convert("RGB")
    image = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    return {"prediction": predicted_class, "confidence": float(np.max(score))}

# -------------------------------
# File: submit_training.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, CommandJob, CodeConfiguration
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

ml_client = MLClient(DefaultAzureCredential(), "<subscription_id>", "<resource_group>", "<workspace>")

# Upload and register data
input_data = ml_client.data.create_or_update(
    Data(
        name="defect-images",
        description="Training images",
        path="./data/sample_images",
        type=AssetTypes.URI_FOLDER,
    )
)

# Create environment
env = Environment(image="mcr.microsoft.com/azureml/tensorflow-2.9-ubuntu20.04-py38-cpu:latest")

# Submit training job
job = CommandJob(
    code=CodeConfiguration(code="scripts", script="train.py"),
    environment=env,
    inputs={"data-path": Input(type=AssetTypes.URI_FOLDER, path=input_data.id)},
    compute="cpu-cluster",
    experiment_name="tf-defect-exp",
    display_name="tensorflow-training-job",
)

ml_client.jobs.create_or_update(job)

# -------------------------------
# File: register_model.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

ml_client = MLClient(DefaultAzureCredential(), "<subscription_id>", "<resource_group>", "<workspace>")

model = ml_client.models.create_or_update(
    Model(
        path="outputs/tf_defect_model",
        name="tf-defect-detector",
        type="mlflow_model",
        description="TF defect classifier"
    )
)

print(f"Registered model: {model.name} version: {model.version}")

# -------------------------------
# File: deploy_model.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

ml_client = MLClient(DefaultAzureCredential(), "<subscription_id>", "<resource_group>", "<workspace>")

endpoint = ManagedOnlineEndpoint(name="tf-endpoint", auth_mode="key")
ml_client.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="tf-deploy",
    endpoint_name="tf-endpoint",
    model="tf-defect-detector:1",
    code_path="scripts",
    scoring_script="score.py",
    environment="AzureML-tensorflow-2.9-ubuntu20.04-py38-cpu:latest",
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

ml_client.begin_create_or_update(deployment).result()

ml_client.online_endpoints.begin_traffic_update(
    name="tf-endpoint", traffic={"tf-deploy": 100}
).result()

print("Deployment complete.")
