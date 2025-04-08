project-root/
│
├── config.json                    # Azure ML Workspace config (dummy sample below)
├── requirements.txt               # Python dependencies
├── scripts/
│   ├── train.py                   # DL training script
│   └── score.py                   # Inference script for deployment
│
├── data/
│   └── sample_images/            # Example data folder (optional)
│
├── register_model.py             # Register trained model
├── submit_training.py            # Submits training job to Azure ML
├── deploy_model.py               # Deploys model as web service
└── utils/
    └── __init__.py               # (optional) shared helper functions

# -------------------------------
# File: config.json (sample template)
{
  "subscription_id": "your-subscription-id",
  "resource_group": "your-resource-group",
  "workspace_name": "your-workspace-name",
  "tenant_id": "your-tenant-id",
  "client_id": "your-client-id",
  "client_secret": "your-client-secret"
}

# -------------------------------
# File: requirements.txt
azureml-core
azureml-train
azureml-sdk
azureml-dataprep
azureml-defaults
pandas
numpy
torch
torchvision
scikit-learn
Pillow
joblib

# -------------------------------
# File: scripts/train.py
import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from azureml.core import Run
from joblib import dump

run = Run.get_context()

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(os.path.join(args.data_path), transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    run.log("epoch_loss", running_loss / len(dataloader))

os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/resnet_model.pt")
dump(dataset.classes, "outputs/class_names.joblib")

# -------------------------------
# File: scripts/score.py
import torch
from torchvision import models, transforms
from azureml.core.model import Model
from PIL import Image
import io
from joblib import load

model = None
classes = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def init():
    global model, classes
    model_path = Model.get_model_path('resnet-defect-detector')
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Temporary placeholder
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    try:
        classes = load("/var/azureml-app/outputs/class_names.joblib")
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    except Exception as e:
        classes = ['class_0', 'class_1']
        print("Warning: Failed to load class names. Using default.", str(e))


def run(raw_data):
    image = Image.open(io.BytesIO(raw_data)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return {"prediction": classes[predicted.item()]}

# -------------------------------
# File: submit_training.py
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.data.dataset_factory import FileDatasetFactory

ws = Workspace.from_config()

dataset = FileDatasetFactory.from_files(path=(ws.get_default_datastore(), 'images/train/'))

compute_name = "gpu-cluster"
if compute_name not in ws.compute_targets:
    config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', max_nodes=1)
    compute_target = ComputeTarget.create(ws, compute_name, config)
    compute_target.wait_for_completion()
else:
    compute_target = ws.compute_targets[compute_name]

env = Environment.from_pip_requirements(name="dl-env", file_path="requirements.txt")

src = ScriptRunConfig(source_directory="scripts",
                      script="train.py",
                      arguments=['--data-path', dataset.as_mount()],
                      compute_target=compute_target,
                      environment=env)

exp = Experiment(workspace=ws, name="defect-detector-exp")
run = exp.submit(src)
run.wait_for_completion(show_output=True)

# -------------------------------
# File: register_model.py
from azureml.core import Workspace, Run

ws = Workspace.from_config()

exp = ws.experiments["defect-detector-exp"]
run = list(exp.get_runs())[0]  # most recent run

model = run.register_model(model_name="resnet-defect-detector",
                           model_path="outputs/resnet_model.pt")
print(f"Model registered: {model.name} | Version: {model.version}")

# -------------------------------
# File: deploy_model.py
from azureml.core import Workspace, Environment, Model, InferenceConfig
from azureml.core.webservice import AciWebservice

ws = Workspace.from_config()

model = Model(ws, name="resnet-defect-detector")
env = Environment.from_pip_requirements(name="dl-env", file_path="requirements.txt")

inference_config = InferenceConfig(entry_script="scripts/score.py", environment=env)
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(ws, "defect-detector-service", [model], inference_config, deployment_config)
service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")

# -------------------------------
# File: utils/__init__.py
# (Optional helper functions can go here)
