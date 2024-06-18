import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Load environment variables from the .env file
load_dotenv()

# Initialize API
api = HfApi()

# Set your Hugging Face user name and repository name
hf_user_name = "Eugenememe"
repository_name = "tortoise-vi-finetune"

# Construct the repository ID
repo_id = f"{hf_user_name}/{repository_name}"

# Set the path to your fine-tuned model
fine_tuned_model_path = "../experiments/custom_language_gpt/models/1500_gpt.pth"

# Load the Hugging Face authentication token from the .env file
hf_auth_token = os.getenv("HF_AUTH_TOKEN")

# Try to create a new repository (if it doesn't already exist)
try:
    api.create_repo(repo_id=repo_id, token=hf_auth_token, repo_type="model")
    print(f"Repository '{repo_id}' created successfully.")
except HfHubHTTPError as e:
    if e.response.status_code == 409:
        print(f"Repository '{repo_id}' already exists.")
    else:
        raise e

# Upload the model to the repository
model_url = api.upload_file(
    path_or_fileobj=fine_tuned_model_path,  # Ensure the variable is used correctly
    path_in_repo="custom_language_gpt.pth",
    repo_id=repo_id,
    token=hf_auth_token,  # Ensure the token is provided for the upload
    repo_type="model",
)

print(f"The fine-tuned model was uploaded to: {model_url}")
