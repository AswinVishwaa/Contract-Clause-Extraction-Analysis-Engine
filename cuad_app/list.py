import google.generativeai as genai
import os

# Configure your API key
genai.configure(api_key="")

# List all models and their supported methods
print("Available models:")
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"Name: {model.name}")
        print(f"Description: {model.description}")
        print(f"Supported methods: {model.supported_generation_methods}\n")
