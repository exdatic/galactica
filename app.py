from transformers import pipeline
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("text-generation", model="facebook/galactica-6.7b", device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    tokens = model_inputs.get('max_new_tokens', 100)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model(prompt, max_new_tokens=tokens)

    # Return the results as a dictionary
    return result
