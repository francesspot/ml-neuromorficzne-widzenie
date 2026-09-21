from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import io

app = FastAPI(title="Neuromorficzne widzenie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = np.load(io.BytesIO(content))
        tensor_data = torch.tensor(data, dtype=torch.float).to(device)
        
        if len(tensor_data.shape) == 4:
            tensor_data = tensor_data.unsqueeze(1) 
            
        return {
            "status": "success",
            "filename": file.filename,
            "predicted_class": "MockClass_Test",
            "tensor_shape": list(tensor_data.shape)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}