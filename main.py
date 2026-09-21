from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import io

from snn_model import FranciszekSCNN 

app = FastAPI(title="Neuromorficzne widzenie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FranciszekSCNN(
    beta=0.8417870100435023, 
    threshold=0.8033307250431925,
    dropout_p=0.213169938848635,
    slope=15,
    population=10
).to(device)

model.load_state_dict(torch.load('franciszek_scnn_53.pth', map_location=device))
model.eval() 

NCALTECH_CLASSES = [
    'BACKGROUND_Google', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = np.load(io.BytesIO(content))
        tensor_data = torch.tensor(data, dtype=torch.float).to(device)
        
        if len(tensor_data.shape) == 4:
            tensor_data = tensor_data.unsqueeze(1) 
            
        with torch.no_grad():
            spk_out, _ = model(tensor_data)
            
            spike_count = spk_out.sum(dim=0)
            
            spike_count_population = spike_count.view(tensor_data.size(1), 101, 10).sum(dim=2)
            
            predicted_idx = spike_count_population.argmax(dim=1).item()
            
            predicted_class_name = NCALTECH_CLASSES[predicted_idx] if predicted_idx < len(NCALTECH_CLASSES) else f"Class_{predicted_idx}"

        return {
            "status": "success",
            "filename": file.filename,
            "predicted_class": predicted_class_name,
            "tensor_shape": list(tensor_data.shape)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}