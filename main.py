from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import tempfile
import imageio
import os
import base64

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

def process_mp4_to_snn(video_path, target_size=(80, 80), diff_threshold=8):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return None, None
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, target_size)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    snn_tensor_list = []
    visualization_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, target_size)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        curr_gray_int = curr_gray.astype(np.int16)
        prev_gray_int = prev_gray.astype(np.int16)
        diff = curr_gray_int - prev_gray_int
        
        spikes_on = (diff > diff_threshold).astype(np.float32)
        spikes_off = (diff < -diff_threshold).astype(np.float32)
        
        vis_dvs = np.ones(target_size, dtype=np.uint8) * 128
        vis_dvs[spikes_on == 1] = 255  
        vis_dvs[spikes_off == 1] = 0   
        vis_dvs_bgr = cv2.cvtColor(vis_dvs, cv2.COLOR_GRAY2BGR)

        preview_size = (256, 256)
        frame_preview = cv2.resize(cv2.resize(frame, target_size), preview_size, interpolation=cv2.INTER_NEAREST)
        vis_dvs_preview = cv2.resize(vis_dvs_bgr, preview_size, interpolation=cv2.INTER_NEAREST)
        side_by_side = np.hstack((frame_preview, vis_dvs_preview))
        visualization_frames.append(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB))
        
        spike_frame = np.stack([spikes_on, spikes_off], axis=0) 
        
        for _ in range(3):
            snn_tensor_list.append(spike_frame)
        
        prev_gray = curr_gray
        
    cap.release()
    
    if not snn_tensor_list:
        return None, None
        
    snn_tensor = torch.tensor(np.array(snn_tensor_list), dtype=torch.float32)
    
    return snn_tensor, visualization_frames

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        content = await file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name
        
    try:
        tensor_data, viz_frames = process_mp4_to_snn(temp_video_path)
        
        if tensor_data is None:
            os.remove(temp_video_path)
            return {"status": "error", "message": "Nie udało się odczytać pliku wideo."}
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as temp_gif:
            imageio.mimsave(temp_gif.name, viz_frames, fps=15)
            with open(temp_gif.name, "rb") as gif_file:
                gif_base64 = base64.b64encode(gif_file.read()).decode('utf-8')
        
        os.remove(temp_video_path)
        os.remove(temp_gif.name)
        
        tensor_data = tensor_data.to(device)
        if len(tensor_data.shape) == 4:
            tensor_data = tensor_data.unsqueeze(1) 
            
        with torch.no_grad():
            spk_out, _ = model(tensor_data)
            
            spike_count = spk_out.sum(dim=0)
            spike_count_population = spike_count.view(tensor_data.size(1), 101, 10).sum(dim=2)
            predicted_idx = spike_count_population.argmax(dim=1).item()
            predicted_class_name = NCALTECH_CLASSES[predicted_idx] if predicted_idx < len(NCALTECH_CLASSES) else f"Class_{predicted_idx}"

            probs = torch.softmax(spike_count_population[0], dim=0)
            top_probs, top_idx = torch.topk(probs, k=3)
            top3 = [
                {
                    "class": NCALTECH_CLASSES[i] if i < len(NCALTECH_CLASSES) else f"Class_{i}",
                    "confidence": round(p.item(), 4),
                }
                for p, i in zip(top_probs, top_idx.tolist())
            ]

        return {
            "status": "success",
            "filename": file.filename,
            "predicted_class": predicted_class_name,
            "top3": top3,
            "animation_base64": f"data:image/gif;base64,{gif_base64}"
        }
        
    except Exception as e:
        if os.path.exists(temp_video_path): 
            os.remove(temp_video_path)
        return {"status": "error", "message": str(e)}