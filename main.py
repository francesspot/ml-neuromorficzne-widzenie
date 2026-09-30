import os
import io
import base64
import tempfile
import threading
from datetime import datetime

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import snntorch.functional as SF
from spikingjelly.clock_driven import functional

from snn_model import AppModelManager

app = FastAPI(title="Neuromorficzne widzenie")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manager = AppModelManager(device)
model_lock = threading.Lock()

CORRECTIONS_DIR = "corrections_data"
os.makedirs(CORRECTIONS_DIR, exist_ok=True)

NCALTECH_CLASSES = [
    'BACKGROUND_Google', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant',
    'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera',
    'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
    'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin',
    'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo',
    'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone',
    'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop',
    'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus',
    'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino',
    'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball',
    'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite',
    'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang'
]


def _resize_and_center_crop(img, target_size):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h))
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return resized[top:top + target_h, left:left + target_w]


def process_mp4_to_snn(video_path, target_size=(80, 80), spike_percentile=98.5, min_diff_threshold=15):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = _resize_and_center_crop(prev_gray, target_size)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    snn_tensor_list = []
    visualization_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = _resize_and_center_crop(curr_gray, target_size)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        curr_gray_int = curr_gray.astype(np.int16)
        prev_gray_int = prev_gray.astype(np.int16)
        diff = curr_gray_int - prev_gray_int

        abs_diff = np.abs(diff)
        adaptive_threshold = max(min_diff_threshold, float(np.percentile(abs_diff, spike_percentile)))

        spikes_on = (diff > adaptive_threshold).astype(np.float32)
        spikes_off = (diff < -adaptive_threshold).astype(np.float32)

        vis_dvs = np.ones(target_size, dtype=np.uint8) * 128
        vis_dvs[spikes_on == 1] = 255
        vis_dvs[spikes_off == 1] = 0
        vis_dvs_bgr = cv2.cvtColor(vis_dvs, cv2.COLOR_GRAY2BGR)

        preview_size = (256, 256)
        frame_preview = cv2.resize(_resize_and_center_crop(frame, target_size), preview_size, interpolation=cv2.INTER_NEAREST)
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

    return torch.tensor(np.array(snn_tensor_list), dtype=torch.float32), visualization_frames


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("franciszek_scnn")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        content = await file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name

    try:
        tensor_data, viz_frames = process_mp4_to_snn(temp_video_path)

        if tensor_data is None:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return {"status": "error", "message": "Nie udało się odczytać pliku wideo."}

        buf = io.BytesIO()
        imageio.mimsave(buf, viz_frames, format='GIF', fps=15, loop=0)
        gif_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        tensor_data = tensor_data.to(device)
        if len(tensor_data.shape) == 4:
            tensor_data = tensor_data.unsqueeze(1)

        tensor_data_flipped = torch.flip(tensor_data, dims=[-1])

        with model_lock:
            logits = manager.get_logits(model_name, tensor_data, tensor_data_flipped)
            predicted_idx = logits.argmax(dim=0).item()
            predicted_class_name = NCALTECH_CLASSES[predicted_idx] if predicted_idx < len(NCALTECH_CLASSES) else f"Class_{predicted_idx}"

            probs = torch.softmax(logits, dim=0)
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


@app.post("/retrain")
async def retrain_model(
    file: UploadFile = File(...),
    correct_class: str = Form(...),
    model_name: str = Form("franciszek_scnn")
):
    try:
        target_idx = NCALTECH_CLASSES.index(correct_class)
    except ValueError:
        return {"status": "error", "message": f"Nieznana klasa: {correct_class}"}

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        content = await file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_class = correct_class.replace("/", "_").replace(" ", "_")
    correction_path = os.path.join(CORRECTIONS_DIR, f"{ts}__{safe_class}.mp4")
    with open(correction_path, "wb") as f:
        f.write(content)

    try:
        snn_tensor, _ = process_mp4_to_snn(temp_video_path)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        if snn_tensor is None:
            return {"status": "error", "message": "Błąd przetwarzania pliku wideo."}

        snn_tensor = snn_tensor.to(device)
        if len(snn_tensor.shape) == 4:
            snn_tensor = snn_tensor.unsqueeze(1)

        target_tensor = torch.tensor([target_idx], dtype=torch.long).to(device)

        model = manager.get_model(model_name)

        if model_name == "franciszek_scnn":
            head_params = (
                list(model.fc1.parameters())
                + list(model.lif4.parameters())
                + list(model.fc2.parameters())
                + list(model.lif5.parameters())
            )
        elif model_name == "liudmyla_scnn":
            head_params = list(model.linear.parameters()) + list(model.leaky2.parameters())
        elif model_name == "weronika_sew18":
            head_params = list(model.fc.parameters())
        elif model_name == "weronika_t_sew18":
            head_params = list(model.fc.parameters())
            if hasattr(model, "t_out") and model.t_out is not None:
                head_params += list(model.t_out.parameters())
        else:
            return {"status": "error", "message": f"Nieobsługiwany model: {model_name}"}

        optimizer = optim.Adam(head_params, lr=0.0005)

        def compute_loss(input_tensor, targets):
            if model_name == "franciszek_scnn":
                spk, _ = model(input_tensor)
                loss_fn_pop = SF.ce_count_loss(population_code=True, num_classes=101)
                return loss_fn_pop(spk, targets)

            elif model_name == "liudmyla_scnn":
                spk, _ = model(input_tensor)
                loss_fn_rate = SF.ce_count_loss(population_code=False, num_classes=101)
                return loss_fn_rate(spk, targets)

            elif model_name == "weronika_sew18":
                out = model(input_tensor)
                functional.reset_net(model)
                if out.dim() == 3:
                    out = out.mean(dim=0)
                ce_loss = nn.CrossEntropyLoss()
                return ce_loss(out, targets)

            elif model_name == "weronika_t_sew18":
                out = model(input_tensor)
                functional.reset_net(model)
                temp_loss = SF.ce_temporal_loss()
                return temp_loss(out, targets)

        final_loss = None
        with model_lock:
            model.train()
            try:
                for _ in range(6):
                    optimizer.zero_grad()
                    loss = compute_loss(snn_tensor, target_tensor)
                    loss.backward()
                    optimizer.step()
                    final_loss = loss.item()
                    if final_loss < 0.05:
                        break
            finally:
                model.eval()
                if "weronika" in model_name:
                    functional.reset_net(model)

        return {
            "status": "success",
            "message": f"Warstwy klasyfikatora pomyślnie zaktualizowane ({model_name}). Loss: {final_loss:.4f}. Nagranie zapisane.",
            "new_loss": float(final_loss)
        }

    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return {"status": "error", "message": str(e)}