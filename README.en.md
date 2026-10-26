[Polski](README.md) | **English**

# Neuromorphic Vision

A full-stack application demonstrating **Spiking Neural Networks (SNNs)** classifying data from event cameras. The project is based on the N-Caltech101 dataset and brings several SNN architectures together in one graphical interface, allowing video analysis and real-time fine-tuning of the models.

## ✨ Main features

- **On-the-fly DVS camera emulator:** the system takes standard video recordings (`.mp4`) and algorithmically converts them into event streams (ON/OFF spikes) in the format SNNs expect. The interface generates a side-by-side preview as a looping GIF.
- **Support for multiple SNN architectures:** the app loads 4 independent models built with the `snnTorch` and `SpikingJelly` libraries.
- **Online Continual Learning (micro-training):** a built-in mechanism that lets the user correct a wrong prediction. The system takes the file the user points to, trims the time sequence, initialises an optimiser for the last classifier layers (transfer learning) and fine-tunes the model in a few short epochs, reducing the loss without _catastrophic forgetting_.

## 🧠 Implemented models

1. **Deep Pop-SCNN (Franciszek)**
   - **Framework:** `snnTorch`
   - **Characteristics:** an architecture with 3 convolutional and 2 fully connected layers. It uses **population coding** (10 neurons per class) and adaptive _Leaky Integrate-and-Fire_ (LIF) neurons that learn their own membrane parameters ($\beta$) and firing thresholds during training. It offers the greatest flexibility and robustness to noise.
2. **CSNN (Liudmyla)**
   - **Framework:** `snnTorch`
   - **Characteristics:** a lightweight convolutional network (2 Conv layers + 1 Linear) using rate coding. Optimised for lower resource usage.
3. **SEW-ResNet18 — Rate Coding (Veronika)**
   - **Framework:** `SpikingJelly`
   - **Characteristics:** a deep, 18-layer network using Spiking Element-Wise blocks (ADD connections), which solves the vanishing-gradient problem. It achieves the highest baseline accuracy.
4. **T-SEW-ResNet18 — Temporal Coding (Veronika)**
   - **Framework:** `SpikingJelly`
   - **Characteristics:** a variant of the SEW-ResNet architecture based on temporal (spike-timing) coding, with a loss function that looks at when a spike occurs rather than only at the total spike count.

## 📂 Repository structure

- `snn-ui/` — frontend source code written in Next.js / React and styled with Tailwind CSS.
- `main.py` — the main FastAPI backend server, handling video processing, prediction (`/predict`) and fine-tuning (`/retrain`).
- `snn_model.py` / `sew_resnet.py` — model class definitions and the weights manager.
- `*.pth` — saved trained weights for each architecture (e.g. `franciszek_scnn_best.pth`, `sew_resnet18_best.pth`).
- `Dockerfile` / `Docker-compose.yml` — configuration files for full containerisation of the app.
- `requirements.txt` — Python dependencies (PyTorch, snnTorch, SpikingJelly, OpenCV, FastAPI).

## 🚀 Running the app (Docker)

The recommended way to run the project is with Docker, which sets up the whole environment (including PyTorch's complex dependencies) automatically.

1.  Clone the repository:

        git clone https://github.com/francesspot/ml-neuromorficzne-widzenie.git
        cd ml-neuromorficzne-widzenie

2.  Build and start the containers (the file is named `Docker-compose.yml`, so it has to be passed explicitly):

        docker compose -f Docker-compose.yml up --build

3.  Open your browser and go to:
    - **Frontend (user interface):** `http://localhost:3000`
    - **Backend (API docs):** `http://localhost:8000/docs`

## 🛠️ Technologies

- **Machine Learning:** PyTorch, snnTorch, SpikingJelly
- **Backend:** FastAPI, Python, OpenCV (video processing), Uvicorn
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **DevOps:** Docker, Docker Compose

## 👥 Project team

The project was carried out as part of a university course by:

- **Franciszek Pora** (Pop-Coding SCNN architecture, full-stack integration, continual learning)
- **Liudmyla Tkachenko** (baseline CSNN architecture)
- **Veronika Lachynova** (SEW-ResNet18 architecture in Rate and Temporal variants)
- **Jan Kranz** (CNN vs SNN)
