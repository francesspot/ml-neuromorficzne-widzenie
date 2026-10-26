🇵🇱 [Polski](README.md) | 🇬🇧 **English**

# 🧠 Neuromorphic Vision

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

> 🎓 Team final project of the AI/ML track of the **Summer Challenge 2026** (*Wakacyjne Wyzwanie*) run by the **KN Solvro** student research group (Wrocław University of Science and Technology)

A full-stack app that shows how **Spiking Neural Networks (SNNs)** work — brain-inspired networks that pass "spikes" instead of plain numbers. The app takes a regular video, converts it into data from an emulated **event camera** and classifies its content (101 categories from the N-Caltech101 dataset).

## 🙋 My contribution

- **Deep Pop-SCNN model** (snnTorch) — 3 convolutional + 2 fully connected layers, **population coding** (10 neurons per class) and LIF neurons that learn their own parameters
- **Full-stack integration** — FastAPI backend, Next.js frontend, bringing all 4 of the team's models into one app, Docker containerisation
- **Online fine-tuning** — when the model gets it wrong, the user points to the right class and the model retrains its last layers in a few short epochs without forgetting what it already knows

## ✨ App features

- **DVS camera emulator** — turns a regular `.mp4` into ON/OFF spikes, with a GIF preview
- **4 SNN architectures** to choose from (snnTorch and SpikingJelly), including SEW-ResNet18 with rate and temporal coding
- A `/retrain` endpoint for fine-tuning the models

## 👥 Team

**Franciszek Pora** · Liudmyla Tkachenko · Veronika Lachynova · Jan Kranz

## 🚀 Run it

```bash
git clone https://github.com/francesspot/ml-neuromorficzne-widzenie.git
cd ml-neuromorficzne-widzenie
docker compose -f Docker-compose.yml up --build
```

Frontend: <http://localhost:3000> · API: <http://localhost:8000/docs>
