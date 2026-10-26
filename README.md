🇵🇱 **Polski** | 🇬🇧 [English](README.en.md)

# 🧠 Neuromorficzne Widzenie

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

> 🎓 Projekt końcowy zespołu na ścieżce AI/ML **Wakacyjnego Wyzwania 2026** koła naukowego **KN Solvro** (Politechnika Wrocławska)

Aplikacja full-stack, która pokazuje, jak działają **impulsowe sieci neuronowe (SNN)** — sieci inspirowane mózgiem, które zamiast liczb przesyłają „impulsy”. ⚡ Wrzucasz zwykłe nagranie wideo, aplikacja zamienia je na dane z wirtualnej **kamery zdarzeniowej** i rozpoznaje, co jest na filmie (101 kategorii ze zbioru N-Caltech101).

## 🙋 Mój wkład

- 🏗️ **Model Deep Pop-SCNN** (snnTorch) — 3 warstwy splotowe + 2 w pełni połączone, **kodowanie populacyjne** (10 neuronów na klasę) i neurony LIF, które same uczą się swoich parametrów
- 🔗 **Integracja full-stack** — backend w FastAPI, frontend w Next.js, połączenie wszystkich 4 modeli zespołu w jednej aplikacji, konteneryzacja w Dockerze
- 🔁 **Douczanie na żywo** — gdy model się pomyli, użytkownik wskazuje poprawną klasę, a model douczy ostatnie warstwy w kilku krótkich epokach, nie zapominając tego, co już umie

## ✨ Funkcje aplikacji

- 🎥 **Emulator kamery DVS** — zamiana zwykłego `.mp4` na impulsy ON/OFF z podglądem jako GIF
- 🧩 **4 architektury SNN** do wyboru (snnTorch i SpikingJelly), m.in. SEW-ResNet18 w wersjach z kodowaniem częstotliwościowym i czasowym
- 🔁 Endpoint `/retrain` do douczania modeli

## 👥 Zespół

**Franciszek Pora** · Liudmyla Tkachenko · Veronika Lachynova · Jan Kranz

## 🚀 Uruchomienie

```bash
git clone https://github.com/francesspot/ml-neuromorficzne-widzenie.git
cd ml-neuromorficzne-widzenie
docker compose -f Docker-compose.yml up --build
```

🎨 Frontend: <http://localhost:3000> · ⚙️ API: <http://localhost:8000/docs>
