# Neuromorficzne Widzenie (Neuromorphic Vision)

Aplikacja full-stack demonstrująca działanie impulsowych sieci neuronowych (SNN - Spiking Neural Networks) w zadaniu klasyfikacji danych z kamer zdarzeniowych (event cameras). Projekt opiera się na zbiorze danych N-Caltech101 i integruje kilka różnych architektur SNN w jednym interfejsie graficznym, umożliwiając analizę wideo oraz douczanie modeli w czasie rzeczywistym.

## ✨ Główne funkcjonalności

- **Emulator kamery DVS w locie:** System przyjmuje standardowe nagrania wideo (`.mp4`), a następnie algorytmicznie przelicza je na ciągi zdarzeń (spikes ON/OFF) w formacie obsługiwanym przez SNN. W interfejsie generowany jest podgląd porównawczy w formie zapętlonej animacji GIF.
- **Wsparcie dla wielu architektur SNN:** Aplikacja ładuje 4 niezależne modele, bazujące na bibliotekach `snnTorch` oraz `SpikingJelly`.
- **Online Continual Learning (Mikro-trening):** Wbudowany mechanizm pozwalający użytkownikowi na poprawienie błędnej predykcji. System pobiera wskazany przez użytkownika plik, obcina sekwencję czasową, inicjalizuje optymalizator dla ostatnich warstw klasyfikatora (transfer learning) i doucza model w kilku krótkich epokach, minimalizując straty bez zjawiska _catastrophic forgetting_.

## 🧠 Zaimplementowane modele

1. **Deep Pop-SCNN (Franciszek)**
   - **Framework:** `snnTorch`
   - **Charakterystyka:** Architektura oparta na 3 warstwach splotowych i 2 w pełni połączonych. Wykorzystuje **kodowanie populacyjne** (10 neuronów na klasę) oraz adaptacyjne neurony _Leaky Integrate-and-Fire_ (LIF), które w trakcie treningu uczą się własnych parametrów membrany ($\beta$) i progu pobudzenia. Zapewnia najwyższą elastyczność i odporność na szum.
2. **CSNN (Liudmyla)**
   - **Framework:** `snnTorch`
   - **Charakterystyka:** Lekka sieć splotowa (2 warstwy Conv + 1 Linear) używająca kodowania częstotliwościowego. Zoptymalizowana pod kątem mniejszego zużycia zasobów.
3. **SEW-ResNet18 - Rate Coding (Veronika)**
   - **Framework:** `SpikingJelly`
   - **Charakterystyka:** Głęboka, 18-warstwowa sieć wykorzystująca bloki Spiking Element-Wise (połączenia typu ADD), rozwiązująca problem zanikającego gradientu. Osiąga najwyższą bazową skuteczność (Accuracy).
4. **T-SEW-ResNet18 - Temporal Coding (Veronika)**
   - **Framework:** `SpikingJelly`
   - **Charakterystyka:** Wariant architektury SEW-ResNet bazujący na kodowaniu czasowym (Spike-timing) i funkcji straty analizującej moment wystąpienia impulsu, a nie tylko ich zliczoną sumę.

## 📂 Struktura repozytorium

- `snn-ui/` - Kod źródłowy frontendu napisanego w Next.js / React, stylizowany za pomocą Tailwind CSS.
- `main.py` - Główny plik serwera backendowego w FastAPI, obsługujący przetwarzanie wideo, predykcję oraz endpoint `/retrain`.
- `snn_model.py` / `sew_resnet.py` - Definicje klas modeli i menedżer wag.
- `*.pth` - Zapisane, wyuczone wagi modeli dla poszczególnych architektur (np. `franciszek_scnn_best.pth`, `sew_resnet18_best.pth`).
- `Dockerfile` / `docker-compose.yml` - Pliki konfiguracyjne do pełnej konteneryzacji aplikacji.
- `requirements.txt` - Zależności dla środowiska Python (PyTorch, snnTorch, SpikingJelly, OpenCV, FastAPI).

## 🚀 Uruchomienie aplikacji (Docker)

Zalecanym sposobem uruchomienia projektu jest użycie Dockera. Dzięki temu całe środowisko (wraz ze skomplikowanymi zależnościami PyTorcha) skonfiguruje się automatycznie.

1.  Sklonuj repozytorium:

        git clone [https://github.com/twoj-profil/ml-neuromorficzne-widzenie.git](https://github.com/twoj-profil/ml-neuromorficzne-widzenie.git)
        cd ml-neuromorficzne-widzenie

2.  Zbuduj i podnieś kontenery:

        docker-compose up --build

3.  Otwórz przeglądarkę i przejdź do adresów:
    - **Frontend (Interfejs Użytkownika):** `http://localhost:3000`
    - **Backend (API Docs):** `http://localhost:8000/docs`

## 🛠️ Technologie

- **Machine Learning:** PyTorch, snnTorch, SpikingJelly
- **Backend:** FastAPI, Python, OpenCV (przetwarzanie wideo), Uvicorn
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **DevOps:** Docker, Docker Compose

## 👥 Zespół Projektowy

Projekt zrealizowany w ramach przedmiotu w składzie:

- **Franciszek Pora** (Architektura SCNN Pop-Coding, Full-stack integration, Continual Learning)
- **Liudmyla Tkachenko** (Architektura podstawowa CSNN)
- **Veronika Lachynova** (Architektura SEW-ResNet18 w wariantach Rate i Temporal)
- **Jan Kranz** (CNN vs SNN)
