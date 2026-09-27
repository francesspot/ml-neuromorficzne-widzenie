## Różnica Datasetów
Caltech101: Klasa `Faces`
Caltech101: [Usunięta klasa `BACKGROUND_Google`](https://github.com/pytorch/vision/blob/main/torchvision/datasets/caltech.py)
N-Caltech101: [Brak klasy `Faces`](https://www.garrickorchard.com/datasets/n-caltech101)
N-Caltech101: [Obecna klasa tła](https://www.garrickorchard.com/datasets/n-caltech101)

Rozwiązanie:
    CNN: Usuń `Faces` + remapowanie etykiet by zalepić lukę
    SNN: Usuń `BACKGROUND_Google`
(!)
    torchvision.datasets.Caltech101 -> gotowy label encoding -> zabawa z łataniem dziury
    tonic.datasets.NCALTECH101 -> label encoding robiony manualnie -> usunięcie `BACKGROUND_Google` przed tym jak stanie się ono problemem.

## Augmentancje Danych
Caltech101:
    Resize + Normalize
    (t) Horizontal Flip
    (t) Color Jitter
    ToTensor

N-Caltech101:
    Downsampling (Resize dla SNN)
    ToFrame -> spłaszczenie segmentów czasowych do jednego "zdjęcia"
    from_numpy.float() -> tonic(Pytorch) oraz BatchNormalization chce floaty zamiast integerów

## Wspólny Model
Wspólna architektura modelowa: [Resnet-18 (SmallResNet)](https://medium.com/@YasinShafiei/residual-networks-resnets-with-implementation-from-scratch-713b7c11f612)
Zmieniłem pierwszą warstwę konwolucji z `stride=1, padding=0` na `stride=2, padding=3` gdyż są klasyczne wartości dla modelu ResNet, tak jak [napisano w tym artykule](https://arxiv.org/pdf/1512.03385)

## Różnice archiketuralne CNN vs SNN

dodatkowe 2 warstwy lif1, brak relu
CNN bierze RGB (in_channel=3), podczas gdy SNN bierze 2 ( On counter, Off counter)
SNN colapsuje zmiany w pikselach co 10 ms według `n_time_bins`

## Trenowanie
Do obliczenia funkcji strat użyłem `CrossEntropyLoss` do CNN, a do SNN `mse_count_loss`, więc nie mogę bezpośrednio porównać różnicy wyników funkcji strat podczas trenowania obu modeli.
w roli optymalizatora użyłem algorytm `Adam` z wspólnymi wartościami między modelami

Metryki jakie użyłem:
* `accuracy` - dokładność predykcji
* `peak GPU` - użycie pamięci GPU
* Czas treningu
## Optymalizacje pamięci
torch.amp.autocast + `GradScaler`
W miejscach gdzie to możliwe, używam 16-bitowych wartości numerycznych zamiast 32-bitowych (np gradienty). `GradScaler` ma za zadanie wyskalować wartości małych gradientów i ochronić je przed wyzerowaniem

Skonfigurowałem także segmentację pamięci na elastyczną za pomocą `PYTORCH_CUDA_ALLOC_CONF`, przez co przy usunięciu bloków VRAM, są one zasysane do istniejących bloków, zamiast fragmentować pamięć

Dodatkowo, używam trochę bardziej optymalnego handlingu batchy, gdyż zamiast trzymania 16 batchy w pamięci na raz, trzymam pierwsze 8, a potem drugie 8

SNN w pomiarze pamięci używa także `torch.utils.checkpoint.checkpoint` w celach optymalizacji użycia pamięci ,by nie zapisywać w pamięci wartości aktywacji, co wymusza dodatkowy forwardpass.

wiekszość użyta w pętli treningu

## Wyniki
### Proces Porównania Na Colabie
100%(Czas pracy nad porównaniem) = 40%(Tworzenie Modeli) + 35%(Optymalizowanie Pamięci) + 25%(Czekanie aż CUDA zcraszuje Colaba i wykończy limit tygodniowy na GPU)
### Dokładność
Końcowa dokładność po równej ilości epoch wykazuje zdecydowaną przewagę klasycznego pipeline'u: CNN + normalna kamera(**47.6%**), w porównaniu do SNN + kamera neuromorficza(**26%**)
### Czas Treningu
Różnica między czasem treningu CNN(2.3 min) a SNN(19.0 min) podkreśla znacznie większe (~8.3 krotne) zapotrzebowanie czasowe w fazie tworzenia modelu.
### Zużycie Pamięci
CNN przy batchu liczącym 1920 zdjęć jest nadal mniej wymagający pamięciowo (11.69 GiB) niż SNN z batchem równym 1024 i checkpointami (14.14 GiB)
