## Teoretyczne Zyski Energetyczne
SNN'y mają przede wszystkim dorównać ANN'om ze względu na wydajność, z równoczesną redukcją zużycia energetycznego na dedykowanym hardwarze.
Dobrze zooptymalizowane SNN w warunkach laboratoryjnych wykazują [od 100 do 1000 krotnego zmniejszenia poboru energetycznego](https://arxiv.org/pdf/2109.12894)
## Różnica Datasetów
Z tytułu, iż chcieliśmy porównać pipeline CNN (normalna kamera + CNN) z SNN (neuromorficzna kamera + SNN), zdecydowaliśmy się użyć datasetu Caltech101 dla CNN i N-Caltech101 dla SNN.
Okazało się to bardziej skomplikowane niż przewidywaliśmy, gdyż chociaż N-Caltech101 pełni funkcję neuromorficznego odpowiednika datasetu Caltech101, to różnią się one z perspektywy zawartości.
* Klasyczny Caltech101 posiada `Faces` oraz `FacesEasy`, podczas gdy N-Caltech101 został uproszczony poprzez [zawężenie zakresu](https://www.garrickorchard.com/datasets/n-caltech101) do tylko `FacesEasy`
* Caltech101 importowany przez Pytorch (`torchvision`) [domyślnie usuwa klasę "BACKGROUND_Google"](https://github.com/pytorch/vision/blob/main/torchvision/datasets/caltech.py), podczas gdy N-Caltech101 z `tonic` [ją zachowuje](https://www.garrickorchard.com/datasets/n-caltech101)

By rzetelnie porównać pipeline, podjeliśmy następujące czynności:
* Usuneliśmy `Faces` z Caltech101 (+ remapowanie etykiet by zalepić lukę)
* Usuneliśmy `BACKGROUND_Google` z N-Caltech101
## Augmentancje Danych
W celach uczciwego porównania sieci, podjeliśmy decyzję nie wprowadzać augumentacji danych, zważając na możliwe różnice funkcyjne tego samego typu transformacji dla danych różniących się fundamentalnym formatem reprezentacji.
## Wspólny Model
Do rzetelnego benchmarku pipeline'u musieliśmy zdecydować się na zbieżną architekturę macierzystą obu modeli.
Zdecydowaliśmy się na Resnet-18 (SmallResNet).
## Różnice Architekturalne CNN vs SNN
Model CNN został zaimplementowany według [tego źródła](https://medium.com/@YasinShafiei/residual-networks-resnets-with-implementation-from-scratch-713b7c11f612) + poprawka `stride=2, padding=3` [według tego artykułu](https://arxiv.org/pdf/1512.03385)

Potem CNN przekonwertowaliśmy na SNN:
* Dodaliśmy 2 warsty LiF i usuneliśmy warsty funkcji aktywacyjnych ReLu
* Skonfigurowaliśmy wejściową głębokość kanałową z 3 na 2 (RGB -> Polarization counters)
* Z tytułu, iż sieci SNN na wejście biorą klatki, trzeba ciąg zmian pikselowych złożyć w klatkę zależnie od segmentów czasowych o długości `n_time_bins` mili sekund
## Trenowanie
Modele różnią się używaną funkcją straty, nie można porównywać bezpośrednio ich wartości -> ważne że straty spadają.
W roli optymalizatora użyliśmy algorytm `Adam` z wspólnymi wartościami między modelami
Główne metryki mierzone z sprawą treningu:
* `accuracy` - dokładność predykcji
* `peak GPU` - użycie pamięci GPU
* Czas treningu
## Optymalizacje pamięci
Pamięć Cuda została skonfigurowana za pomocą `PYTORCH_CUDA_ALLOC_CONF`, co ratuje przed fragmentacją.
Za pomocą `torch.amp.autocast` + `GradScaler` gdzie możliwe używamy 16-bitowych zamiast 32-bitowych wartości do zapisu wartości gradientów podczas treningu

Dodatkowo, używamy mniej wymagającego handlingu batchy, gdyż zamiast trzymania 16 batchy w pamięci na raz, trzymam pierwsze 8, a potem drugie 8

SNN w pomiarze pamięci używa także `torch.utils.checkpoint.checkpoint` w celach optymalizacji użycia pamięci ,by nie zapisywać w pamięci wartości aktywacji, co wymusza dodatkowy forwardpass.

## Wyniki
### Proces Porównania Na Colabie
100%(Czas pracy nad porównaniem) = 40%(Tworzenie Modeli) + 35%(Optymalizowanie Pamięci) + 25%(Czekanie aż CUDA zcraszuje Colaba i wykończy limit tygodniowy na GPU)
### Dokładność
Końcowa dokładność po równej ilości epoch wykazuje zdecydowaną przewagę klasycznego pipeline'u: CNN + normalna kamera(**~47**), w porównaniu do SNN + kamera neuromorficza(**26%**), dwukrotnie większa dokładność.
### Czas Treningu
Różnica między czasem treningu CNN(2.3 min) a SNN(19.0 min) podkreśla znacznie większe (~8.3 krotne) zapotrzebowanie czasowe w fazie tworzenia modelu.
### Zużycie Pamięci
CNN przy batchu liczącym 1920 zdjęć jest nadal mniej wymagający pamięciowo (11.69 GiB) niż SNN z batchem równym 1024 i checkpointami (14.14 GiB)
