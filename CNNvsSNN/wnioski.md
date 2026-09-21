## Problem z pamięcią na Colabbie
## Gorsza dokładność dla pipeline'u SNN + event camera
## Train Lossy inne, gdyż inne algorytmy (MSE vs CrossEntropy)
## Gorszy Czas Treningu dla SNN + event camera

W skrócie, badania wykazały:
* Trenowanie SNN jest o wiele razy dłuższe niz trenowanie CNN.
* Dokładność architektury SNN jest gorsza przy tych samych hiperparametrach niż CNN
* Dla tej samej karty graficznej (T4 na colabie), SNN potrebuje więcej pamięci w porównaniu do CNNa
