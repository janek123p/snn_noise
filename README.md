# Source-Code zur Bachelorarbeit "Robustheit lernender, biologischer neuronaler Netze gegenüber Rauscheinflüssen"

Der Code basiert auf dem Netzwerk von [Diehl et al.](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full) Der ursprüngliche Code kann in [diesem Github-Respository](https://github.com/peter-u-diehl/stdp-mnist) gefunden werden. Ausgehend von diesem Code wurde die Implementierung des Netzwerkes auf BRIAN2 und Python3 migriert. Der dazugehörige Code kann [hier](https://github.com/sdpenguin/Brian2STDPMNIST) gefunden werden. 

Im Rahmen der Bachelorarbeit "Robustheit lernender, biologischer neuronaler Netze gegenüber Rauscheinflüssen" wurde der Code leicht angepasst und es wurde mit dem Clopath-Synapsen-Modell ein weiteres Synapsenmodell hinzugefügt. Darüber hinaus wurden eine Reihe von Rauschszenarien implementiert, die über die entsprechenden Kommandozeilenargumente eingeschaltet werden können. 

## Dokumentation der Skripte

### global_settings.py

In der Datei global_settings.py sind globale, nutzerspezifische Einstellungen zu finden. Aktuell enthält diese Datei ausschließlich den Pfad, an dem die Simulationen und die dazugehörigen Dateien erzeugt werden sollen.

### init_directory_structure.py

Das Skript init_directory_structure.py wird aufgerufen, um die Verzeichnisstruktur für eine Simulation anzulegen und die zufälligen Startgewichte der SImulation zu erzeugen. Das Skript kann wie folgt aufgerufen werden:

```
python init_directory_structure.py [-h] -label LABEL [-N N] [-input {mnist,cifar10}]
```
Das Skript stellt folgende individualisierbare Parameter zur Verfügung:

| Argument  | Bedeutung | optional (Standard) |
| :---      | :---    |    :---  |
| -h        | Hilfe aufrufen | ja  |
| -label    | Eindeutiger Bezeichner der Simulation | nein |
| -N        | Anzahl exzitatorischer Neuronen | ja (400)|
| -input    | Input-Datensatz: Entweder 'mnist' oder 'cifar' | ja ('mnist')|

Wird das Skript aufgerufen, so wird innerhalb des Simulationsordners, der in global_settings.py hinterlegt ist, ein Unterordner angelegt in dem alle Daten der Simulation abgespeichert werden. Dieser Ordner enthält die Unterordner 'activity', 'meta', 'plots', 'random' und 'weights'. Im Unterordner 'activity' werden nach der Simulation die Aktivitäten der exzitatorschen Neuronen sowie die zugehörigen Input-Label abgelet, um später daraus Ergebnisse berechnen zu können. Im Ordner 'meta' werden eventuelle Meta-Daten und statistische Daten zur Simulation abgelegt. Im Ordner 'plots' werden Plots zur Simulation abgelegt. Im Ordner 'random' sind nach Initialisierung durch das Skript die Gewichte für alle Synapsen zu finden. Hierzu zählen auch die zufällig initialisierten Gewichte der STDP-Synapsen. Im Ordner 'weights' werden während und nach der Simulation die trainierten Gewichte abgelegt.

### simulate.py
Das Skript simulate.py kann zur Simulation des Netzwerkes verwendet werden. Es wird über 
```
python simulate.py [-h] -mode {test,train,TEST,TRAIN,training,TRAINING} 
                   -label label
                   [-data DATAPATH] [-epochs EPOCHS]
                   [-train_size TRAIN_SIZE] [-test_size TEST_SIZE]
                   [-plasticity]
                   [-synapse_model {triplet,clopath,TRIPLET,CLOPATH,clopath-similar,CLOPATH-SIMILAR}]
                   [-debug] [-test_label TEST_LABEL] [-N N]
                   [-input {mnist,cifar10}]
                   [-rand_threshold_max RAND_THRESH_MAX]
                   [-rand_threshold_min RAND_THRESH_MIN]
                   [-noise_membrane_voltage_max NOISE_MEMBRANE_VOLTAGE_MAX]
                   [-noise_membrane_voltage_min NOISE_MEMBRANE_VOLTAGE_MIN]
                   [-voltage_noise_sigma SIGMA_V]
                   [-voltage_noise_sigma_inh SIGMA_V_INH]
                   [-membrane_voltage_quant MEMBRANE_VOLTAGE_QUANT]
                   [-membrane_voltage_quant_inh MEMBRANE_VOLTAGE_QUANT_INH]
                   [-weight_quant WEIGHT_QUANT]
                   [-stoch_weight_quant STOCH_WEIGHT_QUANT]
                   [-salt_and_pepper_alpha SALT_PEPPER_ALPHA]
                   [-rectangle_noise_min RECTANGLE_NOISE_MIN]
                   [-rectangle_noise_max RECTANGLE_NOISE_MAX]
                   [-p_dont_send_spike P_DONT_SEND_SPIKE]
                   [-p_dont_send_spike_inh P_DONT_SEND_SPIKE_INH]
                   [-sigma_heterogenity SIGMA_HET]
```

aufgerufen. Die Bedutung der Parameter wird in der folgenden Tabelle erläutert.

| Argument  | Bedeutung | optional (Standard) |
| :---      | :---    |    :---  |
| -h        | Hilfe aufrufen | ja  |
| -mode        | Modus der Simulation: 'train' oder 'test' | nein |
| -label    | Eindeutiger Bezeichner der Simulation | nein |
| -data    | Datenpfad der Input-Daten | ja ('./mnist/')|
| -epochs    | Anzahl der Trainings-Epochen | ja (1)|
| -train_size | Anzahl der Iterationen pro Epoche während des Trainings | ja (60000)|
| -test_size  | Anzahl der Iterationen pro Epoche während des Testens | ja (10000)|
| -plasticity | Wenn die Flag gesetzt ist wird auch während des Testens STDP und der adaptive Threshold verwendet, während des Trainings wird dies unabhängig von der Flag getan | ja |
| -synapse_model  | Zu verwendendes Synapsenmodell: Entweder 'triplet', 'clopath' oder 'clopath-similar'. 'clopath-similar' ist ein Synapsenmodell, welches nur durch einen Fehler entstanden ist und nicht verwendet werden sollte. | ja ('triplet')|
| -debug | Wenn die Flag gesetzt ist, werden zusätzliche Debug-Informationen gespeichert. Dieser Parameter kann bei langen Simulationen zu Speicherproblemen führen | ja |
| -test_label | Eindeutige Bezeichnung des Testfalls, der auf den Gewichten der Simulation ausgeführt werden soll | ja |
| -N        | Anzahl exzitatorischer Neuronen | ja (400)|
| -input        | Input-Datensatz: Entweder 'mnist' oder 'cifar10' | ja ('mnist')|
| -rand_thresh_max | Maximaler zufälliger Threshold, der auf den Threshold aufaddiert wird. Zufälliger Threshold wird gleichverteilt. | ja (0) |
| -rand_thresh_min | Minimaler zufälliger Threshold, der auf den Threshold aufaddiert wird. Zufälliger Threshold wird gleichverteilt. | ja (0) |
| -noise_membrane_voltage_max | Maximale gleichverteilte zufällige Veränderung der Membranspannung pro Zeitschritt. | ja (0) |
| -noise_membrane_voltage_min | Minimale gleichverteilte zufällige Veränderung der Membranspannung pro Zeitschritt. | ja (0) |
| -voltage_noise_sigma | Standardabweichung der Normalverteilung die als Rauschen auf die Membranspannung der exzitatorischen Neuronene angewandt wird | ja (0) |
| -voltage_noise_sigma_inh | Standardabweichung der Normalverteilung die als Rauschen auf die Membranspannung der inhibitorischen Neuronene angewandt wird | ja (0) |
| -membrane_voltage_quant | Anzahl an Bits mit der die Nachkommastellen der Membranspannung der exzitatorischen Neuronen quantisiert werden sollen. | ja |
| -membrane_voltage_quant_inh | Anzahl an Bits mit der die Nachkommastellen der Membranspannung der inhibitorischen Neuronen quantisiert werden sollen. | ja |
| -weight_quant | Anzahl an Bits mit der die Nachkommastellen der synaptischen Gewichte quantisiert werden soll | ja |
| -stoch_weight_quant | Anzahl an Bits mit der die Nachkommastellen der synaptischen Gewichte stochastisch quantisiert werden soll | ja |
| -salt_and_pepper_alpha | Intensität des Salt and Pepper Rauschens auf dem Input (zwsichen 0 und 1) | ja (0) |
| -rectangle_noise_min | Minimale Rechteckbreite und -länge, die aus dem Input entfernt werden soll | ja (0) |
| -rectangle_noise_max | Maximale Rechteckbreite und -länge, die aus dem Input entfernt werden soll | ja (0) |
| -p_dont_send_spike | Wahrscheinlichkeit mit der ein Aktionspotential eines exzitatorischen Neurons nicht zu einer Veränderung der postsynaptischen Leitfähigkeit führt. | ja (0) |
| -p_dont_send_spike_inh | Wahrscheinlichkeit mit der ein Aktionspotential eines inhibitorischen Neurons nicht zu einer Veränderung der postsynaptischen Leitfähigkeit führt. | ja (0) |
| -sigma_heterogenity | Standardabweichung der Normalverteilung als Anteil vom Mittelwert, mit der die neuronale Heterogenität durchgeführt werden soll | ja (0) |

### evaluate.py

Nachdem das Training und das Testen abgesclossen ist, kann das Ergebnis ausgewertet werden. Dazu muss das Skript evaluate.py aufgerufen werden. Dies geschieht über 

```
python evaluate.py [-h] -label LABEL [-num_assigns ASSIGNMENT_NUMBER]
                   [-datapath DATA_PATH] [-test_label TEST_LABEL] [-svm]
                   [-N N]
```

Die dazugehörigen Argumente sind in der folgenden Tabelle erläutert.

| Argument  | Bedeutung | optional (Standard) |
| :---      | :---    |    :---  |
| -h        | Hilfe aufrufen | ja  |
| -label    | Eindeutiger Bezeichner der Simulation | nein |
| -num_assigns | Anzahl an Trainings-Daten, die zur Bestimmung der Zuordnungen der exzitatorischen Neuronen verwendet werden sollen | ja (10000) | 
| -data_path | Pfad zum Laden der Input-Daten | ja ('./mnist/')|
| -test_label | Eindeutige Bezeichnung des Testfalls | nein |
| -svm       | Wenn die Flag gesetzt ist, wird eine SVM trainiert um die Neuronen-Aktivitäten zu interpretieren statt die Zurodnung anhand der häufigsten Ziffern pro Neuron zu berechnen (kann je nach Größe einige Sekunden bis Minuten dauern) | ja |
| -N        | Anzahl exzitatorischer Neuronen | ja (400)|