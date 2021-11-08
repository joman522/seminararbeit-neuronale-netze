import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))
sigmoid_abl = lambda x: x*(1-x)
zufall = np.random.default_rng(12345)
trainingsdaten = []
erwartet = []
netz = []

class Neuron:

    gewichte = []
    ausgabe = 0

    def __init__(self, gewichte):
        self.gewichte = gewichte

    def ausgabe_berechnen(self, eingaben):
        self.ausgabe = 0

        self.ausgabe = sigmoid(sum([self.gewichte[i]*eingaben[i] for i in range(len(eingaben))]))
        
        return self.ausgabe

    def gewichte_anpassen(self, eingaben, fehlersignal):
        for i in range(len(self.gewichte)):
            self.gewichte[i] += fehlersignal * self.ausgabe



# Initialisierung des Netzes
def netz_erstellen_neu():
    global netz
    netz = [
    [Neuron((2*zufall.random(9)-1).tolist()) for i in range(5)], # erste versteckte Ebene mit 5 Neuronen
    [Neuron((2*zufall.random(5)-1).tolist()) for i in range(4)], # zweite versteckte Ebene mit 4 Neuronen
    [Neuron((2*zufall.random(4)-1).tolist()) for i in range(1)]] # Ausgabeebene mit 1 Neuron

# um das Training zu überspringen können eigens gewählte Gewichte eingefügt werden
def netz_erstellen_vorbestimmte_gewichte(gewichtsarray):
    global netz
    netz = [
    [Neuron(gewichtsarray[0][i]) for i in range(5)],
    [Neuron(gewichtsarray[1][i]) for i in range(4)],
    [Neuron(gewichtsarray[2][i]) for i in range(1)]
    ]



# Laden der Trainingsdaten.
# Es werden drei von vier Rotationen zusätzlich generiert um genügend Trainingsmaterial zu erhalten
# Die vierte Rotation wird zum Testen des Netzes verwendet
def trainingsdaten_laden():
    with open('trainingsdaten_tictactoe_endstadien.txt') as datei:
        zeilen = datei.readlines()

        for zeile in zeilen:
            brett = []
            for zeichen in zeile:
                if zeichen == "X":
                    brett.append(1)
                elif zeichen == "O":
                    brett.append(0)
                elif zeichen == "_":
                    brett.append(0.5)
            trainingsdaten.append(brett)
            rot180 = brett.copy()
            rot180.reverse()
            trainingsdaten.append(rot180) #Umkehrung der Liste = Rotation um 180°
            rot90 = [brett[6],brett[3],brett[0],brett[7],brett[4],brett[1],brett[8],brett[5],brett[2]]
            trainingsdaten.append(rot90.copy()) # Rotation um 90° im Uhrzeigersinn
            rot90gegen = rot90.copy()
            rot90gegen.reverse()
            trainingsdaten.append(rot90gegen)

# Generation der erwarteten Ausgabewerte
def trainingsausgabe_generieren():
    global trainingsdaten
    global erwartet
    for brett in trainingsdaten:

        if any([
            brett[0] == brett[1] == brett[2]==1,
            brett[3] == brett[4] == brett[5]==1,
            brett[6] == brett[7] == brett[8]==1,
            brett[0] == brett[3] == brett[6]==1,
            brett[1] == brett[4] == brett[7]==1,
            brett[2] == brett[5] == brett[8]==1,
            brett[0] == brett[4] == brett[8]==1,
            brett[2] == brett[4] == brett[6]==1]
            ):
            erwartet.append([1])
        elif any([
            brett[0] == brett[1] == brett[2]==0,
            brett[3] == brett[4] == brett[5]==0,
            brett[6] == brett[7] == brett[8]==0,
            brett[0] == brett[3] == brett[6]==0,
            brett[1] == brett[4] == brett[7]==0,
            brett[2] == brett[5] == brett[8]==0,
            brett[0] == brett[4] == brett[8]==0,
            brett[2] == brett[4] == brett[6]==0]):
            erwartet.append([0])
        else:
            erwartet.append([0.5])




def trainieren(input, erwartet):
    global netz
    ergebnis_e0 = []
    ergebnis_e1 = []
    ergebnis_e2 = []
    # Forward Propagation
    for neuron in netz[0]:
        ergebnis_e0.append(neuron.ausgabe_berechnen(input))

    for neuron in netz[1]:
        ergebnis_e1.append(neuron.ausgabe_berechnen(ergebnis_e0))

    for neuron in netz[2]:
        ergebnis_e2.append(neuron.ausgabe_berechnen(ergebnis_e1))
        print("Ausgabe: " + str(ergebnis_e2) + " Erwartet: " + str(erwartet))

    # Berechnung der Fehlersignale
    fehlersignal_e2 = [sigmoid_abl(ergebnis_e2[i])*(erwartet[i]-ergebnis_e2[i]) for i in range(len(netz[2]))]
    fehlersignal_e1 = [sigmoid_abl(ergebnis_e1[i])*(sum([fehlersignal_e2[j] * netz[2][j].gewichte[i] for j in range(len(netz[2]))])) for i in range(len(netz[1]))]
    fehlersignal_e0 = [sigmoid_abl(ergebnis_e0[i])*(sum([fehlersignal_e1[j] * netz[1][j].gewichte[i] for j in range(len(netz[1]))])) for i in range(len(netz[0]))]

    # Backpropagation

    for neuron in netz[2]:
        neuron.gewichte_anpassen(ergebnis_e2, fehlersignal_e2[netz[2].index(neuron)])
        print("Neuron (2, " + str(netz[2].index(neuron)) + ") besitzt nun die Gewichte " + str(neuron.gewichte))

    for neuron in netz[1]:
        neuron.gewichte_anpassen(ergebnis_e1, fehlersignal_e1[netz[1].index(neuron)])
        print("Neuron (1, " + str(netz[1].index(neuron)) + ") besitzt nun die Gewichte " + str(neuron.gewichte))

    for neuron in netz[0]:
        neuron.gewichte_anpassen(input, fehlersignal_e0[netz[0].index(neuron)])
        print("Neuron (0, " + str(netz[0].index(neuron)) + ") besitzt nun die Gewichte " + str(neuron.gewichte))

    print("Runde abgeschlossen")


def testen(input):
    ergebnis_e0 = []
    ergebnis_e1 = []
    ergebnis_e2 = []

    for neuron in netz[0]:
        ergebnis_e0.append(neuron.ausgabe_berechnen(input))

    for neuron in netz[1]:
        ergebnis_e1.append(neuron.ausgabe_berechnen(ergebnis_e0))

    for neuron in netz[2]:
        ergebnis_e2.append(neuron.ausgabe_berechnen(ergebnis_e1))
        print("Ausgabe: " + str(ergebnis_e2))

def gewichtsarray_generieren():
    gewichtsarray = [[netz[0][i].gewichte for i in range(5)],[netz[1][i].gewichte for i in range(4)],[netz[2][i].gewichte for i in range(1)]]
    return gewichtsarray

# Main Methode || Aus- bzw. Einkommentieren welche Funktion gewünscht ist.
def main():
    global trainingsdaten
    global erwartet
    #netz_erstellen_neu()
    netz_erstellen_vorbestimmte_gewichte([[[-0.38376494208810535, -0.20492030760292435, 0.7562939276430493, 0.5140723544795324, -0.05621788581861621, -0.17280913128965786, 0.3581805201519352, -0.46496861581501375, 0.5070751010068247], [0.8783054811226074, -0.5088088201581512, 0.8924620542493695, 0.3291746567834597, -0.8135043782290428, -0.12162091708165673, 0.7676595892377684, 0.3896067503467682, -0.35235452127705175], [0.08126259345946149, -0.9463238221097126, -1.2234045941162355, -1.066802531050606, -0.7063933632912656, -0.4562074257965813, -0.8537516766191431, 0.24495907364892436, -1.0000049546216976], [0.7883277644205706, 0.7127191151560386, 1.726525639396013, 2.2388734208142114, 1.7326320954536107, 2.393366334338243, 1.9789523342502264, 2.2504922468527817, 2.388065215216914], [0.1293383888587997, 0.9123122882295807, 0.02694225085173425, -0.41548726432594224, -0.0594762143563984, 0.36704421749314337, -0.30125176837180423, 0.843874384310545, -0.4488852787528486]], [[-0.10427346009856749, -0.26622333803333614, -0.07303717543063294, -0.7738854678849276, 0.4732789528801693], [-0.17291914351874016, -0.601509179393068, 0.4959733961438057, -0.3850319178065535, -0.12890778392979474], [-0.23294407121498903, -0.8143130108417966, -0.6788599667963313, -0.16605146230012946, -0.16197998280483428], [-0.5011841697109707, -0.41651834105387026, -0.45351463772243006, -0.4904904523673725, -0.046125691751353205]], [[0.6969972535200977, 1.2643003741632444, 0.7655618324785706, 1.950004485750926]]]) #Daten aus einem vorherigen Trainingsdurchlauf
    testen([0,0.5,1,0,0.5,1,0,1,0.5]) # Überprüft ob das Netz einen O Gewinn erkennt
    testen([1,0.5,0,1,0.5,0,1,0,0.5]) # Überprüft ob das Netz einen X Gewinn erkennt


    # trainingsdaten_laden()
    # trainingsausgabe_generieren()
    # for i in range(1000):
    #     for input in trainingsdaten:
    #         trainieren(input, erwartet[trainingsdaten.index(input)])
    # print(gewichtsarray_generieren())


if __name__ == '__main__':
    main()
