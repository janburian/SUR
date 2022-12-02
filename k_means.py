# Importy modulů
import numpy as np
import matplotlib.pyplot as plt


# POMOCNE METODY (Eukleid, načítání soboru, matice vzdálenosti)
# Metoda nacita soubor s daty
def load(infile):
    fin = open(infile, "rt")
    poleVektoru = []
    for line in fin:
        souradniceX, souradniceY = line.split()
        v = tuple([float(souradniceX), float(souradniceY)])
        poleVektoru.append(v)
    fin.close()
    print("Data v pořádku načtena.")
    return poleVektoru


# Metoda, ktera pocita a vraci vzdalenost mezi jednotlivymi prvky
def vypoctiEukleidVzdalenost(vektor_X, vektor_Y):
    rozdilX = vektor_X[0] - vektor_Y[0]
    rozdilY = vektor_X[1] - vektor_Y[1]
    Eukleid_X = rozdilX ** 2
    Eukleid_Y = rozdilY ** 2
    dist = Eukleid_X + Eukleid_Y
    return dist


# Metoda pocita a vraci matici vzdalenosti
def vypoctiMaticiVzdalenosti(data):
    print("Počítám matici vzdálenosti.")
    dist = 0
    n = len(data)
    matrix = np.zeros((n, n))

    for i in range(len(matrix)):  # vypocet matice vzdalenosti (horni trojuhelnik)
        for j in range(i + 1, len(matrix)):
            matrix[i, j] = vypoctiEukleidVzdalenost(data[i], data[j])
    matrix = np.triu(matrix) + np.tril(matrix.T, 1)  # spojeni horni a dolni trojuhelnikove matice
    print("Matice vzdálenosti vypočtena.")
    return matrix


# ROZDĚLENÍ DAT DO JEDNOTLIVÝCH TŘÍD
# Pomocna metoda k metode k-means
def vypocitejMi(seznamIndexu, data):
    miX = 0
    miY = 0
    for i in range(len(seznamIndexu)):
        vektor = data[seznamIndexu[i]]
        miX += vektor[0]
        miY += vektor[1]
    miX = miX / len(seznamIndexu)
    miY = miY / len(seznamIndexu)
    mi = (miX, miY)
    return mi


# Metoda K-means
def k_means(R, data):
    stredy_zacatek = []  # pocet stredu je zavisly na poctu trid neboli R = pocet stredu
    listVzdalenostiBoduKeStredum = []
    finalListMinVzdalenosti = []
    stredy = []
    listTrid = []

    for i in range(R):
        stredy_zacatek.append(data[i])

    stredy = stredy_zacatek

    switch = 1
    while switch == 1:
        finalListMinVzdalenosti.clear()
        listMi = []
        J = 0
        J_previous = J

        for i in range(R):
            listTrid.append([])  # list listu na kazdem indexu je pridan dalsi list len(listTrid) = pocet trid

        for i in range(len(data)):
            bod = data[i]
            for j in range(len(stredy)):
                stred = stredy[j]
                dist = vypoctiEukleidVzdalenost(bod, stred)
                listVzdalenostiBoduKeStredum.append((dist, j))  # tuple -  vzdalenost a index sloupce
            minDist = min(listVzdalenostiBoduKeStredum)
            finalListMinVzdalenosti.append(minDist)
            listTrid[minDist[1]].append(data[i])
            listVzdalenostiBoduKeStredum.clear()

        pomocne_NoveStredy = []
        for i in range(0, R):
            pomocne_NoveStredy.append([])

        for i in range(len(finalListMinVzdalenosti)):
            tuple_minVzdalenost_sloupec = finalListMinVzdalenosti[i]
            index = tuple_minVzdalenost_sloupec[1]
            J += tuple_minVzdalenost_sloupec[0]
            pomocne_NoveStredy[index].append(i)

        print("Kriterialni hodnota - k-means: J = {}".format(J))

        listMi = []  # nove stredy
        for i in range(len(pomocne_NoveStredy)):
            mi = vypocitejMi(pomocne_NoveStredy[i], data)
            listMi.append(mi)

        if (stredy != listMi):
            if (J_previous == J):
                stredy == listMi
            if (J_previous * 0.95 < J):
                switch = 1
                listTrid.clear()
            stredy = listMi
        else:
            switch = 0

    print("HOTOVO - Data rozdělena pomocí k-means.")
    return [listTrid, listMi]

def vypoctiKriterialniFunkci(trida, stred):
    J = 0
    for vektor in trida:
        dist = vypoctiEukleidVzdalenost(vektor, stred)
        J += dist
    return J

# METODY - Vykreslování grafů
def vykresliData(data):
    print("Vykresluji data...")
    for vektor in data:
        X = vektor[0]
        Y = vektor[1]
        plt.scatter(X, Y)
        plt.title("Vykreslení dat")
        plt.xlabel('x')
        plt.ylabel('y')
    plt.axvline(x=0, c="black", label="x=0")
    plt.axhline(y=0, c="black", label="y=0")
    plt.show()


def vykresli_k_means(rozdeleniDoTrid_k_means, stredy_kmeans):
    print("Vykresluji k-means rozdělení...")
    colours = ["red", "green", "yellow", "magenta"]
    i = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color=colours[i])
        i += 1
    for stred in stredy_kmeans:
        plt.scatter(stred[0], stred[1], color='black', marker='+')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axvline(x=0, c="black", label="x=0")
    plt.axhline(y=0, c="black", label="y=0")
    plt.title("K-means")
    plt.show()


# VÝKONNÝ KÓD
if __name__ == "__main__":
    # Název souboru
    filename = "data.txt"

    # Načítání dat
    data = load(filename)

    # DĚLENÍ DO TŘÍD
    # K-means
    pocetTrid = 3  # potreba menit podle dat
    vysledek_k_means = k_means(pocetTrid, data)
    rozdeleniDoTrid_k_means = vysledek_k_means[0]
    stredy_z_k_means = vysledek_k_means[1]

    # VYKRESLOVÁNÍ GRAFŮ
    # Vykreslování dat
    vykresliData(data)

    # Vykreslování k-means
    vykresli_k_means(rozdeleniDoTrid_k_means.copy(), stredy_z_k_means)