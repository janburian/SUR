# Importy modulů
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math as math


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


# ZJIŠTĚNÍ POČTU TŘÍD
# Metoda MAXIMIN
def MAXIMIN(maticeVzdalenosti, index_pocatek, q):
    A = maticeVzdalenosti
    stredy = []
    listVzdalenostiPrvkuKeStredum = []
    slovnik = {}

    listNejkratsiVzdalenosti = []
    line = A[index_pocatek, :]  # startovni radek a pote radek, s nejvetsi vzdalenosti
    max_vzdalenost = (np.amax(line[np.nonzero(line)]))
    indexRadku_max = np.where(line == max_vzdalenost)
    index_data = indexRadku_max[0][0]

    stredy.append(data[index_pocatek])
    stredy.append(data[index_data])

    switch = 1  # promenna slouzici k zastaveni cyklu while
    while switch == 1:
        listNejkratsiVzdalenosti.clear()
        for i in range(len(A)):  # Vzdalenosti ke stredum
            bod = data[i]
            for j in range(len(stredy)):
                stred = stredy[j]
                dist = vypoctiEukleidVzdalenost(bod, stred)
                listVzdalenostiPrvkuKeStredum.append(dist)  # pomocny seznam
            listNejkratsiVzdalenosti.append(min(listVzdalenostiPrvkuKeStredum))
            slovnik.update({i: min(listVzdalenostiPrvkuKeStredum)})
            listVzdalenostiPrvkuKeStredum.clear()

        for i in range(len(stredy)):
            listNejkratsiVzdalenosti.remove(0)

        dMax = max(listNejkratsiVzdalenosti)
        listVzdalenostiStredu = []
        for u in range(len(stredy) - 1):
            stred1 = stredy[u]
            for e in range(u + 1, len(stredy)):
                stred2 = stredy[e]
                dist = vypoctiEukleidVzdalenost(stred1, stred2)
                listVzdalenostiStredu.append(dist)

        vypocet = 0
        soucetVzdalenosti = 0
        for i in range(len(listVzdalenostiStredu)):
            soucetVzdalenosti += listVzdalenostiStredu[i]
        vypocet = q * (1 / (len(stredy))) * soucetVzdalenosti

        if (dMax > vypocet):
            values = slovnik.values()
            maximum = max(values)  # maximum z minimalnich vzdalenosti
            index_novy_stred = list(slovnik.keys())[list(slovnik.values()).index(maximum)]
            stredy.append(data[index_novy_stred])
            switch = 1
        else:
            switch = 0

    pocetTrid = len(stredy)
    print("MAXIMIN - počet tříd je: " + str(pocetTrid))
    return pocetTrid


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
    plt.show()


# VÝKONNÝ KÓD
if __name__ == "__main__":
    # Název souboru
    filename = "data.txt"

    # Načítání dat
    data = load(filename)

    # Matice vzdálenosti
    maticeVzdalenosti = vypoctiMaticiVzdalenosti(data)
    print(maticeVzdalenosti)

    # URČENÍ POČTU TŘÍD
    # MAXIMIN
    q = 0.5
    index_pocatek_random = rand.randint(0, len(data) - 1)
    pocetTrid_MAXIMIN = MAXIMIN(maticeVzdalenosti.copy(), index_pocatek_random, q)






