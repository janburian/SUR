import numpy as np

def vynasob_vektory(vec1, vec2):
    return vec1.dot(vec2)

def spocti_ro(c: int, t: int):
    return c / (t+1)

def vypocitej_souciny(T, q):
    result_list = []

    for vec in T:
        res = vynasob_vektory(q, vec)
        result_list.append(res)

    return result_list

def ziskej_T_error(T1: list, T2: list, q):
    T_err = []

    souciny_T1 = vypocitej_souciny(T1, q)
    souciny_T2 = vypocitej_souciny(T2, q)

    i = 0
    T1_err_pomocna = []
    for soucin in souciny_T1:
        if soucin <= 0:
            T1_err_pomocna.append(i)
            i += 1
        else:
            i += 1
    T_err.append(T1_err_pomocna)

    T2_err_pomocna = []
    for soucin in souciny_T2:
        if soucin >= 0:
            T2_err_pomocna.append(i)
            i += 1
        else:
            i += 1
    T_err.append(T2_err_pomocna)

    return T_err

def spocti_nove_q(q, T: list, ro, T_err):
    suma = 0

    kappa = [-1, 1]
    i = 0
    for error in T_err:
        for idx in error:
            suma += kappa[i] * T[idx]
        i += 1

    nove_q = q - ro * suma

    return nove_q

if __name__ == "__main__":
    #T1 = [np.array([1, 2, 1]), np.array([1, 2, 4]), np.array([1, 0, 2])]
    T2 = [np.array([1, 0, 1]), np.array([1, -2, 1]), np.array([1, -1, -2])]
    T3 = [np.array([1, 0, -1]), np.array([1, 3, 1]), np.array([1, 1, -2])]

    # Ze cviceni
    '''
    T1 = [np.array([1, 2, 1]), np.array([1, 2, -2]), np.array([1, 0, -2])]
    T2 = [np.array([1, 0, -1]), np.array([1, -2, -3]), np.array([1, -1, 1])]
    T = T1 + T2
    '''

    T = T2 + T3

    t = 0
    c = 0.5
    q = np.array([0, 0, 1])  # primka x; pocatecni

    print("Iteration " + str(t) + ": " + "q = " + str(q))
    while True:
        T_error = ziskej_T_error(T2, T3, q) # list listu
        if not any(T_error):
            print("Empty T_error!")
            break
        ro = spocti_ro(c, t)
        new_q = spocti_nove_q(q, T, ro, T_error)
        print("Iteration " + str(t+1) + ": " + "q = " + str(new_q))
        q = new_q

        t += 1
