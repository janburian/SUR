import numpy as np
import math

def count_values_normal_distribution(C_list: list, dim: int, varis, means, vectors):
    result_list = list()
    for vector in vectors:
        i = 0
        probs = list()
        for C, var, mean in zip(C_list[0], varis[0], means[0]):
            prob_part1 = 1 / (var * math.sqrt(2 * math.pi))
            prob_part2 = math.exp(-0.5 * ((vector - mean) ** 2 / (var ** 2)))
            res = C * prob_part1 * prob_part2
            probs.append(res)
            i += 1

        result_list.append(probs)

    return result_list

def count_ln_L(values_normal_distribution):
    res = 0
    for probs in values_normal_distribution:
        sum_probs = sum(probs)
        ln_sum_probs = math.log(sum_probs)
        res += ln_sum_probs

    return res

def count_new_values_condtitional_probability(values_normal_distribution):
    res = list()
    for probs in values_normal_distribution:
        temp_list = list()
        for i in range(len(probs)):
            numerator = probs[i]
            denominator = sum(probs)
            temp_list.append(numerator / denominator)

        res.append(temp_list)

    return res

def count_new_C(M: int, N: int, cond_probs: list):
    C_res = [0] * M

    for probs in cond_probs:
        i = 0
        for prob in probs:
            C_res[i] += prob
            i += 1

    C_res = [x / N for x in C_res]
    test = sum(C_res) # = 1

    return C_res

def count_new_means(vectors, M: int, cond_probs: list):
    res = [0] * M
    cond_probs_transposed = np.array(cond_probs).T.tolist()
    for i in range(len(res)):
        cond_probs = cond_probs_transposed[i]
        numerator = 0
        denominator = sum(cond_probs)
        for vector, prob in zip(vectors, cond_probs):
            numerator += prob * vector
        res[i] = numerator / denominator

    return res

def count_new_variances(vectors, M: int, cond_probs: list, new_means):
    res = [0] * M
    cond_probs_transposed = np.array(cond_probs).T.tolist()
    for i in range(len(res)):
        cond_probs = cond_probs_transposed[i]
        numerator = 0
        denominator = sum(cond_probs)
        for vector, prob in zip(vectors, cond_probs):
            numerator += prob * (vector - new_means[i]) ** 2
        res[i] = math.sqrt(numerator / denominator)

    return res

if __name__ == "__main__":
    dim = 1
    T = [-4, -2, -1, 1, 2, 5]
    N = len(T)
    epsilon = 0.01

    # Starting parameters
    C_1 = 0.5
    C_2 = 0.5

    mean_1 = -5
    mean_2 = 5


    var_1 = 1
    var_2 = 1

    iteration = 0
    C = ([C_1, C_2], iteration)
    means = ([mean_1, mean_2], iteration)
    varis = ([var_1, var_2], iteration) # tuple (list of matrices, iteration)

    L_previous = -math.inf
    for i in range(5):
        print(f"Iteration: {iteration}")
        print(f"C_{iteration}")
        idx = 1
        for c in C[0]:
            print(f"C_{idx}^{iteration} = {c}")
            idx += 1
        print('-------------------------------')
        print(f"Means_{iteration}")
        idx = 1
        for mean in means[0]:
            print(f"mean_{idx}^{iteration} = {mean}")
            idx += 1
        print('-------------------------------')
        print(f"Variance_{iteration}")
        idx = 1
        for var in varis[0]:
            print(f"Variance_{idx}^{iteration} = {var}")
            idx += 1

        values_normal_distribution = count_values_normal_distribution(C, dim, varis, means, T)
        L = count_ln_L(values_normal_distribution) # likelihood
        print(f"Likelihood^{iteration} = {L}")
        print()
        print()
        if (L - L_previous) < epsilon: # Stop condition
             break
        cond_probs = count_new_values_condtitional_probability(values_normal_distribution)
        C_new = count_new_C(2, 6, cond_probs)
        means_new = count_new_means(T, 2, cond_probs)
        varis_new = count_new_variances(T, 2, cond_probs, means_new)

        iteration += 1
        C = (C_new, iteration)
        means = (means_new, iteration)
        varis = (varis_new, iteration)
        L_previous = L


