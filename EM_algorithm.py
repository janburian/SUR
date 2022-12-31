import numpy as np
import math
import matplotlib.pyplot as plt

def count_values_normal_distribution(C_list: list, dim: int, cov_matrices, means, vectors):
    result_list = list()
    for vector in vectors:
        i = 0
        probs = list()
        for C, cov, mean in zip(C_list[0], cov_matrices[0], means[0]):
            prob_part1 = 1 / ((2 * math.pi)**(dim/2) * (math.sqrt(np.linalg.det(cov))))
            prob_part2 = math.exp(-0.5 * (vector - mean) * np.linalg.inv(cov) * (vector - mean).reshape(-1, 1))
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
    res = [np.array([0, 0])] * M
    cond_probs_transposed = np.array(cond_probs).T.tolist()
    for i in range(len(res)):
        cond_probs = cond_probs_transposed[i]
        numerator = 0
        denominator = sum(cond_probs)
        for vector, prob in zip(vectors, cond_probs):
            numerator += prob * vector
        res[i] = numerator / denominator

    return res

def count_new_covariances(vectors, M: int, cond_probs: list, new_means):
    res = [np.matrix('0 0; 0 0')] * M
    cond_probs_transposed = np.array(cond_probs).T.tolist()
    for i in range(len(res)):
        cond_probs = cond_probs_transposed[i]
        numerator = np.array([[0, 0],
                              [0, 0]])
        denominator = sum(cond_probs)
        for vector, prob in zip(vectors, cond_probs):
            numerator = np.asmatrix(numerator + prob * ((vector - new_means[i]) * (vector - new_means[i]).reshape(-1, 1)))
            #numerator = np.asmatrix(prob * ((vector - new_means[i]) * (vector - new_means[i]).reshape(-1, 1)))
            #tmp = (vector - new_means[i]) * (vector - new_means[i]).reshape(-1, 1)
        res[i] = numerator / denominator

    return res

def seperate_data_to_classes(vectors, cond_probs: list, R: int):
    res = []
    for i in range(R):
        res.append([])

    idx = 0
    for probs in cond_probs:
        max_prob = max(probs)
        idx_max_prob = probs.index(max_prob)
        res[idx_max_prob].append(vectors[idx])
        idx += 1

    return res

def plot_classes_distribution(separated_data: list, means: list):
    colours = ["red", "green", "blue", "magenta"]
    i = 0
    for cluster in separated_data:
        for vector in cluster:
            x = vector[0]
            y = vector[1]
            plt.scatter(x, y, color=colours[i])
        i += 1

    for mean in means[0]:
        x = mean[0]
        y = mean[1]
        plt.scatter(x, y, color='black', marker='+')

    plt.xlabel('x')
    plt.ylabel('y')
    #plt.axvline(x=0, c="black", label="x=0")
    #plt.axhline(y=0, c="black", label="y=0")
    plt.title("EM algorithm")
    plt.show()

if __name__ == "__main__":
    dim = 2
    R = 3
    vectors = [np.array([-3, -2]), np.array([3, -3]), np.array([0, 1]), np.array([-5, -2]), np.array([2, -2]),
               np.array([1, -1]), np.array([-2, 2]), np.array([-2, 1]), np.array([-5, -4]), np.array([1, -3])]
    N = len(vectors)
    epsilon = 0.01

    # Starting parameters
    C_1 = 1 / 3
    C_2 = 1 / 3
    C_3 = 1 / 3

    mean_1 = np.array([-5, -1])
    mean_2 = np.array([1, -2])
    mean_3 = np.array([-1, 4])

    cov_1 = np.matrix('1 0; 0 1')
    cov_2 = np.matrix('1 0; 0 1')
    cov_3 = np.matrix('1 0; 0 1')

    iteration = 0
    C = ([C_1, C_2, C_3], iteration)
    means = ([mean_1, mean_2, mean_3], iteration)
    cov_matrices = ([cov_1, cov_2, cov_3], iteration) # tuple (list of matrices, iteration)

    L_previous = -math.inf
    for i in range(100):
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
        print(f"Covariances_{iteration}")
        idx = 1
        for cov in cov_matrices[0]:
            print(f"covariance_{idx}^{iteration} = {cov}")
            idx += 1

        values_normal_distribution = count_values_normal_distribution(C, dim, cov_matrices, means, vectors)
        L = count_ln_L(values_normal_distribution) # likelihood
        print(f"Likelihood^{iteration} = {L}")
        print()
        print()
        cond_probs = count_new_values_condtitional_probability(values_normal_distribution)
        if (L - L_previous) < epsilon: # Stop condition
            separated_data = seperate_data_to_classes(vectors, cond_probs, R)
            plot_classes_distribution(separated_data, means)
            break
        C_new = count_new_C(3, 10, cond_probs)
        means_new = count_new_means(vectors, 3, cond_probs)
        covariances_new = count_new_covariances(vectors, 3, cond_probs, means_new)

        iteration += 1
        C = (C_new, iteration)
        means = (means_new, iteration)
        cov_matrices = (covariances_new, iteration)
        L_previous = L


