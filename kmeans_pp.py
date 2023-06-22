import sys
import pandas as pd
import numpy as np
import mykmeanssp


iterr ,k , N ,epsilon  = None , None , None , None
erors = {"k": "invalid number of clusters!" ,"iterr":"Invalid maximum iteration!"}

def check_args():
    if k<= 1 or k>=N:
        print(erors["k"])
        exit()
    if iterr >= 1000 or iterr <= 1:
        print(erors["iterr"])
        exit()

def handle_argument():
    n = len(sys.argv)
    global k, iterr, eps, file_1, file_2
    if n == 6:
        k  , iterr , eps, file_1 , file_2 = sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    elif n == 5:
        k  , eps , file_1  , file_2 = sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4]
        iterr = 300
    else:
        exit()
    try:    
        k = int(k)
    except:
        print([erors["k"]])
        exit()
    try:
        iterr = int(iterr)
    except:
        print([erors["iterr"]])
        exit()
    try:
        eps = float(eps)
    except:
        exit()
        
    return (file_1 ,file_2)

def combine_files(file_1 ,file_2):
    global N
    data_1 = pd.read_csv(file_1,header=None)
    data_2 = pd.read_csv(file_2,header=None)
    data = data_1.merge(data_2,'inner',0)
    data = data.sort_values(0,ascending= True)
    N = data.shape[0]
    return data

def calc_distance(u, v):
    distance = np.linalg.norm(u-v)
    return distance



def update_distance(new_centroid,distance,data):
    sum_distance = 0.0
    for i in range(N):
        curr = data[i]
        old_distance = distance[i]
        new_distance = calc_distance(new_centroid,curr)
        distance[i] = old_distance if old_distance < new_distance else new_distance
        sum_distance += distance[i]
    return sum_distance


def update_p(p,distance,sum_distance):
    for i in range(N):
        p[i] = distance[i]/sum_distance

def print_initial_centroids(chosen_index):
    chosen_index = [str(chosen_index[i]) for i in range(len(chosen_index))]
    to_print = ','.join(chosen_index)
    print(to_print)


def pick_centroids(data):
    np.random.seed(0)
    index = [i for i in range(N)]
    first_index = np.random.choice(index)
    chosen_index = [first_index for i in range(k)]
    first_centroid = data[first_index]
    centroids = [first_centroid for i in range(k)]
    distance = [calc_distance(first_centroid,data[i]) for i in range(N)]
    probabilities = [0.0 for i in range(N)]
    sum_distance = update_distance(first_centroid,distance,data)
    update_p(probabilities,distance,sum_distance)

    for i in range(1,k):
        new_index = np.random.choice(index,p=probabilities)
        chosen_index[i] = new_index
        new_centroid = data[new_index]
        centroids[i] = new_centroid
        sum_distance = update_distance(new_centroid,distance,data)
        update_p(probabilities,distance,sum_distance)

    print_initial_centroids(chosen_index)
    return centroids

def print_centroids(centroids):
    size_vector = len(centroids[0])
    to_print = [['{:.4f}'.format(centroids[i][j]) for j in range(size_vector)] for i in range(k)]
    to_print = [",".join(to_print[i]) for i in range(k)]
    for i in range(k):
        print(to_print[i])

def numpy_to_float(data):
    for i in range(len(data)):
        data[i] = data[i].tolist()
        
                   


def main():
    files = handle_argument()
    data = combine_files(files[0],files[1])
    check_args()
    data_matrix = np.delete(data.values,0,axis=1)
    centroids = pick_centroids(data_matrix)
    numpy_to_float(centroids)
    data_matrix = data_matrix.tolist()
    centroids = mykmeanssp.fit(k, iterr,len(centroids[0]),len(data_matrix), eps,centroids,data_matrix)
    print_centroids(centroids) if centroids != None else None


main()


