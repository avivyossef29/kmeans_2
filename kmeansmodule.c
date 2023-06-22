#define  _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
# define PY_SSIZE_T_CLEAN
# include <Python.h>


double** create_matrix(int n , int size_of_vector);
void set_matrix(PyObject* lis_of_vectors ,double ** matrix ,  int n , int size_of_vector);
PyObject* result_to_py(double** centroids,int k,int size_of_vector);
void free_matrix(double** matrix , int n );
double find_distance(double *u , double*v , int size_of_vector);
double ** k_means(int k , int iter , double epsilon , double** data , double** centroids , int N , int size_of_vector);
int find_closet_cluster(double* u,int k ,int size_of_vector,double** centroids);
void update_closet_centroids(double** sums ,int* cluster_size,int N ,int k , int size_of_vector,double** centroids , double** data );
int calc_new_centroids(double epsilon ,double** sums, int* cluster_size,int k , int vector_size,double** centroids);
void restart(double** sums , int* cluster_size,int size_of_vector,int k);


char* error = "An Error Has Occurred";

double** create_matrix(int n , int size_of_vector){
    double** data = (double**)malloc(n * sizeof(double*));
    double* curr;
    if (data == NULL){
        return NULL;
    }
    for(int i = 0 ; i < n ; i++){
        curr = (double*)malloc(size_of_vector*sizeof(double));
        if (curr == NULL){
            printf("%s",error);
            return NULL;
        }
        data[i] = curr;
    }
    return data;
    
}

void free_matrix(double** matrix , int n){
    for(int i = 0 ; i < n ; i++){
            free(matrix[i]);
    }
    free(matrix);
}



double find_distance(double *u , double*v , int size_of_vector){
    double result = 0.0;
    for(int i = 0 ; i < size_of_vector ; i++){  
        result += pow(v[i] - u[i],2);
    }
    return sqrt(result);
}


int find_closet_cluster(double* u,int k ,int size_of_vector,double** centroids){
    double minn;
    int index_min = 0;
    double curr_dis;
    
    for(int i = 0 ; i < k ; i ++){
        if (i == 0){
            minn = find_distance(u,centroids[i],size_of_vector);
            continue;
        }
        curr_dis = find_distance(u,centroids[i],size_of_vector);

        if (curr_dis < minn){
            minn = curr_dis;
            index_min = i;
        }
    }
    return index_min;
}

void update_closet_centroids(double** sums ,int* cluster_size,int N ,int k , int size_of_vector,double** centroids , double** data ){
    double* curr;
    int close_cluster;
    for (int i = 0 ; i < N; i++){
        curr = data[i];
        close_cluster = find_closet_cluster(data[i],k,size_of_vector , centroids);
        cluster_size[close_cluster] +=1;
        for(int j = 0 ; j < size_of_vector  ; j++ ){
            sums[close_cluster][j] += curr[j]; 
        }
    }
}

int calc_new_centroids(double epsilon ,double** sums, int* cluster_size,int k , int vector_size,double** centroids){
    int delta = 0;
    double distance_old_new;

    for (int i = 0 ; i < k ; i ++){
        double* new_centroid = (double*)malloc(vector_size*sizeof(double));
        double* old_centroid = centroids[i];
        for (int j = 0 ; j < vector_size ; j++){
            new_centroid[j] = sums[i][j]/cluster_size[i];
        }
        if (delta == 0){
            distance_old_new = find_distance(old_centroid,new_centroid,vector_size);
            if (distance_old_new> epsilon){
                delta += 1;
            }
        }
        centroids[i] = new_centroid;
        free(old_centroid);
    }
    return delta;
}

void restart(double** sums , int* cluster_size,int size_of_vector , int k){
    for(int i = 0 ; i < k ; i++ ){
        cluster_size[i] = 0;
        for(int j = 0 ; j < size_of_vector ; j++){
            sums[i][j] = 0.0;
        }
    }
}


double ** k_means(int k , int iter , double epsilon , double** data , double** centroids , int N , int size_of_vector){
    int delta;
    double** sums = create_matrix(k,size_of_vector);
    int* cluster_size = (int*)malloc(sizeof(int) *k);
    if(sums == NULL || cluster_size == NULL ){
        printf("%s",error);
        return NULL;
    }
    restart(sums,cluster_size,size_of_vector,k);

    for (int i = 0 ; i < iter ; i++){

        update_closet_centroids(sums,cluster_size,N,k,size_of_vector,centroids,data);
        delta = calc_new_centroids(epsilon,sums,cluster_size,k,size_of_vector,centroids);
        if (delta == 0){
            free(sums);
            free(cluster_size);
            return centroids;
        }

        restart(sums,cluster_size,size_of_vector,k);
    }
    free(sums);
    free(cluster_size);
    return centroids;
}





void set_matrix(PyObject* lis_of_vectors ,double ** matrix ,  int n , int size_of_vector){
    PyObject* curr;
    for(int i = 0 ; i < n;  i++ ){
        curr = PyList_GetItem(lis_of_vectors, i);
        for(int j = 0 ;  j < size_of_vector; j++){
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(curr,j ));
        }
    }
}

PyObject* result_to_py(double** centroids,int k,int size_of_vector){
    PyObject* result = PyList_New(k);
    PyObject* curr;
    for (int i = 0; i < k; ++i){
        curr = PyList_New(size_of_vector);
        for (int j = 0; j < size_of_vector; ++j) {
            PyList_SetItem(curr, j, Py_BuildValue("d", centroids[i][j]));
        }
        PyList_SetItem(result, i, curr);
    }
    return result;
}

static PyObject* fit(PyObject *self, PyObject *args){
    PyObject* centroids_1d;
    PyObject* data_1d; 
    PyObject* result;
    int N = 0;
    int size_of_vector = 0;
    double epsilon = 0.0;
    int iter = 0;
    int k = 0;
    double** centroids;
    double** data;

    if(!PyArg_ParseTuple(args, "iiiifOO", &k, &iter,&size_of_vector
    ,&N ,&epsilon,  &centroids_1d, &data_1d)) {/*  transfer python variables to */
        return NULL;
    }
    centroids = create_matrix(k,size_of_vector);
    data = create_matrix(N,size_of_vector);
    set_matrix(centroids_1d , centroids , k , size_of_vector);
    set_matrix(data_1d , data , N , size_of_vector);
    centroids = k_means(k, iter, epsilon,data , centroids, N, size_of_vector);
    if (centroids == NULL){
        return NULL;
    }
    result = result_to_py(centroids , k , size_of_vector);
    free_matrix(centroids , k);
    free_matrix(data ,N);
    return result;
}

static PyMethodDef kmeansMethods[] = {{"fit", (PyCFunction) fit, METH_VARARGS, PyDoc_STR("function expects: number of clusters, max iteration, epsilon, vectors list, centroitds list")}, {NULL,NULL,0,NULL}};

static struct PyModuleDef kmeansmodule = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        kmeansMethods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&kmeansmodule);
    if (!m) {
        Py_RETURN_NONE;
    }
    return m;
}

