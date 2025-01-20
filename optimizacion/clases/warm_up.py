



a = [1,2,3]
b = [4,-1,0]

def punto(a,b):
    suma = 0 
    for n in range(len(a)):
        suma += a[n] * b[n]
    print(suma)



AM = [[1,2],[3,4]]
BM = [[5,6],[7,8]]


def matriz(a,b):

    filas_a, columnas_a = len(a), len(a[0])
    filas_b, columnas_b = len(b), len(b[0])
    
    vector = [[0 for _ in range(columnas_b)] for _ in range(filas_a)]
    
    for i in range(filas_a):
        for j in range(columnas_b):
            for k in range(columnas_a):
                vector[i][j] +=  a[i][k] * b[k][j]
    print(vector)


punto(a,b)
matriz(AM,BM)


