



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
    vector = [[0,0],[0,0]]
    for i in range(len(a)):
        for k in range(len(b)):
            vector[i][k]= (a[i][k] * b[k][i]) + (a[i][k] * b[k+1][i])
    print(vector)

punto(a,b)
matriz(AM,BM)


