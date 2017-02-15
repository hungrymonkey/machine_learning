import  csv
import  numpy as np

def back_solve( mat, b ):
   n,m = np.shape(mat)
   solution = np.zeros(m)
   for i in reversed(range(m)):
       row = mat[i,i:]
       c = np.sum(solution[i:] * row)
       solution[i] = (b[i]-c) /row[0];
   return solution.T

def back_solve_ha( matb ):
   n,m = np.shape(matb)
   solution = np.zeros(m-1)
   mat = matb[:,:m-1]
   b = matb[:,m-1]
   for i in reversed(range(m-1)):
       row = mat[i,i:]
       c = np.sum(solution[i:] * row)
       solution[i] = (b[i]-c) /row[0];
   return solution.T

def product( i, a, b):
    return np.dot( a[i:], b[i:])

def transform( i, aj, vi ):
    fi = 2 * product(i, vi, aj)
    aj[i:] -= fi * vi[i:]

def eliminate( i, ai, vi ):
    anorm = product(i, ai,ai)
    if ai[i] > 0.0:
       dii = -anorm
    else:
       dii = anorm
    wii = ai[i] - dii
    fi = np.sqrt( -2.0 * wii * dii )
    vi[i] = wii/fi
    ai[i] = dii
    vi[i+1:] = ai[i+1:]/fi
    ai[i+1:] = 0.0


def qr_decompose( train, b ):
    a = np.c_[train,b]
    n,m = a.shape
    vi = np.zeros(n)
    for i in xrange(m-1):
        eliminate(i,a[:,i], vi )
        for j in xrange(i+1,m):
            transform( i, a[:,j], vi)
        transform( i, a[:,-1], vi )
    return a

def qr_decompose_test():
   a = np.array([[2,2,4],[1,3,-2],[3,1,3]])
   b = np.array([18,1,14])
   ha = qr_decompose(a,b)
   

# load  data
def main():

   with  open('regression-0.05.csv', 'rb') as  csvfile:
        reader = csv.reader(csvfile , delimiter=',')
        data = np.array ([[ float(r) for r in row] for row in  reader ])

   # create  random  sample  train  (80%)  and  test  set  (20%)
   np.random.seed (0)
   np.random.shuffle(data)

   train_num = int(data.shape [0] * 0.8)
   X_train = data[:train_num ,:-1]
   Y_train = data[:train_num ,-1]
   X_test   = data[train_num :,:-1]
   Y_test   = data[train_num :,-1]
   # linear  least  square Y = X beta
   #Q,R = np.linalg.qr(X_train) example qr composition
   #beta = back_solve(R, np.dot(Q.T, Y_train))
   ha = qr_decompose(X_train, Y_train)
   beta = back_solve_ha(ha)

   # root  mean  square  error
   print  np.sqrt(np.mean((np.dot(X_test , beta) - Y_test) ** 2))

if __name__ == "__main__":
    main()
