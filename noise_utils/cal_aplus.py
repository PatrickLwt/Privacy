import cvxpy as cp
import numpy as np
# import tensorflow as tf



def getAplus(m):
  X = cp.Variable((m, m))
  U = cp.Variable(m)
  W = cp.Parameter((m, m))
  W.value = np.identity(m)

  constraints = []
  for i in range(m):
    wi = cp.reshape(W[i, :], (1, m))
    wiT = wi.T
    ui = cp.reshape(U[i], (1, 1))
    New = cp.bmat([[X, wiT], [wi, ui]])
    constraints += [New >> 0, X[i, i] <= 1]

  obj = cp.Minimize(cp.sum(U))

  prob = cp.Problem(obj, constraints)
  prob.solve()
  x = X.value
  a = np.linalg.cholesky(x)
  ap = np.linalg.pinv(a)

  print("Gotten Aplus!!")
  
  return ap

def getAplusWithW(w):
  # m, n = list(w.size())
  m, n = w.shape
  X = cp.Variable((n, n), PSD=True)
  U = cp.Variable(m)
  W = cp.Parameter((m, n))
  W.value = w
  # try:
  #   W.value = w.cpu().numpy()
  # except:
  #   W.value = w.cpu().detach().numpy()
  constraints = []
  for i in range(m):
    wi = cp.reshape(W[i, :], (1, n))
    wiT = wi.T
    ui = cp.reshape(U[i], (1, 1))
    New = cp.bmat([[X, wiT], [wi, ui]])
    # constraints += [cp.lambda_min(New) >= 0, X[i, i] <= 1]
    constraints += [New >> 0]

  for i in range(n):
    constraints += [X[i, i] <= 1]

  obj = cp.Minimize(cp.sum(U))

  prob = cp.Problem(obj, constraints)
  prob.solve()
  x = X.value

  ## get eigenvalues and make them positive
  e_vals,e_vecs = np.linalg.eig(x)
  e_vals[e_vals<0] = 1.e-10

  # print("min e_Vals: ", np.min(e_vals))

  ## reconstruct the matrix
  diag = np.diag(e_vals)
  psdx = np.matmul(e_vecs, np.matmul(diag, np.linalg.inv(e_vecs)))

  ## get a and pinv a 
  a = np.linalg.cholesky(psdx)
  ap = np.linalg.pinv(a) 
  # print("ap: ", ap)
  
  return ap
