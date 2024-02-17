import numpy as np
class MultipleLogisticRegression:

    def __init__(self, init_w, init_b):
        self.w = init_w
        self.b = init_b
    
    def fit(self, X_train, y_train, lr, epoch):
        for i in range(epoch):
            dldw, dldb = compute_gradient(X_train, y_train, self.w, self.b)
            self.w = self.w - lr * dldw
            self.b = self.b - lr * dldb
            if i % 100 == 0:
                print('Epoch: ', i, 'Cost: ', compute_cost(X_train, y_train, self.w, self.b))

    def predict_proba(self, X_predict):
        y_pred = sigmoid(np.dot(X_predict, self.w) + self.b)
        response = []
        y_other = 1 - y_pred
        response.append(y_other)
        response.append(y_pred)
        return response

    def predict(self, X_predict):
        y_pred = sigmoid(np.dot(X_predict, self.w) + self.b)
        if y_pred > 0.5:
            return 1
        else:
            return 0

def sigmoid(z):
    g= 1 / (1 + np.exp(-z))
    return g
    
def compute_cost(X, y, w, b):
    m,n = X.shape
    total_cost = 0
    for i in range(m):
        fx_i = np.dot(X[i], w) + b
        fz = sigmoid(fx_i)
        epsilon = 1e-10  # to avoid log(0)
        total_cost += (- y[i] * np.log(fz + epsilon)) - (1 - y[i]) * np.log(1 - fz + epsilon)
    total_cost= total_cost/m
    return total_cost

def compute_gradient(X, y, w, b):
    m,n = X.shape
    dldw = np.zeros(w.shape)
    dldb = 0.
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb - y[i]
        for j in range(n):
            dldw[j] = dldw[j] + err_i * X[i][j]
        dldb = dldb +  err_i
    dldw = dldw/m
    dldb = dldb/m
    return dldw, dldb
    

