import numpy as np

class SimpleLogisticRegression:

    def __init__(self, init_w, init_b):
        self.w = init_w
        self.b = init_b
        self.costs = []
    
    def fit(self, X_train, y_train, lr, epoch):
        for i in range(epoch):
            dldw, dldb = compute_gradient(X_train, y_train, self.w, self.b)
            self.w = self.w - lr * dldw
            self.b = self.b - lr * dldb
            if i % 1000 == 0:
                cost = compute_cost(X_train, y_train, self.w, self.b)
                print('Epoch: ', i, 'Cost: ', cost)
                self.costs.append(cost)


    def predict_proba(self, X_predict):
        y = self.w * X_predict + self.b
        y_pred = sigmoid(y)
        response = []
        y_other = 1 - y_pred
        response.append(y_other)
        response.append(y_pred)
        return response

    def predict(self, X_predict):
        y = self.w * X_predict + self.b
        y_pred = sigmoid(y)

        if y_pred > 0.5:
            return 1
        else:
            return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(x, y, w, b):
        m = len(x)
        total_cost = 0
        for i in range(m):
            fz = sigmoid(w * x[i] + b)
            total_cost += (- y[i] * np.log(fz)) - (1-y[i]) * np.log(1-fz)
        
        total_cost= total_cost/m
        return total_cost

def compute_gradient( x, y, w, b):
    m = len(x)
    dldw = 0
    dldb = 0
    for i in range(m):
        f_wb = sigmoid(w * x[i] + b)
        err_i = f_wb - y[i]
        dldw_i = err_i * x[i]
        dldb_i = err_i
        dldw += dldw_i
        dldb += dldb_i
    
    dldw = dldw/m
    dldb = dldb/m

    return dldw, dldb