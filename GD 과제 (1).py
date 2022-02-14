import pandas as pd
import numpy as np
import pymysql
import time
import sys

def load_dbscore_data():
    conn = pymysql.connect(host='localhost', user='datascience', password='datascience', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from db_score"
    curs.execute(sql)
    
    data  = curs.fetchall()
    
    curs.close()
    conn.close()


    
    X = [ (t['attendance'], t['homework'], t['final'] ) for t in data ]
    #X = [ ( t['midterm'] ) for t in data ]
    X = np.array(X)
    
    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X, y


def LS_linear_regression(X, y):

    import statsmodels.api as sm
    X_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_const)
    ls = model.fit()
    
    print(ls.summary())
    
    return ls.params


def gradient_descent_naive(X, y, num_params):
    
    epochs = 1000000
    min_grad = 0.0001
    learning_rate = 0.001
    
    n = len(y)
    
    m = [0.0] * num_params
    c = 0.0
    
    for epoch in range(epochs):
        m_grad = [0.0] * num_params
        c_grad = 0
        
        for i in range(n):
            y_pred = c
            for j in range(num_params):
                y_pred += m[j] * X[i][j]
            
            c_grad += 2*(y_pred - y[i])
            for j in range(num_params):
                m_grad[j] += 2*(y_pred-y[i]) * X[i][j]
        
        c_grad /= n
        for j in range(num_params):
            m_grad[j] /= n
            
        c -= learning_rate * c_grad
        for j in range(num_params):
            m[j] -= learning_rate * m_grad[j]
            
        if epoch % 1000 == 0:
            print("epoch %d:"%epoch)
            print("gradients:", c_grad, m_grad)
            print("parameters:", c, m)
            print("\n")  
        
        if abs(c_grad) > min_grad:
                continue
            
        break_condition = True
        
        for j in range(num_params):
            if abs(m_grad[j]) < min_grad:
                continue
            else:
                break_condition = False
                break
        
        if break_condition:
            break
        
    return c, m
        


def gradient_descent_vectorized(X, y, num_params):
    epochs = 1000000
    min_grad = 0.0001
    learning_rate = 0.0015
    
    m = np.zeros(num_params)
    c = 0.0
    
    #print('m=', m)
    #print('m.shape=', m.shape)
    
    n = len(y)
    
    
    for epoch in range(epochs):
        y_pred = np.sum(m*X, axis=1) + c
        
        #print('y_pred=', y_pred)
        #print('y_pred.shaped=', y_pred.shape)
        
        m_grad = np.sum(2*(y_pred-y)*np.transpose(X), axis=1)/n
        c_grad = np.sum(2*(y_pred-y))/n
        
        #print('m_grad=', m_grad)
        #print('c_grad=', c_grad)

        m -= learning_rate * m_grad
        c -= learning_rate * c_grad
        
        
        if epoch % 1000 == 0:
            print("epoch %d:"%epoch)
            print("gradients:", c_grad, m_grad)
            print("parameters:", c, m)
            print("\n")  
            
            
        if abs(c_grad) > min_grad:
            continue
        
        if np.all(np.absolute(m_grad) < min_grad):
            break
        

    return c, m
    

if __name__ == '__main__':
    X, y = load_dbscore_data()  
    '''
    params = LS_linear_regression(X, y)
    print(params)
    '''

    '''    
    start_time = time.time()
    c, m = gradient_descent_naive(X, y, 3)
    end_time = time.time()
    print("%f seconds" %(end_time - start_time))    
    '''    

    start_time = time.time()
    c, m = gradient_descent_vectorized(X, y, 3)
    end_time = time.time()
    print("%f seconds" %(end_time - start_time)) 



























     