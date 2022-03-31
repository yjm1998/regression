import numpy as np

class linear():
    def __init__(self,itration,wlr,blr):
        self.iteration = itration
        self.wlr = wlr
        self.blr = blr


    def fit(self,x,y,method):
        self.w_grad, self.b_grad, self.w, self.b = np.random.rand(4)
        self.loss=0
        if method =='sgd':
            return self.sgd(x,y)
        if method =='adam':
            return self.adam(x,y)

    def lossfunction(self,x,y):

        ypredict = self.w*x + self.b
        self.loss = 0.5*np.power((ypredict-y), 2).sum()/x.shape[0]
        return ypredict,self.loss

    def sgd(self,x,y):
        for i in range(self.iteration):
            yhat,loss=self.lossfunction(x,y)
            self.w_grad = sum((yhat - y)*x)
            self.b_grad = sum(yhat - y)
            self.w -= self.wlr * self.w_grad
            self.b -= self.blr * self.b_grad
            if i % 50 == 0:
                print('sgd_iteration=', i, 'w=', self.w, 'b=', self.b,'loss=',loss)
            if abs(loss)<0.5:
                print('结果：', 'TOTAL_iteration:', i, 'lr:', self.wlr, self.blr, 'w:', self.w, 'b', self.b, 'loss:',
                      loss)
                break

        return self.loss, self.w_grad, self.b_grad, self.w, self.b

    def adam(self,x,y):
        b1 = 0.9
        b2 = 0.99
        m = 0
        G = 0
        e = 0.0000001

        for i in range(self.iteration):
            yhat, loss = self.lossfunction(x, y)
            self.w_grad = sum((yhat - y) * x)
            self.b_grad = sum(yhat - y)
            g=np.array([self.w_grad,self.b_grad])
            m = b1 * m + (1 - b1) * g
            G = b2 * G + (1 - b2) * g ** 2
            m_hat = m / (1 - b1 ** (i + 1))
            G_hat = G / (1 - b2 ** (i + 1))
            self.w -= (self.wlr * m_hat[0]) / (np.sqrt(G_hat[0] + e))
            self.b -= (self.blr * m_hat[1]) / (np.sqrt(G_hat[1] + e))
            if i % 100 == 0:
                print('adam_iteration=', i, 'w=', self.w, 'b=', self.b,'loss=',loss)
            if abs(loss) < 0.5:
                print('结果：', 'TOTAL_iteration:', i, 'lr:', self.wlr,self.blr, 'w:', self.w,'b',self.b, 'loss:', loss)
                break

if __name__=='__main__':
    x = np.arange(-100, 100,2)
    e = np.random.randn(100)
    y = 3 * x + 4 + e
    model1=linear(itration=350,wlr=0.0000005,blr=0.005)
    model1.fit(x,y,method='sgd')
    print('___________________________________')
    model2=linear(itration=1000,wlr=0.01,blr=0.01)
    model2.fit(x,y,method='adam')




