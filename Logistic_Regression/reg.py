import numpy as np
import math

class LogisticRegression:
    def __init__(self, d):
        self.w = np.random.randn(d)
        #self.loss_func = 0.0
        #self.gradient = np.zeros_like(self.w)

    def compute_loss(self, X, Y):
        """
        Compute l(w) with n samples.
        Inputs:
            X  - A numpy array of size (n, d). Each row is a sample.
            Y  - A numpy array of size (n,). Each element is 0 or 1.
        Returns:
            A float.
        """
        loss_func = 0.0
        n = Y.size
        sum_of_terms = 0.0
        for i in range(n):
            prod = np.dot(X[i],self.w)
            term = math.log(1+ pow(math.e,prod)) -  prod*Y[i]
            sum_of_terms += term
        loss_func = sum_of_terms/n
        return loss_func

        #raise NotImplementedError

    def compute_grad(self, X, Y):
        """
        Compute the derivative of l(w).
        Inputs: Same as above.
        Returns:
            A numpy array of size (d,).
        """
        gradient = np.zeros_like(self.w)
        d = self.w.size
        n = Y.size

        for j in range(d):
            avg_term = 0.0
            sum_of_terms = 0.0
            for i in range(n):
                e_term = pow(math.e,np.matmul(self.w,X[i]))
                frac_term = 1/(1 + e_term)
                term = frac_term * e_term * X[i][j] - Y[i]* X[i][j]
                sum_of_terms += term
            avg_term = sum_of_terms/n
            gradient[j] = avg_term

        return gradient

        #raise NotImplementedError

    def train(self, X, Y, eta, rho):
        """
        Train the model with gradient descent.
        Update self.w with the algorithm listed in the problem.
        Returns: Nothing.
        """
        while True:
            gradient = self.compute_grad(X,Y)
            some_sum = 0.0
            for wj in gradient:
                some_sum += wj*wj
            magnitude_of_gradient = math.sqrt(some_sum)
            if magnitude_of_gradient < rho:
                break
            else:
                for j in range(gradient.size):
                    self.w[j] -= gradient[j]



        #raise NotImplementedError


if __name__ == '__main__':
    # Sample Input/Output
    d = 10
    n = 1000

    np.random.seed(0)
    X = np.random.randn(n, d)
    Y = np.array([0] * (n // 2) + [1] * (n // 2))
    eta = 1e-3
    rho = 1e-6

    reg = LogisticRegression(d)
    reg.train(X, Y, eta, rho)
    print(reg.w)
