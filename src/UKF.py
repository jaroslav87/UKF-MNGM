import numpy as np
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from src.MNGM2 import MNGM2

class UKF:

    def __init__(self, n, m):

        self.n = n
        self.m = m

        # UKF params
        self.kappa = 4.0
        self.alfa = 1.0
        self.beta = 0.0
        self.lambda_ = (self.n + self.kappa) * self.alfa * self.alfa - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        self.W0m = self.lambda_ / (self.n + self.lambda_)
        self.W0c = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alfa * self.alfa + self.beta);
        self.W = 1.0 / (2.0 * (self.n + self.lambda_))

        # all vectors used in the UKF process
        self.x_apriori = np.zeros((self.n, 1), dtype=float)
        self.x_aposteriori = np.zeros((self.n, 1), dtype=float)
        self.x_P = np.zeros((self.n, 1), dtype=float)
        self.y_P = np.zeros((self.m, 1), dtype=float)
        self.in_ = np.zeros((self.m, 1), dtype=float)
        self.y = np.zeros((self.m, 1), dtype=float)

        # covarince matrices used in the process

        self.P_apriori = np.zeros((self.n, self.n), dtype=float)
        self.P_aprioriP = np.zeros((self.n, self.n), dtype=float)

        self.P_aposteriori = np.zeros((self.n, self.n), dtype=float)

        # square root product of a given covariances s
        self.sP_apriori = np.zeros((self.n, self.n), dtype=float)
        self.sP_aposteriori = np.zeros((self.n, self.n), dtype=float)

        # clear sigma points
        self.y_sigma = np.zeros((self.m, (2 * self.n + 1)), dtype=float)
        self.x_sigma = np.zeros((self.n, (2 * self.n + 1)), dtype=float)

        # sigma points after passing through the function f/h
        self.x_sigma_f = np.zeros((self.n, (2 * self.n + 1)), dtype=float)

        # cross covariances
        self.P_xy = np.zeros((self.n, self.m), dtype=float)
        self.P_xyP = np.zeros((self.n, self.m), dtype=float)

        self.P_y = np.zeros((self.m, self.m), dtype=float)
        self.oP_y = np.zeros((self.m, self.m), dtype=float)
        self.P_y_P = np.zeros((self.m, self.m), dtype=float)
        self.K = np.zeros((self.n, self.m), dtype=float)
        self.K_0 = np.zeros((self.n, self.m), dtype=float)
        self.K_UKF_T = np.zeros((self.m, self.n), dtype=float)

        self.Q = np.zeros((self.n, self.n), dtype=float)
        self.R = np.zeros((self.m, self.m), dtype=float)

        self.Rs = 0
        self.Qs = 0

        self.mngm = 0



    def resetUKF(self, _Q, _R, x_0):
        # Q - filter process noise covraiance
        # R - measurement noise covariance,
        # P - init covariance noise

        self.mngm = MNGM2(self.n, x_0)
        # init of all vectors and matrices where the first dim := n
        for i in range(self.m):
            self.y[i, 0] = 0
            self.y_P[i, 0] = 0

            for j in range(2 * self.n + 1):
                self.y_sigma[i, j] = 0

            for j in range(self.m):
                self.P_y[i, j] = 0
                self.oP_y[i, j] = 0
                self.P_y_P[i, j] = 0

        # init of all vectors and matrices where the first dim := n_UKF
        for i in range(0, self.n):
            self.x_apriori[i, 0] = x_0[i, 0]
            self.x_aposteriori[i, 0] = x_0[i, 0]
            self.x_P[i, 0] = 0

            for j in range(2 * self.n + 1):
                self.x_sigma[i, j] = 0
                self.x_sigma_f[i, j] = 0

            for j in range(self.m):
                self.P_xy[i, j] = 0
                self.P_xyP[i, j] = 0
                self.K[i, j] = 0
                self.K_0[i, j] = 0

            for j in range(self.n):
                self.P_apriori[i, j] = 0
                self.P_aposteriori[i, j] = 0

                if i == j:
                    self.P_apriori[i, j] = _Q
                    self.P_aposteriori[i, j] = _Q

        self.setCovariances(_Q, _R)

    def setCovariances(self, _Q, _R):
        for i in range(self.n):
            for j in range(self.n):
                self.Q[i, j] = 0
                if i == j:
                    self.Q[i, j] = _Q

        for i in range(self.m):
            for j in range(self.m):
                self.R[i, j] = 0
                if i == j:
                    self.R[i, j] = _R

    def sigma_points(self, vect_X, matrix_S):
        # vect_X - state vector
        # sigma points are drawn from P
        for i in range(self.n):
            self.x_sigma[i, 0] = vect_X[i, 0]  # the first column
        for k in range(1, self.n + 1):
            for i in range(self.n):
                self.x_sigma[i, k] = vect_X[i, 0] + self.gamma * matrix_S[i, k - 1]
                self.x_sigma[i, self.n + k] = vect_X[i, 0] - self.gamma * matrix_S[i, k - 1]


    def y_UKF_calc(self):
        # finding the y = h(x, ...)
        # the input is x_sigma, which is using h(...) then we find y_sigma_UKF from which we get to the y_UKF
        for k in range(2 * self.n + 1):
            #for i in range(self.m):
            xi = self.x_sigma[:, k]
            self.y_sigma[:, k] = self.mngm.output(xi) #xi ** 2) / 20.0

        # y_UKF
        for i in range(self.m):
            self.y[i, 0] = self.W0m * self.y_sigma[i, 0]


        for k in range(1, 2 * self.n + 1):
            for i in range(0, self.n):
                self.y[i, 0] = self.y[i, 0] + self.W * self.y_sigma[i, k]


    def state(self, w):
        # w - input vector data,
        for j in range(2 * self.n + 1):
            xp = self.x_sigma[:, j]
            self.x_sigma_f[:, j] = self.mngm.state(w[0, 0], xp)#0.5*xp + 25.0*(xp/(1.0 + xp**2)) + 8.0*np.cos(1.2*(w[0, 0] - 1.0))


    def squareRoot(self, in_):
        out_ = scipy.linalg.cholesky(in_, lower=False)
        return out_

    def setZero(self, n_, m_, M):
        for i in range(n_):
            for j in range(m_):
                M[i, j] = 0

    def timeUpdate(self, w):

        self.sP_aposteriori = self.squareRoot(self.P_aposteriori)

        self.sigma_points(self.x_aposteriori, self.sP_aposteriori)

        self.state(w)

        # apriori state:
        for i in range(self.n):
            self.x_apriori[i, 0] = self.W0m * self.x_sigma_f[i, 0]
            for k in range(1, 2 * self.n + 1):
                self.x_apriori[i, 0] = self.x_apriori[i, 0] + self.W * self.x_sigma_f[i, k]


        #apriori covariance matrix:
        self.setZero(self.n, self.n, self.P_apriori)

        for k in range(2 * self.n + 1):
            #for i in range(self.n):
            self.x_P[:, 0] = self.x_sigma_f[:, k]

            self.x_P = self.x_P - self.x_apriori
            self.P_aprioriP = np.matmul(self.x_P, np.transpose(self.x_P))

            if k == 0:
                self.P_aprioriP = np.multiply(self.W0c, self.P_aprioriP)
            else:
                self.P_aprioriP = np.multiply(self.W, self.P_aprioriP)
            self.P_apriori = self.P_apriori + self.P_aprioriP

        self.P_apriori = self.P_apriori + self.Q

        self.sP_apriori = self.squareRoot(self.P_apriori)

        self.sigma_points(self.x_apriori, self.sP_apriori)

        self.y_UKF_calc()


    def measurementUpdate(self, mes):
        # cov matrix oytpu/output
        self.setZero(self.m, self.m, self.P_y)

        for k in range(2 * self.n + 1):
            #for i in range(self.n):
            self.y_P[:, 0] = self.y_sigma[:, k]

            self.y_P = self.y_P - self.y
            self.P_y_P = np.matmul(self.y_P, np.transpose(self.y_P))

            if k == 0:
                self.P_y_P = np.multiply(self.W0c, self.P_y_P)
            else:
                self.P_y_P = np.multiply(self.W, self.P_y_P)
            self.P_y = self.P_y + self.P_y_P

        self.P_y = self.P_y + self.R


        # cross cov matrix input/output:
        self.setZero(self.n, self.m, self.P_xy)

        for k in range(2 * self.n + 1):
            self.x_P[:, 0] = self.x_sigma_f[:, k]
            self.y_P[:, 0] = self.y_sigma[:, k]

            self.x_P = self.x_P - self.x_apriori
            self.y_P = self.y_P - self.y
            self.P_xyP = np.matmul(self.x_P, np.transpose(self.y_P))

            if k == 0:
                self.P_xyP = np.multiply(self.W0c, self.P_xyP)
            else:
                self.P_xyP = np.multiply(self.W, self.P_xyP)
            self.P_xy = self.P_xy + self.P_xyP

        # kalman gain:
        self.K = np.matmul(self.P_xy, np.linalg.inv(self.P_y))

        # aposteriori state:
        self.in_[:, 0] = mes[:, 0]
        self.y_P = self.in_ - self.y
        self.x_aposteriori = self.x_apriori + np.matmul(self.K, self.y_P)


        # cov aposteriori:
        self.P_aposteriori = self.P_apriori - np.matmul(np.matmul(self.K, self.P_y), np.transpose(self.K))

