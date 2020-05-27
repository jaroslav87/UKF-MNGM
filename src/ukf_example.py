import os.path
from src.MNGM2 import MNGM2
from src.UKF import UKF
import numpy as np
from matplotlib import pyplot as plt


def estimateState():
    n = 2 #size of the state vector
    m = 2 #size of the output vector

    #initial x value
    x_0 = np.zeros((n, 1))
    x_0[0, 0] = 0.1
    x_0[1, 0] = 0.1

    mngm = MNGM2(500, x_0)
    mngm.generate_data()

    ukf = UKF(n, m)

    #generated data:
    dataX = mngm.x
    dataY = mngm.y

    size_n = dataX.shape[0]


    ukf.resetUKF(0.1,  0.001, x_0)

    timeUpdateInput = np.zeros((n, 1))
    measurementUpdateInput = np.zeros((m, 1))

    err_total = 0

    est_state = np.zeros((size_n, n))
    est_output = np.zeros((size_n, n))

    # estimation loop
    for i in range(size_n):

        timeUpdateInput[0, 0] = i
        measurementUpdateInput[:, 0] = dataY[i, :]

        # recursively go through time update and measurement correction
        ukf.timeUpdate(timeUpdateInput)
        ukf.measurementUpdate(measurementUpdateInput)

        err = 0
        for j in range(n):
            err = err + (ukf.x_aposteriori[j, 0] - dataX[i, j])**2

        est_state[i, 0] = ukf.x_aposteriori[0, 0]
        est_state[i, 1] = ukf.x_aposteriori[1, 0]
        est_output[i, 0] = ukf.y[0, 0]
        est_output[i, 1] = ukf.y[1, 0]

        est_output[i, 0] = ukf.K[0, 1]

        err_total = err_total + err

        #print(err)

    print("total error:", err_total)

    plt.plot(dataX[:, 0], 'g', label='x_1 original')  # X from the orginal ungm
    plt.plot(dataX[:, 1], 'b', label='x_2 original')  # X from the orginal ungm
    plt.plot(est_state[:, 0], 'r-', label='x_1 estimated') #estimated X
    plt.plot(est_state[:, 1], 'k-', label='x_2 estimated')  # estimated X
    plt.legend(loc='upper right')

    #plt.plot(dataY[:, 1], 'g')  # Y from the orginal ungm
    #plt.plot(est_output[:, 1], 'b')  # estimated Y

    #plt.plot(est_output[:, 0], 'b')  # estimated Y

    plt.show()


estimateState()
