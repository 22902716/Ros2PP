from matplotlib import pyplot as plt
import numpy as np

def plot_solution_MPC():
    data = np.loadtxt("csv/MPC_sol_CornerHallE_car_data_0.3.csv",delimiter=',')
    trajectory = np.loadtxt("maps/CornerHallE_raceline.csv",delimiter=',')
    plt.figure()
    for counter in range(0,len(data)):
        print(data[counter, 3:9], data[counter, 9:15])
        plt.plot(trajectory[:,1], trajectory[:,2])
        plt.scatter(data[counter, 3:9], data[counter, 9:15], c='b')
        plt.scatter(data[counter, 27], data[counter, 28])
        plt.scatter(data[counter, 30], data[counter, 31])
        plt.scatter(data[counter, 33], data[counter, 34])
        plt.scatter(data[counter, 36], data[counter, 37])
        plt.scatter(data[counter, 39], data[counter, 40])
        plt.scatter(data[counter, 42], data[counter, 43])

        plt.pause(0.05)
        plt.clf()

def plot_solution_MPCC():
    # data = np.loadtxt("csv/MPCC_sol_CornerHallE_rviz.csv",delimiter=',')
    data = np.loadtxt("csv/MPCC_sol_CornerHallE_car_data.csv",delimiter=',')
    trajectory = np.loadtxt("maps/CornerHallE_centerline.csv",delimiter=',')
    plt.figure()
    for counter in range(0,len(data)):
        plt.plot(trajectory[:,0], trajectory[:,1])
        plt.scatter(data[counter, 3:9], data[counter, 9:15], c='b')

        plt.scatter(data[counter, 27], data[counter, 28], c='r')
        plt.scatter(data[counter, 31], data[counter, 32], c='r')
        plt.scatter(data[counter, 35], data[counter, 36], c='r')
        plt.scatter(data[counter, 39], data[counter, 40], c='r')
        plt.scatter(data[counter, 43], data[counter, 44], c='r')
        plt.scatter(data[counter, 47], data[counter, 48], c='r')

        plt.pause(0.05)
        plt.clf()

        

def main():
    plot_solution_MPC()
    # plot_solution_MPCC()



if __name__ == "__main__":
    main()