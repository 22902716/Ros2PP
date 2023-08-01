import matplotlib.pyplot as plt
import numpy as np

def main():
    mapname = 'f1_aut_wide_raceline'
    iteration_no = 1
    TESTMODE = "Benchmark"
    Max_iter = 5
    raceline = np.loadtxt('src/my_waypoint_follower/my_waypoint_follower/' + mapname+'.csv',delimiter=',')
    # raceline_float = raceline_str.astype(np.float)
    while iteration_no <= Max_iter:
        data = np.loadtxt('src/my_waypoint_follower/my_waypoint_follower/csv/'+mapname+'/'+TESTMODE+"/lap"+str(iteration_no)+'.csv',delimiter=',',skiprows=2)
        # print(data)
        plt.figure()
        plt.plot(data[:,0],data[:,1],label = "Actual paths",marker='x',markersize = 1)
        plt.plot(raceline[:,1],raceline[:,2],label = "Desired paths")
        plt.legend()
        plt.savefig(f"src/my_waypoint_follower/my_waypoint_follower/csv/{mapname}/{TESTMODE}/graph/Trajectory_lap_{str(iteration_no)}.svg")
        plt.figure()
        plt.plot(data[:,5],data[:,3],label = "Speed Profile")
        plt.plot(data[:,5],data[:,4],label = "Acttual speed")
        plt.legend()
        plt.savefig(f"src/my_waypoint_follower/my_waypoint_follower/csv/{mapname}/{TESTMODE}/graph/Speed_lap_{str(iteration_no)}.svg")
        iteration_no += 1

    for i in range(0,Max_iter):
        pass
        


if __name__ == "__main__":
    main()