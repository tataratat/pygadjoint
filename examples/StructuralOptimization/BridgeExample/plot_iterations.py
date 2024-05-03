from matplotlib import pyplot as plt
import numpy as np

pwd = "/student/19/e11914665/work/pygadjoints/examples/StructuralOptimization/BridgeExample/"
names = ["2Dpara_test","2Dpara_deg_0_n_10_5_r_0_c"]
dof = 54
fig, axs = plt.subplots(3, 2, sharex="col", sharey="row")
fig.suptitle("Comparison of Convergence\n left: "+names[0]+" right: "+names[1])

for i, name in enumerate(names):
    iterations = np.genfromtxt(pwd+name+"/log_file_iterations.csv", delimiter=",")
    sensis = np.genfromtxt(pwd+name+"/log_file_sensitivities.csv", delimiter=",")
    volume = np.genfromtxt(pwd+name+"/log_file_volume.csv", delimiter=",")

    axs[0, i].plot(iterations[np.int64(sensis[:, 0])-1, 0], iterations[np.int64(sensis[:, 0])-1, 1], "gs")
    axs[0, i].plot(iterations[:, 0], iterations[:, 1],"x-")
    axs[0, i].set_title("Objective Function")

    axs[1, i].plot(volume[np.int64(sensis[:, 0]-1), 0], volume[np.int64(sensis[:, 0])-1, 1], "gs")
    axs[1, i].plot(volume[1:, 0], volume[1:, 1], "x-")
    axs[1, i].set_title("Volume")

    axs[2, i].plot(sensis[:, 0], np.linalg.norm(sensis[:, 1:-dof], axis=1))
    axs[2, i].plot(sensis[:, 0], np.linalg.norm(sensis[:, 1:-dof], axis=1), "gs")
    axs[2, i].set_title("Sensitivities")

plt.show()