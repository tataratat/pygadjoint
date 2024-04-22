from matplotlib import pyplot as plt
import numpy as np

pwd = "/student/19/e11914665/work/pygadjoints/examples/StructuralOptimization/BridgeExample/"
name = "2Dpara_deg_0_n_10_5_r_0"
dof = 54
fig, axs = plt.subplots(3, 2, sharex="col", sharey="row")
fig.suptitle("Statistics of "+name+" (left: c, right: d)")

for i, suf in enumerate(["_c"]):
    iterations = np.genfromtxt(pwd+name+suf+"/log_file_iterations.csv", delimiter=",")
    sensis = np.genfromtxt(pwd+name+suf+"/log_file_sensitivities.csv", delimiter=",")
    volume = np.genfromtxt(pwd+name+suf+"/log_file_volume.csv", delimiter=",")

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