import numpy as np

import vtk, vtktools
import sys,os

from importance_map_reduced import *

import matplotlib.pyplot as plt

def read_coords_and_result_from_vtu(file_range, file_prefix="snapshots/Flowpast_2d_Re3900_",
                                    field_name="Velocity"):
    values_all = []
    filename = file_prefix + str(0) + ".vtu"
    coords = get_node_coords_from_vtk(filename)
    for i in range(file_range):
        filename = file_prefix + str(i) + ".vtu"
        vtu_data = vtktools.vtu(filename)
        values_all.append(vtu_data.GetField(field_name))
    return np.array(values_all), np.array(coords)

def probe_from_vtu(coords_all,
                   factor,
                   file_prefix="snapshots/Flowpast_2d_Re3900_",
                   field_name="Velocity",
                   file_range=100):
    probe_all = []
    for i in range(file_range):
        filename = file_prefix + str(factor*i) + ".vtu"
        vtu_data = vtktools.vtu(filename)
        probe = vtktools.VTU_Probe(vtu_data.ugrid, coords_all)
        solution_at_index = probe.GetField(field_name)
        probe_all.append(solution_at_index)
    return np.array(probe_all).transpose((1,2,0))


def compare_vtu(file_prefix_1 = "snapshots/Flowpast_2d_Re3900_",
                range_1 = 2000, file_prefix_2="dd4_1/nirom_prediction_",
                range_2=100,
                x_coords=1,
                y_coords=1):

    values_1, coords_1 = read_coords_and_result_from_vtu(file_prefix=file_prefix_1, file_range=range_1)
    values_2, coords_2 = read_coords_and_result_from_vtu(file_prefix=file_prefix_2, file_range=range_2)


    # only take 1/20 of values_1 as input
    values = []
    for i in range(0, values_1.shape[0], 20):
        values.append(values_1[i])
    values_1 = np.array(values)

    # reduce the 3rd column
    values_1 = np.delete(values_1, 2, axis=2)
    values_2 = np.delete(values_2, 2, axis=2)

    print(values_1.shape)
    print(values_2.shape)

    assert values_1.shape[0] == values_2.shape[0]

    subtracted = np.subtract(values_1, values_2)
    print(subtracted)

    norm2=[]
    norm_max=[]
    for i in range(values_1.shape[0]):
        norm2.append(np.linalg.norm(subtracted[i]))
        # norm_max = np.linalg.norm(np.subtract(values_1, values_2), ord = 3)
        norm_max.append(np.max(np.sum(np.abs(subtracted[i]), axis=1)))

    norm2=np.array(norm2)
    norm_max=np.array(norm_max)
    print(norm2)
    print(norm_max)

    coords_all = np.array([[0.205, 0.3, 0],[3, 3, 0],[3, 3, 0]]) #[0.3, 0.3, 0],[0.1, 0.3, 0]])

    # factor means how many vtu files needs to be skiped
    probe_1 = probe_from_vtu(coords_all, factor=20, file_prefix=file_prefix_1)
    probe_2 = probe_from_vtu(coords_all, factor=1, file_prefix=file_prefix_2)

    print(probe_1.shape)
    print(probe_2.shape)

    print(coords_1.shape)
    print(np.min(coords_1[:,0]), np.max(coords_1[:,0]))
    print(np.min(coords_1[:,1]), np.max(coords_1[:,1]))

    for i in range(probe_1.shape[0]):
        plt.figure(i+1)
        plt.subplot(2,1,1)
        plt.plot(probe_1[i, 0, 1:])
        plt.subplot(2,1,2)
        plt.plot(probe_2[i, 0, 1:])

    for i in range(probe_1.shape[0]):
        plt.figure(i+4)
        plt.subplot(2,1,1)
        plt.plot(probe_1[i, 1, 1:])
        plt.subplot(2,1,2)
        plt.plot(probe_2[i, 1, 1:])

    plt.figure(7)
    plt.plot(norm2[1:])
    plt.figure(8)
    plt.plot(norm_max[1:])

    plt.show()


compare_vtu()
