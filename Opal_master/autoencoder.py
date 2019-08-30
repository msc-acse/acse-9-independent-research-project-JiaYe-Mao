import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib.pyplot as plt
from keras.models import load_model

import vtk, vtktools
import sys,os

from importance_map_reduced import *

def interpolation_from_vtu_file(filename="snapshots/Flowpast_2d_Re3900_200.vtu", fieldname="Velocity"):

    # get the volume_fraction field from the results file called slug_214.vtu
    volume_fraction = get_field_from_vtk(filename, fieldname)
    #print "shape of volume_fraction array", (volume_fraction[0].shape)
    #print "maximum value of volume fraction", np.max(volume_fraction[0])
    #print "minimum value of volume fraction", np.min(volume_fraction[0])

    coords = get_node_coords_from_vtk(filename)
    # find the bounding box of the cylinder
    #print "x in", np.min(coords[:,0]), np.max(coords[:,0])
    #print "y in", np.min(coords[:,1]), np.max(coords[:,1])
    #print "z in", np.min(coords[:,2]), np.max(coords[:,2])

    # find out what the value of volume fraction is at this point
    vtu_data = vtktools.vtu(filename)

    # generate a [64*64, 3] matrix and do interpolation
    index = []

    for j in range(64):
        for i in range(64):
            index.append([float(i)/63*np.max(coords[:,0]), float(j)/63*np.max(coords[:,1]), 0])
    index = np.array(index)

    probe = vtktools.VTU_Probe(vtu_data.ugrid, index)
    solution_at_index = probe.GetField(fieldname)
    #print "at point", index, "..."
    #print "point shape", index.shape
    #print "...the volume fraction is ", solution_at_index
    #print "...the shape of volume fraction is ", solution_at_index.shape

    solution_at_index = solution_at_index.reshape(64,64,3)
    solution_at_index = np.delete(solution_at_index, 2, 2)
    #print "the shape after reshape", solution_at_index.shape
    #print "the volume fraction after reshape", solution_at_index
    return solution_at_index



def generate_all_matrix(range_vtu=2000, start_index=0, file_prefix="Flowpast_2d_Re3900_"):
    interpolated_matrix = []
    for i in range(start_index, range_vtu, 1):
        filename = file_prefix + str(i) + ".vtu"
        interpolated_matrix.append(interpolation_from_vtu_file(filename=filename).tolist())
	#print "i", start_index, "num", i
    return np.array(interpolated_matrix)


def get_original_data_from_vtu(index=19999,):

    filename = "../Flowpast_2d_Re3900_" + str(index) + ".vtu"

    coords = get_node_coords_from_vtk(filename)

    vtu_data = vtktools.vtu(filename)
    velocity = vtu_data.GetField("Velocity")

    print("velocity", velocity)
    print(velocity.shape)

    print(coords.shape)

    max_x = np.max(coords[:,1])
    max_y = np.max(coords[:,0])
    return coords, velocity

def interpolation_back_to_unstructured_mesh(point, msh, dx, dy, start_x, start_y, feature_map):
    print("point", point)
    index_y = (int)((point[1]-start_y) / dy)
    index_x = (int)((point[0]-start_x) / dx)
    print("index:", index_y, index_x)
    if index_x >= msh.shape[1]-1:
        index_x = index_x-1
    if index_y >= msh.shape[0]-1:
        index_y = index_y-1

    wy = (point[1] - start_y - index_y * dy) / dy

    C_left = wy * msh[index_y+1, index_x, feature_map] + (1-wy) * msh[index_y, index_x, feature_map]
    C_right = wy * msh[index_y+1, index_x+1, feature_map] + (1-wy) * msh[index_y, index_x+1, feature_map]
    #print("inter", C_left, C_right)
    wx = (point[0] - start_x - index_x*dx)/dx
    C_interpolation = wx * C_right + (1-wx) * C_left
    return C_interpolation


# def predict_and_output_vtu_dd(data, directory="Result", file_prefix="nirom_prediction_", model_name="decoder_dd.h5", filename='snapshots/Flowpast_2d_Re3900_200.vtu'):
#
#     coords = get_node_coords_from_vtk(filename)
#
#     # only suit for flow past a cylinder problem
#     max_y = np.max(coords[:,1])
#     max_x = np.max(coords[:,0])
#     print(max_x)
#     print(max_y)
#     dx_1 = float(max_x) / 47 /2
#     dy_1 = float(max_y) / 47
#     dx_2 = float(max_x) / 31 /2
#     dy_2 = float(max_y) / 31
#
#     decoder = load_model(model_name)
#     preds = decoder.predict(data) ##### check shape of this
#     preds_left = preds[0]
#     preds_right = preds[1]
#     print "U_decoded shape", preds_left.shape
#
#     np.save("output_left.npy", preds_left)
#     np.save("output_right.npy", preds_right)
#
#     clean_vtu = get_clean_vtk_file(filename)
#     string = "mkdir " + directory
#     os.system(string)
#
#     for i in range(preds_left.shape[0]):
#         coords_pred = []
#         for j in range(coords.shape[0]):
#             if coords[j, 0] < max_x/2:
#                 x1 = interpolation_back_to_unstructured_mesh(coords[j], preds_left[i], dx_1, dy_1, 0, 0, 0)   # use 10*i, since we only generate 1/10 vtu files
#                 x2 = interpolation_back_to_unstructured_mesh(coords[j], preds_left[i], dx_1, dy_1, 0, 0, 1)
#             else:
#                 x1 = interpolation_back_to_unstructured_mesh(coords[j], preds_right[i], dx_2, dy_2, max_x/2, 0, 0)   # use 10*i, since we only generate 1/10 vtu files
#                 x2 = interpolation_back_to_unstructured_mesh(coords[j], preds_right[i], dx_2, dy_2, max_x/2, 0, 1)
#             coords_pred.append(np.array((x1, x2)))
#         coords_pred = np.array(coords_pred)
#
#         new_vtu = vtktools.vtu()
#         new_vtu.ugrid.DeepCopy(clean_vtu.ugrid)
#
#         new_vtu.filename = directory + "/" + file_prefix +str(i)+ ".vtu"
#
#         preds_vals = np.zeros((coords_pred.shape[0], coords_pred.shape[1]+1))
#         preds_vals[:,:-1] = coords_pred
#         print(preds_vals.shape)
#         new_vtu.AddField('Velocity',preds_vals)
#         new_vtu.Write()


def predict_and_output_vtu(data, directory="Result", file_prefix="nirom_prediction_", model_name='decoder.h5', filename='snapshots/Flowpast_2d_Re3900_200.vtu'):

    coords = get_node_coords_from_vtk(filename)
    max_x = np.max(coords[:,1])
    max_y = np.max(coords[:,0])
    dx = max_x / 63
    dy = max_y / 63

    decoder = load_model(model_name)
    preds = decoder.predict(data) ##### check shape of this
    print "U_decoded shape", preds.shape

    clean_vtu = get_clean_vtk_file(filename)
    string = "mkdir " + directory
    os.system(string)

    for i in range(preds.shape[0]):
        coords_pred = []
        for j in range(coords.shape[0]):
            x1 = interpolation_back_to_unstructured_mesh(coords[j], preds[i], dx, dy, 0, 0, 0)   # use 10*i, since we only generate 1/10 vtu files
            x2 = interpolation_back_to_unstructured_mesh(coords[j], preds[i], dx, dy, 0, 0, 1)
            coords_pred.append(np.array((x1, x2)))
        coords_pred = np.array(coords_pred)

        new_vtu = vtktools.vtu()
        new_vtu.ugrid.DeepCopy(clean_vtu.ugrid)

        new_vtu.filename = directory + "/" + file_prefix +str(i)+ ".vtu"

        preds_vals = np.zeros((coords_pred.shape[0], coords_pred.shape[1]+1))
        preds_vals[:,:-1] = coords_pred
        print(preds_vals.shape)
        new_vtu.AddField('Velocity',preds_vals)
        new_vtu.Write()
