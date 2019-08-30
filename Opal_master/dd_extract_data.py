#!/usr/bin/env python

import vtk, vtktools
import numpy as np
import sys,os

from importance_map_reduced import get_node_coords_from_vtk, get_field_from_vtk

# generate a [64*64, 3] matrix and do interpolation
def find_coords_for_matrix(start_coords_x, end_coords_x, start_coords_y, end_coords_y, split_nodes):
    index = []
    for j in range(split_nodes):
        for i in range(split_nodes):
            index.append([start_coords_x + float(i)/(split_nodes-1)*(end_coords_x-start_coords_x),
                        start_coords_y + float(j)/(split_nodes-1)*(end_coords_y-start_coords_y),
                        0])
    return np.array(index)

def interpolation_from_vtu_file_split2(filename="Flowpast_2d_Re3900_200.vtu", fieldname="Velocity"):

    left_split_nodes = 48
    right_split_nodes = 32

    # get the volume_fraction field from the results file called slug_214.vtu
    volume_fraction = get_field_from_vtk(filename, fieldname)
    print "shape of volume_fraction array", (volume_fraction[0].shape)
    print "maximum value of volume fraction", np.max(volume_fraction[0])
    print "minimum value of volume fraction", np.min(volume_fraction[0])

    coords = get_node_coords_from_vtk(filename)
    # find the bounding box of the cylinder
    print "x in", np.min(coords[:,0]), np.max(coords[:,0])
    print "y in", np.min(coords[:,1]), np.max(coords[:,1])
    print "z in", np.min(coords[:,2]), np.max(coords[:,2])

    # find out what the value of volume fraction is at this point
    vtu_data = vtktools.vtu(filename)

    index_left = find_coords_for_matrix(start_coords_x=np.min(coords[:,0]),
                                    end_coords_x=(np.min(coords[:,0])+np.max(coords[:,0]))/2,
                                    start_coords_y=np.min(coords[:,1]),
                                    end_coords_y=np.max(coords[:,1]),
                                    split_nodes=left_split_nodes)

    index_right = find_coords_for_matrix(start_coords_x=(np.min(coords[:,0])+np.max(coords[:,0]))/2,
                                end_coords_x=np.max(coords[:,0]),
                                start_coords_y=np.min(coords[:,1]),
                                end_coords_y=np.max(coords[:,1]),
                                split_nodes=right_split_nodes)

    probe_left = vtktools.VTU_Probe(vtu_data.ugrid, index_left)
    probe_right = vtktools.VTU_Probe(vtu_data.ugrid, index_right)

    solution_at_index_left = probe_left.GetField(fieldname)
    solution_at_index_right = probe_right.GetField(fieldname)


    solution_at_index_left = solution_at_index_left.reshape(left_split_nodes,left_split_nodes,3)
    solution_at_index_right = solution_at_index_right.reshape(right_split_nodes,right_split_nodes,3)

    solution_at_index_left = np.delete(solution_at_index_left, 2, 2)
    solution_at_index_right = np.delete(solution_at_index_right, 2, 2)

    return solution_at_index_left, solution_at_index_right

def generate_all_matrix_dd(range_vtu=2000, start_index=0, file_prefix="snapshots/Flowpast_2d_Re3900_"):
    interpolated_matrix_left = []
    interpolated_matrix_right = []
    for i in range(range_vtu):
        filename = file_prefix + str(i) + ".vtu"
        solution_at_index_left, solution_at_index_right = interpolation_from_vtu_file_split2(filename=filename)
        interpolated_matrix_left.append(solution_at_index_left.tolist())
        interpolated_matrix_right.append(solution_at_index_right.tolist())

    np.save('interpolated_matrix_left', interpolated_matrix_left)
    np.save('interpolated_matrix_right', interpolated_matrix_right)



# generate_all_matrix_dd()
