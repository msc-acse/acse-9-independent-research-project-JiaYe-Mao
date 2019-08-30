#!/usr/bin/env python

import vtk, vtktools
import numpy as np
import sys,os

from importance_map_reduced import get_node_coords_from_vtk, get_field_from_vtk

def interpolation_from_vtu_file(filename="Flowpast_2d_Re3900_200.vtu", fieldname="Velocity"):

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

    # generate a [64*64, 3] matrix and do interpolation
    index = []

    for i in range(64):
        for j in range(64):
            index.append([float(i)/63*np.max(coords[:,0]), float(j)/63*np.max(coords[:,1]), 0])
    index = np.array(index)

    probe = vtktools.VTU_Probe(vtu_data.ugrid, index)
    solution_at_index = probe.GetField(fieldname)
    print "at point", index, "..."
    print "point shape", index.shape
    print "...the volume fraction is ", solution_at_index
    print "...the shape of volume fraction is ", solution_at_index.shape

    solution_at_index = solution_at_index.reshape(64,64,3)
    solution_at_index = np.delete(solution_at_index, 2, 2)
    print "the shape after reshape", solution_at_index.shape
    print "the volume fraction after reshape", solution_at_index
    return solution_at_index

def generate_all_matrix(range_vtu=2000):
    interpolated_matrix = []
    for i in range(range_vtu):
        filename = "Flowpast_2d_Re3900_" + str(i) + ".vtu"
        interpolated_matrix.append(interpolation_from_vtu_file(filename=filename).tolist())
    np.save('interpolated_matrix', interpolated_matrix)

generate_all_matrix()
