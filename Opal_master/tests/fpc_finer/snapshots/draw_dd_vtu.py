import numpy as np
import sys
import math

# np.set_printoptions(threshold=sys.maxsize)


from keras.models import Model
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Concatenate, Lambda

from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from keras.layers.core import Flatten, Reshape



import decomp_new
import decomp_dd

from keras.models import load_model

import vtk, vtktools
import sys,os

from importance_map_reduced import *

def output_vtu_dd_gragh(nsplt, dim, directory, file_prefix="dd_graph",  file_prefix_snapshots="Flowpast_2d_Re3900_5.vtu"):
    filename = file_prefix_snapshots+str(0)+".vtu"

    vtu_data = vtktools.vtu(filename)
    coordinates = vtu_data.GetLocations()
    findm,colm,ncolm = decomp_new.fsmfp(coordinates, coordinates.shape[0])

    wnod = decomp_new.weight_calc_non_uni_meshes( findm,colm, coordinates, ncolm, coordinates.shape[0] )

    split_levels = np.zeros((nsplt),dtype='int32')
    split_levels[:] = 2
    havwnod = 2
    havmat = 0
    a = np.zeros(1)
    exact = True
    iexact = 1
    ii=1
    na=0

    witchd = decomp_dd.python_set_up_recbis(split_levels,findm,colm, wnod,a, havwnod,havmat,iexact, nsplt,ncolm,coordinates.shape[0],na)


    clean_vtu = get_clean_vtk_file(filename)
    string = "mkdir " + directory
    os.system(string)

    coords_pred = []
    for j in range(coordinates.shape[0]):
        # domains
        if dim==2:
            x1 = witchd[j]
            x2 = witchd[j]
            coords_pred.append(np.array((x1, x2)))
        if dim==3:
            x1 = witchd[j]
            x2 = witchd[j]
            x3 = witchd[j]
            coords_pred.append(np.array((x1, x2, x3)))
    coords_pred = np.array(coords_pred)

    new_vtu = vtktools.vtu()
    new_vtu.ugrid.DeepCopy(clean_vtu.ugrid)

    new_vtu.filename = directory + "/" + file_prefix + "_" + str(2**nsplt) + ".vtu"

    preds_vals = np.zeros((coords_pred.shape[0], coords_pred.shape[1]+1))
    preds_vals[:,:-1] = coords_pred
    print(preds_vals.shape)
    new_vtu.AddField('Velocity',preds_vals)
    new_vtu.Write()


output_vtu_dd_gragh(nsplt=3, dim=2, directory="domains_draw", file_prefix="dd_graph",  file_prefix_snapshots="Flowpast_2d_Re3900_")
