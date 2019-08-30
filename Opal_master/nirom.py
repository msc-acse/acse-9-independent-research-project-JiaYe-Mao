#!/usr/bin/env python

#    Copyright (C) 2017 Imperial College London and others.
#
#    Please see the AUTHORS file in the main source directory for a full list
#    of copyright holders.
#
#    Prof. C Pain
#    Applied Modelling and Computation Group
#    Department of Earth Science and Engineering
#    Imperial College London
#
#    amcgsoftware@imperial.ac.uk
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation,
#    version 2.1 of the License.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
#    USA

#Authors: Claire Heaney, Pablo Salinas, Dunhui Xiao, Christopher Pain

import numpy as np
import sys
from nirom_tools import *
from opal_classes import Timings, Compression
#from autoencoder import get_pretrained_model

from keras.models import load_model

np.random.seed(1337)  # for reproducibility

def nirom(fwd_options,nirom_options):

# temporarily checking options
#    print "svd            ", nirom_options.compression.svd
#    print "eigh           ", nirom_options.compression.eigh
#    print "autoencoder    ", nirom_options.compression.autoencoder
#    print "svd_autoencoder", nirom_options.compression.svd_autoencoder
#    print "fields         ", nirom_options.compression.field
#    print "epochs         ", nirom_options.compression.epochs
#    print "batch_size     ", nirom_options.compression.batch_size
#    print "nPod           ", nirom_options.compression.nPOD
#    print "cum_tol        ", nirom_options.compression.cumulative_tol
#    print "neurons        ", nirom_options.compression.neurons_each_layer

#    print "LSTM                ", nirom_options.training.LSTM
#    print "GPR                 ", nirom_options.training.GPR
#    print "GPR_scaling         ", nirom_options.training.GPR_scaling
#    print "RBF_length_scale    ", nirom_options.training.GPR_RBF_length_scale
#    print "RBF_length_bounds   ", nirom_options.training.GPR_RBF_length_bounds
#    print "RBF_constant_value  ", nirom_options.training.GPR_constant_value
#    print "RBF_constant_bounds ", nirom_options.training.GPR_constant_bounds

#    sys.exit()

    dd=True
    nsplt=0        # 2**ndomain = ndomain
    generate_directory='dd0_1_test'
    epochs = 200
    dim=2
    nLatent=32
    # file_prefix = "/data/LSBU_results/LSBUv2_"
    file_prefix = "snapshots/Flowpast_2d_Re3900_"
    interpolate_method = 1

    structured_shape = None
    min_all = None
    max_all = None
    witchd = None



    # for reading in snapshots / compressing and training
    if nirom_options.prediction.nTime == 0:

        timings = Timings()

        # generate snapshots if needs be ----------------------------------------------------
        if nirom_options.snapshots.create:
            print 'Sorry - you can only point at pre-generated snapshots at the moment'
            print 'Run the simulations and then point to where the snapshots are located'
            print 'exiting'
            sys.exit(0)

        # get some settings in order to read in the snapshots
        fwd_options, nirom_options  = get_nirom_settings(fwd_options, nirom_options)

        dd=True
        nsplt=nirom_options.compression.nsplt        # 2**ndomain = ndomain
        generate_directory= nirom_options.compression.generate_directory
        epochs = nirom_options.compression.number_epochs
        dim= nirom_options.compression.dim
        nLatent=nirom_options.compression.nLatent
        # file_prefix = "/data/LSBU_results/LSBUv2_"
        file_prefix = fwd_options.path_to_results + "/" + nirom_options.compression.file_prefix
        interpolate_method = nirom_options.compression.method

        print("nsplt", nsplt)
        print("generate_directory", generate_directory)
        print("epochs", epochs)
        print("dim", dim)
        print("nLatent", nLatent)
        print("file_prefix", file_prefix)


        # (1) read in snapshots -----------------------------------------------------------------
        # nirom_options, t = read_in_snapshots(fwd_options, nirom_options)
        t=0
        timings.read_in_snapshots = t
        print "time for snapshots",t

        # (2) compression ----------------------------------------------------------------------
        if not nirom_options.compression.autoencoder:

            nirom_options, t,  U_encoded = compress_snapshots(fwd_options, nirom_options)
            nirom_options.compression.write_sing_values()

            timings.compression = t
        else:
            nirom_options,  U_encoded, structured_shape, min_all, max_all, witchd = compress_snapshots_real_dd(interpolate_method, file_prefix, nLatent, dim, epochs, generate_directory, nsplt, fwd_options, nirom_options) # compress_snapshots_autoencoder(fwd_options, nirom_options)

            timings.compression = 2 # Todo : to be completed

        # else read in basis functions

        # map snapshots to reduced space ready for the training -----------------------------
        # do this here and write to file

        # (3) training of the GPR ---------------------------------------------------------------
        if nirom_options.training.GPR :
            try:
                preds = [np.load(generate_directory+"/output_" + str(i)+".npy") for i in range(2**nsplt)]
                output_vtu_dd(preds, file_prefix, dim, witchd, structured_shape, min_all, max_all, directory=generate_directory, file_prefix="nirom_prediction_")
                t_train=0
                t_pred=0
            except IOError:
                t_train, t_pred = train_the_NIROM(nLatent, generate_directory, file_prefix, dim, generate_directory, witchd, structured_shape, min_all, max_all, nirom_options, fwd_options, U_encoded, dd=dd)





            timings.training = t_train
            timings.replication = t_pred
        if nirom_options.compression.svd:
            svd_eigh = 'svd'
        else:
            svd_eigh = 'eigh'
        timings.print_values(svd_eigh)
        timings.write_values('nirom_timings.dat',svd_eigh)

    # (4) for predicting forward in time from the final snapshot
    elif nirom_options.prediction.nTime > 0:

        # predicting unseen behaviour with the GPR ---------------------------------------------------------------

        #if nirom_options.compression.svd_autoencoder:
        #    # autoencoder / prediction for BE case
        #    t1, t2 = predict_with_the_NIROM(nirom_options, fwd_options)
        #else:
        #    # GB NIROM advection case
        #    t1, t2 = predict_with_the_NIROM_2D_advection_GBN(nirom_options, fwd_options)
        tload, tpred = predict_with_the_NIROM(nirom_options, fwd_options)

        print "time to load neural networks (.sav files)", tload
        print "time to predict", tpred
        timings.prediction = tpred


    return
