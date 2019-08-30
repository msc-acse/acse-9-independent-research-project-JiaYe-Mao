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

class Autoencoder_interpolation():
    def __init__(self, nLatent, img_rows, img_cols, ndomain):
        np.random.seed(1337)  # for reproducibility
        self.nLatent = nLatent
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.size = [int(img_rows[i] / 16 * img_cols[i] / 16 * 8) for i in range(ndomain)]
        self.partition = [0]
        for i in range(ndomain):
            self.partition.append(sum(self.size[:i + 1]))
        print(self.partition)
        self.dd_num = ndomain
        self.channels = 2
        self.img_shape = [None for i in range(self.dd_num)]
        for i in range(self.dd_num):
            self.img_shape[i] = (self.img_rows[i], self.img_cols[i], self.channels)

        self.autoencoder_model, self.encoder_model, self.decoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer='Nadam')
        self.encoder_model.compile(loss='mse', optimizer='Nadam')
        self.decoder_model.compile(loss='mse', optimizer='Nadam')
        self.autoencoder_model.summary()

    def build_model(self):
        input_layer = [Input(shape=self.img_shape[i]) for i in range(self.dd_num)]

        # encoder
        hidden = [None for i in range(self.dd_num)]

        for i in range(self.dd_num):
            hidden[i] = Conv2D(4, (5, 5), strides=(2, 2), activation='elu', padding='same')(input_layer[i])  #
            hidden[i] = Conv2D(8, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2D(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2D(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Flatten()(hidden[i])

        concat_layer = hidden[0]
        if (self.dd_num > 1):
            concat_layer = Concatenate()([hidden[i] for i in range(self.dd_num)])

        h = Dense(2 * self.nLatent, activation='elu')(concat_layer)
        encoded = Dense(self.nLatent, activation='elu')(h)

        # decoder
        h = Dense(2 * self.nLatent, activation='elu')(encoded)
        h = Dense(self.partition[-1], activation='elu')(h)

        output_layer = [None for i in range(self.dd_num)]

        partition = self.partition
        for i in range(self.dd_num):
            a = self.partition[i]
            b = self.partition[i + 1]
            hidden[i] = Lambda(lambda x: x[:, partition[i]:partition[i + 1]])(h)

        for i in range(self.dd_num):
            hidden[i] = Reshape((int(self.img_rows[i] / 16), int(self.img_cols[i] / 16), 8))(hidden[i])

            hidden[i] = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2DTranspose(4, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            output_layer[i] = Conv2DTranspose(2, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])

        autoencoder = Model(inputs=[input_layer[i] for i in range(self.dd_num)],
                            outputs=[output_layer[i] for i in range(self.dd_num)])

        start_index = -int((len(autoencoder.layers) - 1) / 2)
        if self.dd_num == 1:
            start_index -= 1

        decoded_output = [None for i in range(self.dd_num)]

        encoded_input = Input(shape=(self.nLatent,))
        decoded_output_combined = autoencoder.layers[start_index](encoded_input)
        decoded_output_combined = autoencoder.layers[start_index + 1](decoded_output_combined)

        for i in range(self.dd_num):
            a = self.partition[i]
            b = self.partition[i + 1]
            decoded_output[i] = Lambda(lambda x: x[:, partition[i]:partition[i + 1]])(decoded_output_combined)

        start_index = start_index + 2 + self.dd_num
        while (start_index < 0):
            for i in range(self.dd_num):
                decoded_output[i] = autoencoder.layers[start_index](decoded_output[i])
                start_index += 1

        return autoencoder, Model(inputs=[input_layer[i] for i in range(self.dd_num)],
                                  outputs=encoded), \
                            Model(inputs=encoded_input, outputs=[decoded_output[i] for i in range(self.dd_num)])

    def train_model(self, x_train, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1,
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, x_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=None)
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds


class domain():
    def __init__(self, structured_shape, min, max, name, dim):
        self.min=min
        self.max=max
        self.structured_shape=structured_shape
        self.name = name
        if dim==2:
            self.index = self.find_coords_for_matrix_2d(start_coords_x=min[0],
                                    end_coords_x=max[0],
                                    start_coords_y=min[1],
                                    end_coords_y=max[1],
                                    split_nodes_x=self.structured_shape[0],
                                    split_nodes_y=self.structured_shape[1])
        if dim==3:
            self.index = self.find_coords_for_matrix_3d(start_coords_x=min[0],
                                end_coords_x=max[0],
                                start_coords_y=min[1],
                                end_coords_y=max[1],
                                start_coords_z=min[2],
                                end_coords_z=max[2],
                                split_nodes_x=self.structured_shape[0],
                                split_nodes_y=self.structured_shape[1],
                                split_nodes_z=self.structured_shape[2])


    def find_coords_for_matrix_2d(self, start_coords_x, end_coords_x, start_coords_y, end_coords_y, split_nodes_x, split_nodes_y):
        index = []
        for j in range(split_nodes_y):
            for i in range(split_nodes_x):
                index.append([start_coords_x + float(i)/(split_nodes_x-1)*(end_coords_x-start_coords_x),
                            start_coords_y + float(j)/(split_nodes_y-1)*(end_coords_y-start_coords_y),
                            0])

        print("shape index:", np.array(index).shape)
        return np.array(index)

    def find_coords_for_matrix_3d(self, start_coords_x, end_coords_x, start_coords_y, end_coords_y, start_coords_z, end_coords_z, split_nodes_x, split_nodes_y, split_nodes_z):
        index = []
        for k in range(split_nodes_z):
            for j in range(split_nodes_y):
                for i in range(split_nodes_x):
                    index.append([start_coords_x + float(i)/(split_nodes_x-1)*(end_coords_x-start_coords_x),
                                start_coords_y + float(j)/(split_nodes_y-1)*(end_coords_y-start_coords_y),
                                start_coords_z + float(k)/(split_nodes_z-1)*(end_coords_z-start_coords_z)])

        print("shape index:", np.array(index).shape)
        return np.array(index)

    def interpolation_from_vtu_file(self, dim, range_vtu, coords, file_prefix, fieldname="Velocity"):
        solution_at_index_all = []
        for i in range(range_vtu):
            filename = file_prefix + str(i) + ".vtu"
            vtu_data = vtktools.vtu(filename)
            probe = vtktools.VTU_Probe(vtu_data.ugrid, self.index)
            solution_at_index = probe.GetField(fieldname)
            if dim==2:
                solution_at_index = solution_at_index.reshape(self.structured_shape[1],self.structured_shape[0],3)
                solution_at_index = np.delete(solution_at_index, 2, 2)
            if dim==3:
                solution_at_index = solution_at_index.reshape(self.structured_shape[2],self.structured_shape[1],self.structured_shape[0],3)

            solution_at_index_all.append(solution_at_index.tolist())
        return np.array(solution_at_index_all)

def get_coordinates(vtu_filename):
    vtu_data = vtktools.vtu(vtu_filename)
    coordinates = vtu_data.GetLocations()
    return vtu_data,coordinates

def determine_structured_mesh_info(nsplt, min, max, density_ratio, dim=2):
    structured_shape_all = []
    for j in range(2**nsplt):
        length = [max[j][i] - min[j][i] for i in range(dim)]

        length = np.array(length)
        structured_shape = [None for e in range(dim)]
        for i in range(dim):
            if (length[i]==length.min()):
                structured_shape[i]= (int)(density_ratio[j] *2) * 16
            else:
                frac = (int)(float(length[i]) / float(length.min()) * density_ratio[j] *2)
                structured_shape[i]=frac*16
        structured_shape_all.append(structured_shape)
    return np.array(structured_shape_all)

def determine_structured_mesh_info_modified(nsplt, min, max, dim=2):
    density_ratio = [1 for i in range(2**nsplt)]
    structured_shape_all = determine_structured_mesh_info(nsplt, min, max, density_ratio, dim=2)

    area = []
    for j in range(2**nsplt):
        a = 1
        for i in range(dim):
            a *= structured_shape_all[j][i]
        area.append(a)
    area = np.array(area)
    print(area)

    fraction = []
    for i in range(2**nsplt):
        if dim==2:
            fraction.append(math.sqrt(float(area.max())/area[i]))
        elif dim==3:
            fraction.append(pow(float(area.max())/area[i], 1/3))
    print(fraction)
    structured_shape_all = determine_structured_mesh_info(nsplt, min, max, fraction, dim=2)
    return structured_shape_all

def get_witchd(nsplt, coordinates):

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

    print(witchd)
    print(witchd.shape)
    return witchd

def get_domain_info(method, nsplt, witchd, coordinates, dim=2):

    max_all = []
    min_all = []
    for i in range(2**nsplt):
        max=[-99999 for e in range(dim)]
        min=[999999 for e in range(dim)]
        for j in range(witchd.shape[0]):
            if(witchd[j] == i+1):
                for k in range(dim):
                    if(max[k]<coordinates[j, k]):
                        max[k] = coordinates[j, k]
                    if(min[k]>coordinates[j, k]):
                        min[k] = coordinates[j, k]
        max_all.append(max)
        min_all.append(min)

    structured_shape = None
    if method==0:
        density_ratio = [1 for i in range(2**nsplt)]
        structured_shape = determine_structured_mesh_info(nsplt, min_all, max_all, density_ratio, dim=dim)
    elif method==1:
        structured_shape = determine_structured_mesh_info_modified(nsplt, min_all, max_all, dim=dim)
    else:
        print("method not supported, please set method = 0 or 1")
        sys.exit(0)
    return structured_shape, min_all, max_all

def DD_interpolation_from_vtu(directory, nsplt, coords, structured_shape, min_all, max_all, file_prefix, range_vtu=2000, dim=2):

    string = "mkdir " + directory
    os.system(string)

    for i in range(2**nsplt):
        domain_a = domain(structured_shape[i], min_all[i], max_all[i], i+1, dim)
        interpolated_matrix = domain_a.interpolation_from_vtu_file(dim, range_vtu=2000, coords=coords, file_prefix=file_prefix, fieldname="Velocity")
        print(directory+"/interpolated_matrix" + str(i) + " : ", interpolated_matrix.shape)
        np.save(directory+'/interpolated_matrix_'+str(i), interpolated_matrix)

def interpolation_back_to_unstructured_mesh_3d(point, msh, dx, dy, dz, start_x, start_y, start_z, feature_map):
    # print("point", point)
    index_z = (int)((point[2]-start_z) / dz)
    index_y = (int)((point[1]-start_y) / dy)
    index_x = (int)((point[0]-start_x) / dx)
    # print("index:", index_y, index_x)
    if index_z >= msh.shape[2]-1:
        index_z = index_z-1
    if index_x >= msh.shape[1]-1:
        index_x = index_x-1
    if index_y >= msh.shape[0]-1:
        index_y = index_y-1

    wz = (point[2] - start_z - index_z * dz) / dz

    C_up_left = wz * msh[index_z+1, index_y, index_x, feature_map] + (1-wz) * msh[index_z, index_y, index_x, feature_map]
    C_up_right = wz * msh[index_z+1, index_y, index_x+1, feature_map] + (1-wz) * msh[index_z, index_y, index_x+1, feature_map]
    C_down_left = wz * msh[index_z+1, index_y+1, index_x, feature_map] + (1-wz) * msh[index_z, index_y+1, index_x, feature_map]
    C_down_right = wz * msh[index_z+1, index_y+1, index_x+1, feature_map] + (1-wz) * msh[index_z, index_y+1, index_x+1, feature_map]

    wy = (point[1] - start_y - index_y * dy) / dy

    C_left = wy * C_down_left + (1-wy) * C_up_left
    C_right = wy * C_down_right + (1-wy) * C_up_right
    #print("inter", C_left, C_right)
    wx = (point[0] - start_x - index_x*dx)/dx
    C_interpolation = wx * C_right + (1-wx) * C_left
    return C_interpolation

def interpolation_back_to_unstructured_mesh(point, msh, dx, dy, start_x, start_y, feature_map):
    # print("point", point)
    index_y = (int)((point[1]-start_y) / dy)
    index_x = (int)((point[0]-start_x) / dx)
    # print("index:", index_y, index_x)
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

def predict_dd(nLatent, file_prefix_snapshots, generate_directory, structured_shape, encoded):
    filename = file_prefix_snapshots+str(0)+".vtu"
    coords = get_node_coords_from_vtk(filename)
    model_name = generate_directory+"/decoder_dd.h5"
    ae = Autoencoder_interpolation(nLatent, img_rows=[structured_shape[i][1] for i in range(len(structured_shape))], img_cols=[structured_shape[i][0] for i in range(len(structured_shape))], ndomain=len(structured_shape))
    ae.decoder_model.load_weights(model_name, by_name=True)
    decoder = ae.decoder_model
    preds = decoder.predict(encoded) ##### check shape of this


    if len(structured_shape)==1:
        preds=[preds]

    for i in range(len(structured_shape)):
        np.save(generate_directory+"/output_" + str(i)+".npy", preds[i])

    return preds

def output_vtu_dd(preds, file_prefix_snapshots, dim, witchd, structured_shape, min_all, max_all, directory, file_prefix="nirom_prediction_"):

    filename = file_prefix_snapshots+str(0)+".vtu"
    coords = get_node_coords_from_vtk(filename)

    clean_vtu = get_clean_vtk_file(filename)
    string = "mkdir " + directory
    os.system(string)

    print("structured_shape", structured_shape)
    print("min_all", min_all)
    print("max_all", max_all)

    interval = []

    if dim==2:
        for i in range(len(structured_shape)):
            dx = float(max_all[i][0]-min_all[i][0]) / (structured_shape[i][0]-1)
            dy = float(max_all[i][1]-min_all[i][1]) / (structured_shape[i][1]-1)
            interval.append([dx,dy])
    elif dim==3:
        for i in range(len(structured_shape)):
            dx = float(max_all[i][0]-min_all[i][0]) / (structured_shape[i][0]-1)
            dy = float(max_all[i][1]-min_all[i][1]) / (structured_shape[i][1]-1)
            dz = float(max_all[i][2]-min_all[i][2]) / (structured_shape[i][2]-1)
            interval.append([dx,dy,dz])

    print(interval)

    print(preds)
    # snapshots
    for i in range(preds[0].shape[0]):
        coords_pred = []
        # nodes
        for j in range(coords.shape[0]):
            # domains
            if dim==2:

                for k in range(len(structured_shape)):
                    if k+1 == witchd[j]:
                        x1 = interpolation_back_to_unstructured_mesh(coords[j], preds[k][i], interval[k][0], interval[k][1], min_all[k][0], min_all[k][1], 0)
                        x2 = interpolation_back_to_unstructured_mesh(coords[j], preds[k][i], interval[k][0], interval[k][1], min_all[k][0], min_all[k][1], 1)
                coords_pred.append(np.array((x1, x2)))
            if dim==3:
                for k in range(len(structured_shape)):
                    if k+1 == witchd[j]:
                        x1 = interpolation_back_to_unstructured_mesh(coords[j], preds[k][i], interval[k][0], interval[k][1], interval[k][2], min_all[k][0], min_all[k][1], min_all[k][2], 0)
                        x2 = interpolation_back_to_unstructured_mesh(coords[j], preds[k][i], interval[k][0], interval[k][1], interval[k][2], min_all[k][0], min_all[k][1], min_all[k][2], 1)
                        x3 = interpolation_back_to_unstructured_mesh(coords[j], preds[k][i], interval[k][0], interval[k][1], interval[k][2], min_all[k][0], min_all[k][1], min_all[k][2], 2)
                coords_pred.append(np.array((x1, x2, x3)))
        coords_pred = np.array(coords_pred)

        new_vtu = vtktools.vtu()
        new_vtu.ugrid.DeepCopy(clean_vtu.ugrid)

        new_vtu.filename = directory + "/" + file_prefix +str(i)+ ".vtu"

        preds_vals = np.zeros((coords_pred.shape[0], coords_pred.shape[1]+1))
        preds_vals[:,:-1] = coords_pred
        print(preds_vals.shape)
        new_vtu.AddField('Velocity',preds_vals)
        new_vtu.Write()
