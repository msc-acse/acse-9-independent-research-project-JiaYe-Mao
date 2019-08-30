import numpy as np
from importance_map_reduced import *
from keras.models import load_model


# np.set_printoptions(threshold=np.inf)

nsplt=2

def determine_structured_mesh_info(min, max, dim=2):

    length = [max[i] - min[i] for i in range(dim)]

    length = np.array(length)
    structured_shape = [None for e in range(dim)]
    print(type(length))
    print(length)
    for i in range(length.shape[0]):
        if (length[i]==length.min()):
            structured_shape[i]=32
        else:
            frac = (int)(float(length[i]) / float(length.min()) *2)
            print("frac", frac)
            structured_shape[i]=frac*16

    return np.array(structured_shape)

def get_domain_info(nsplt, witchd, coordinates, dim=2):

    max_all = []
    min_all = []
    for i in range(2**nsplt):
        max=[-99999 for e in range(dim)]
        min=[999999 for e in range(dim)]
        print(type(max))
        for j in range(witchd.shape[0]):
            if(witchd[j] == i+1):
                for k in range(dim):
                    if(max[k]<coordinates[j, k]):
                        max[k] = coordinates[j, k]
                    if(min[k]>coordinates[j, k]):
                        min[k] = coordinates[j, k]
        max_all.append(max)
        min_all.append(min)

    print(max_all)
    print(min_all)

    structured_shape = [determine_structured_mesh_info(min_all[i], max_all[i], dim=dim) for i in range(2**nsplt)]

    return structured_shape, min_all, max_all



# msh = []
#
# for i in range(10):
#     msh_x = []
#     for j in range(10):
#         msh_x.append(i+j)
#     msh.append(msh_x)
#
# print(msh)
# msh = np.array(msh)
# print(msh.shape)
#
# max_x = 18
# max_y = 18
# dx = max_x / (msh.shape[1]-1)
# dy = max_y / (msh.shape[0]-1)

# def interpolation(point, msh):
#
#     index_y = (int)(point[0] / dy)
#     index_x = (int)(point[1] / dx)
#     print(index_y, index_x)
#     if (index_y >= 9):
#         index_y = index_y - 1
#     wy = (point[0] - index_y * dy) / dy
#
#     C_left = wy * msh[index_y+1, index_x] + (1-wy) * msh[index_y, index_x]
#     C_right = wy * msh[index_y+1, index_x+1] + (1-wy) * msh[index_y, index_x+1]
#     print(C_left, C_right)
#     wx = (point[1] - index_x*dx)/dx
#     C_interpolation = wx * C_right + (1-wx) * C_left
#     return C_interpolation

def interpolation_back_to_unstructured_mesh(point, msh, dx, dy, start_x, start_y, feature_map):
    print("point", point)
    print("dx_1:", dx, "dy_1", dy)
    print("start_x", start_x, "start_y", start_y)
    index_y = (int)((point[1]-start_y) / dy)
    index_x = (int)((point[0]-start_x) / dx)
    print("index:", index_x, index_y)
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

filename="snapshots/Flowpast_2d_Re3900_200.vtu"
fieldname="Velocity"
file_prefix="nirom_prediction_"
clean_vtu = get_clean_vtk_file(filename)
model_name = "decoder_dd4.h5"
directory = "Result_dd_4"
string = "mkdir " + directory
os.system(string)

coords = get_node_coords_from_vtk(filename)

# get_domain_info(nsplt, filename, dim=2)

witchd = np.load("witchd.npy")
structured_shape, min_all, max_all = get_domain_info(nsplt, witchd, coords, dim=2)

print("structured_shape", structured_shape)
print("min_all", min_all)
print("max_all", max_all)

dx_and_dy = []

for i in range(len(structured_shape)):
    dx = float(max_all[i][0]-min_all[i][0]) / (structured_shape[i][0]-1)
    dy = float(max_all[i][1]-min_all[i][1]) / (structured_shape[i][1]-1)
    dx_and_dy.append([dx,dy])

print(dx_and_dy)


preds = []
for i in range(2**nsplt):
    preds.append(np.load("output_" + str(i) + ".npy"))
    print("preds : ", preds[i].shape)
preds = [preds[2], preds[3], preds[1], preds[0]]


# print("decoded. shape", preds[0].shape)
# print("decoded. shape", preds[1].shape)

# TOdo : delete this, wrong sequence in autoencoder output
# preds = [preds_origin[1], preds_origin[0]]


clean_vtu = get_clean_vtk_file(filename)
string = "mkdir " + directory
os.system(string)

# snapshots
for i in range(preds[0].shape[0]):
    coords_pred = []
    # nodes
    for j in range(coords.shape[0]):
        # domains
        for k in range(len(structured_shape)):
            if k+1 == witchd[j]:
                x1 = interpolation_back_to_unstructured_mesh(coords[j], preds[k][i], dx_and_dy[k][0], dx_and_dy[k][1], min_all[k][0], min_all[k][1], 0)   # use 10*i, since we only generate 1/10 vtu files
                x2 = interpolation_back_to_unstructured_mesh(coords[j], preds[k][i], dx_and_dy[k][0], dx_and_dy[k][1], min_all[k][0], min_all[k][1], 1)
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
