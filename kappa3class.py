import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt

CL_1 = 0
CL_2 = 127
CL_3 = 254

def class2labelimg(image):
    [w,b,_] = image.shape
    li = np.zeros([w,b])
    li[np.where((image == [1,0,0]).all(axis=2))] = CL_1
    li[np.where((image == [0,0,1]).all(axis=2))] = CL_2
    li[np.where((image == [0,0,0]).all(axis=2))] = CL_3
    return li

def get_confusion_matrix(Inet, Itruth):
    confusion_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])

    confusion_matrix[0,0] += np.all([Inet == CL_1, Itruth == CL_1], axis = 0).sum()
    confusion_matrix[0,1] += np.all([Inet == CL_1, Itruth == CL_2], axis = 0).sum()
    confusion_matrix[0,2] += np.all([Inet == CL_1, Itruth == CL_3], axis = 0).sum()

    confusion_matrix[1,0] += np.all([Inet == CL_2, Itruth == CL_1], axis = 0).sum()
    confusion_matrix[1,1] += np.all([Inet == CL_2, Itruth == CL_2], axis = 0).sum()
    confusion_matrix[1,2] += np.all([Inet == CL_2, Itruth == CL_3], axis = 0).sum()

    confusion_matrix[2,0] += np.all([Inet == CL_3, Itruth == CL_1], axis = 0).sum()
    confusion_matrix[2,1] += np.all([Inet == CL_3, Itruth == CL_2], axis = 0).sum()
    confusion_matrix[2,2] += np.all([Inet == CL_3, Itruth == CL_3], axis = 0).sum()

    return confusion_matrix

def conf2kappa(confusion_matrix):
    agreements = np.diagonal(confusion_matrix).sum()
    column_sums = np.array(confusion_matrix).sum(axis=0)
    row_sums = np.array(confusion_matrix).sum(axis=1)
    total = np.array(confusion_matrix).sum()
    expected = float(np.multiply(column_sums, row_sums).sum()) / total
    k = (agreements - expected) / (total - expected)
    return k

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--numimgs', type=int, required=True)
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('-i', '--iter', type=int, required=False)

    args = vars(parser.parse_args())

    confusion_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])

    k_list = []

    for i in range(0, args['numimgs']):
        net_path = '/'.join([args['path'].rstrip('/'), 'prediction_i{}_b0.png'.format(i)])
        truth_path = '/'.join([args['path'].rstrip('/'), 'truth_i{}_b0.png'.format(i)])
        net_c = io.imread(net_path)
        truth_c = io.imread(truth_path)

        #print(net_c.shape, truth_c.shape)
        #net = class2labelimg(net_c)
        #truth = class2labelimg(truth_c)
        net = net_c
        truth = truth_c

        if args['show']:
            norm = plt.Normalize(vmin=1, vmax=3)
            fig, axarr = plt.subplots(1,2, figsize=(20, 10))
            axarr[0].imshow(net, cmap = 'Set1', norm=norm)
            axarr[0].set_title('Network output')
            axarr[1].imshow(truth, cmap = 'Set1', norm=norm)
            axarr[1].set_title('Ground truth')

            plt.show()

        confusion_matrix_i = get_confusion_matrix(net, truth)

#	k_list.append(conf2kappa(confusion_matrix_i))

#	print("Image {}...".format(i));
#	print("\tKappa = {}".format(conf2kappa(confusion_matrix_i)));

        confusion_matrix = np.add(confusion_matrix, confusion_matrix_i)

#    print("Kappa mean {}, standard deviation {}".format(np.mean(k_list), np.std(k_list)))

#    print(confusion_matrix)
#    print(confusion_matrix.sum())

    if args['iter']:
        print("Iter = {}, kappa = {}".format(args['iter'], conf2kappa(confusion_matrix)))
    else:
        print("Kappa = {}".format(conf2kappa(confusion_matrix)))
        


if __name__ == '__main__':
    main()
