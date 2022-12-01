from glob import glob

from sklearn.model_selection import train_test_split
from pathlib import Path
from os import makedirs, path, walk
from shutil import copy
from argparse import ArgumentParser
from glob import glob
from xml.dom import minidom
import numpy as np
import shapely
from tqdm import tqdm
import imageio.v2 as imageio
#from imageio import imread, imwrite
from matplotlib import pyplot as plt
from numpy.random import randint
import os
from sklearn.cluster import KMeans
from PIL import Image
from PIL.ExifTags import TAGS


def get_all_classes(dir_path, label_type="subclass"):
    files = []
    labels = []
    label = None
    for r, d, f in walk(dir_path):

        if label_type == "class" and len(f) <= 0:
            label = r
            labels.append(label)
        elif label_type == "subclass" and len(f) > 0:
            label = r
            labels.append(label)
        elif label_type != "class" and label_type != "subclass":
            raise Exception("[get_all_classes] Invalid Option!")

    return labels


def main():
    

   # path_in = Path("/home/gabriel/research/rs_data/AID/data_split") # /data/bulk/bigearthnet/data/truecolor
    #path_out = Path("/home/gabriel/research/quantize") # /data/bulk/bigearthnet/compressed/truecolor

    #    class_type = 'subclass'
    parser = ArgumentParser(description='This script splits a dataset into train and validation splits')
    parser.add_argument('-i', '--dataset',
                        dest='dataset',
                        help='Root folder path contains multiple patch folders.',
                        required=True)

    parser.add_argument('-o', '--out_folder',
                        dest='out_folder',
                        help='Destination path of the resulting splits.',
                        required=True)

    args = parser.parse_args()

    if args.dataset:
        if not path.exists(args.dataset):
            print('ERROR: folder', args.dataset, 'does not exist')
            exit()

        path_in = Path(args.dataset)
    else:
        print('ERROR: dataset', args.dataset, 'does not exist')
        exit()

    if args.out_folder:
        path_out = Path(args.out_folder)
    else:
        print('ERROR: out_folder', args.out_folder, 'does not exist')
        exit()

    print("Parameter")
    print("=========")
    print(" In:", path_in)
    print(" Out:", path_out)


    if path.exists(path_out) is False:
        makedirs(path_out)
    else:
        print("\nPath already exists!\nCheck if dataset already exists.")
        exit()

    print()
    print("Collect files")
    print("=============")
    print(" start collecting files...")
    labels = get_all_classes(path_in)
    print(" done...")



    for cp_id in range(len(labels)):
        print("[", cp_id, "/", len(labels)-1, "]", labels[cp_id])

        # TODO: Do for train and test
        with open(path_out / (Path(labels[cp_id]).name + "_" + "pixeldata.csv"), "w") as ofs:
            for f in tqdm(glob(labels[cp_id] + "/*"), disable=True):

                m = imageio.imread(f)

                m = m.reshape(-1, m.shape[-1])[:, 0:3]
                m = m[np.random.choice(m.shape[0], 32, replace=False), :]
                np.savetxt(ofs, m)

        X = np.loadtxt(path_out / (Path(labels[cp_id]).name + "_" + "pixeldata.csv"))
        r_array = [1,2,3,4,5,6,7,8]
        for logn in r_array:
            n = 2 ** logn
            print("Working on %d" % (n))

            km = KMeans(n_clusters=n, random_state=42).fit(X)
            clusters = km.cluster_centers_
#            print(labels[cp_id])
            for f in tqdm(glob((labels[cp_id] + "/*")), disable=True):
                file_out = path_out / "reference" / str("km-%d"%logn) / f[len(str(path_in))+1:]

                Path(file_out.parent).mkdir(parents=True, exist_ok=True)

                im = imageio.imread(f)
                
                shape = (im.shape[0], im.shape[1], 3)
                im = im.reshape(-1, im.shape[-1])[:, 0:3]
                imlabels = km.predict(im)
                results = clusters[imlabels].astype(np.uint8).reshape(shape)

                kwargs = {}

                ext =  str(file_out).split(".")[-1]
                if ext == "jpg" or ext == "JPEG":
                    kwargs["quality"] = 100
                elif ext == "png" or ext == "PNG":
                    kwargs["compress_level"] = 0

                imageio.imwrite(file_out, results, **kwargs)

if __name__ == '__main__':
    main()
