import os
import json
from glob import glob
from pathlib import Path
from argparse import ArgumentParser


def create_configs(j_in, f_out, number):
    count = 0
    
    with open(j_in) as j:
        j_data = json.loads(j.read())
        for line in number:
            count += 1
            j_data["name"] = line.strip()
            j_data["dataset"] = line.strip()

            with open(str(f_out) + "/" + line.strip() + ".json", 'w') as outfile:
                json.dump(j_data, outfile, indent=4, sort_keys=True)

            print("Processed line {}: {}".format(count, line.strip()))


def main():

    #    class_type = 'subclass'
    parser = ArgumentParser(description='This script creates *.json files based on a master file.')
    parser.add_argument('-n', '--names',
                        dest='dataset',
                        help='Path to the dataset folder',
                        required=True)
    
    parser.add_argument('-j', '--json',
                        dest='jfile',
                        help='Master Json file',
                        required=True)

    parser.add_argument('-o', '--out_folder',
                        dest='out_folder',
                        help='Output folder for the config files',
                        required=True)

    args = parser.parse_args()

    path_in = Path(args.dataset)
    jfile = Path(args.jfile)
    path_out = Path(args.out_folder)

    if os.path.exists(path_out) is False:
        os.makedirs(path_out) 

    print("Parameter")
    print("=========")
    print(" File names:", path_in)
    print(" Master JSON:", jfile)
    print(" Out:", path_out)

    d_name = str(Path(path_in).name).lower()

    number = []

    for f in glob(str(path_in) + "/*/"):
        file_type = Path(f).name
        for k in glob(str(path_in) + "/" + file_type + "/*/"):
            k_number = Path(k).name
        
        
        
            number.append(d_name + "-" + file_type + "-" + k_number)
    number.sort()

    create_configs(jfile, path_out, number)


if __name__ == '__main__':
    main()
