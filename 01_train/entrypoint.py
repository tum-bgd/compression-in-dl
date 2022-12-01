import os
import sys
import json
import importlib
from types import SimpleNamespace


def main():

    if len(sys.argv) != 2:
        print("[ERROR] Usage: %s job.json, where job.json is a suitable JSON file" % (sys.argv[0]))
        sys.exit(-1)

#    data = json.load(open("./datasets/datasets.json","r"))
    job_cfg = json.load(open(sys.argv[1],"r"))  

    if isinstance(job_cfg, list) is False:
        job_cfg = [job_cfg]

    for job in job_cfg:
#        job.update(data[job["dataset"]])
        print(job)
        mod = importlib.import_module("models.%s" %(job["inference_script"]))
        globals()["model"] = getattr(mod, "train")
        model(**job)


if __name__=="__main__":
    main()
