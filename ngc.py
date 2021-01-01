import sys, os, subprocess, json
import shutil
from pathlib import Path

ngc_bin = "/home/jseo/nvidia/bin/ngc"

def run_ngc(args, jobId, as_json=False):
    if as_json:
        jid = ["--format_type", "json", str(jobId)]
    else:
        jid = [str(jobId)]
    result = subprocess.run([ngc_bin] + args.split() + jid, \
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = result.stderr.decode('utf-8')
    if as_json:
        out = json.loads(result.stdout.decode('utf-8')) if len(result.stdout) else None
    else:
        out = result.stdout.decode('utf-8').split()
    return out, err

def ngc_result_latest(jobId, destDir="ngc_result"):
    # download
    out, err = run_ngc("result info --files", jobId)
    pkls = sorted([x for x in out if x.endswith(".pkl")])
    filepath = pkls[-1]
    cmd = "result download --file %s" % (filepath)
    if destDir:
        if not os.path.exists(destDir):
            os.mkdir(destDir)
        cmd += " --dest %s" % destDir
    out, err = run_ngc(cmd, jobId, True)
    if out["status"] != 'Completed':
        return None
    local_dir = out["local_path"]
    orig_path = local_dir + filepath

    # assemble a new path
    new_path = str(Path(filepath).parent)
    new_path = str(jobId) + "-" + new_path[new_path.find("-")+1:]
    new_path = str(Path(local_dir).parent) + "/" + new_path + filepath[filepath.rfind("-"):]

    while os.path.exists(new_path):
        new_path = new_path[:-3] + "new.pkl"
    shutil.move(orig_path, new_path)
    shutil.rmtree(local_dir)

    return new_path

# ngc_result_latest(1660969)
