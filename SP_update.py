#This script creates symbolic links from AJ's 1909 singlepulse directory to mine

import os

singlepulses = "/fred/oz005/users/ajameson/J1909-3744/2020-02-13-03:05:04/scr/"
target_dir = "/fred/oz002/users/mmiles/SinglePulse"

os.chdir(singlepulses)
for sp in os.listdir(singlepulses):
    os.symlink(os.path.join(singlepulses,sp), os.path.join(target_dir,sp))