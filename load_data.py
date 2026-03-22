import pickle
import sys
import time 
import os

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d
if __name__ == '__main__':
  path_now = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.dirname(path_now)

  dataset="1"
  cfile = os.path.join(project_root, "data/trainset/cam/cam" + dataset + ".p")
  ifile = os.path.join(project_root, "data/trainset/imu/imuRaw" + dataset + ".p")
  vfile = os.path.join(project_root, "data/trainset/vicon/viconRot" + dataset + ".p")

  ts = tic()
  camd = read_data(cfile)
  imud = read_data(ifile)
  vicd = read_data(vfile)
  toc(ts,"Data import")


