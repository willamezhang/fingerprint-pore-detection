import os
import numpy as np

import utils
from validate import utils as vutils

FLAGS = None


def main():
  # load only test files for both pores and detections
  indices = sorted([str(i) for i in range(1, 31)])[-10:]
  pores = []
  detections = []
  for index in indices:
    pores_path = os.path.join(FLAGS.pores_dir, index + '.txt')
    index_pores = utils.load_dets_txt(pores_path)
    pores.append(np.array(index_pores))

    dets_path = os.path.join(FLAGS.dets_dir, index + '.txt')
    index_dets = utils.load_dets_txt(dets_path)
    detections.append(np.array(index_dets))

  # compute statistics
  f_score, tdr, fdr = vutils.compute_statistics(pores, detections)
  print('TDR = {}'.format(tdr))
  print('FDR = {}'.format(fdr))
  print('F-score = {}'.format(f_score))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      'pores_dir',
      type=str,
      help='path to pore coordinates groundtruth folder')
  parser.add_argument('dets_dir', type=str, help='path to detections folder')

  FLAGS = parser.parse_args()

  main()
