import sys
import numpy as np

import utils
import validate.utils as val_utils


def generate_proposals(sess, pred_op, patches_pl, dataset, discard=True):
  """Generates detection proposals with inference in entire images.

  Args:
    sess: tf session with loaded pred_op variables.
    pred_op: tf op for detection probability prediction.
    patches_pl: patch input placeholder for pred_op op.
    dataset: dataset for which to generate proposals.
    discard: whether to only consider keypoints in the area in which
      method is capable of detecting.

  Returns:
    pores: ground truth pore coordinates in shape [N, 2].
    preds: proposals map in shape [N, rows, cols] if discard is true, [N, valid_rows, valid_cols] otherwise.
  """
  patch_size = dataset.patch_size
  half_patch_size = patch_size // 2
  preds = []
  pores = []
  for _ in range(dataset.num_images):
    # get next image and corresponding image label
    (img, *_), (label, *_) = dataset.next_image_batch(1)

    # predict for each image
    pred = sess.run(
        pred_op,
        feed_dict={patches_pl: np.reshape(img, (-1, ) + img.shape + (1, ))})

    # put predictions in image format
    pred = np.array(pred).reshape(img.shape[0] - patch_size + 1,
                                  img.shape[1] - patch_size + 1)

    # treat borders lost in convolution
    if discard:
      label = label[half_patch_size:-half_patch_size, half_patch_size:
                    -half_patch_size]
    else:
      pred = np.pad(pred, ((half_patch_size, half_patch_size),
                           (half_patch_size, half_patch_size)), 'constant')

    # add image prediction to predictions
    preds.append(pred)

    # turn pore label image into list of pore coordinates
    pores.append(np.argwhere(label))

  return np.array(pores), np.array(preds)


def statistics_for_proposed(sess,
                            pred_op,
                            patches_pl,
                            dataset,
                            inter_thrs=None,
                            prob_thrs=None,
                            discard=True):
  '''
  Computes detection parameters that optimize the keypoint detection
  F-score in the dataset with a grid search. This differs from
  by_patches because images are post-processed with thresholding and
  NMS. Parameters of both methods are included in the grid-search.

  Args:
    sess: tf session with loaded pred_op variables.
    pred_op: tf op for detection probability prediction.
    patches_pl: patch input placeholder for pred_op op.
    dataset: dataset to perform grid-search on.
    inter_thrs: iterable of values for NMS threshold. If `None`,
      uses {0.7, 0.6, ..., 0}.
    prob_thrs: iterable of values for probability threshold. If
      `None`, uses {0.9, 0.8, ..., 0.1}.
    discard: whether to only consider keypoints in the area in which
      method is capable of detecting.

  Returns:
    best_f_score: value of best found F-score.
    best_fdr: corresponding value of False Detection Rate.
    best_tdr: corresponding value of True Detection Rate.
    best_inter_thr: NMS intersection threshold that achieves found
      F-score.
    best_prob_thr: probability threshold that achieves found
      F-score.
  '''
  # generate proposals
  pores = None
  preds = None
  pores, preds = generate_proposals(sess, pred_op, patches_pl, dataset,
                                    discard)

  # iterate over thresholds to find best
  if inter_thrs is None:
    inter_thrs = np.arange(0.7, -0.1, -0.1)
  if prob_thrs is None:
    prob_thrs = np.arange(0.9, 0, -0.1)
  best_f_score = 0
  best_tdr = None
  best_fdr = None
  best_inter_thr = None
  best_prob_thr = None
  for prob_thr in prob_thrs:
    coords = []
    probs = []
    # put inference in nms proper format
    for i in range(dataset.num_images):
      img_preds = preds[i]
      pick = img_preds > prob_thr
      coords.append(np.argwhere(pick))
      probs.append(img_preds[pick])

    for inter_thr in inter_thrs:
      # filter detections with nms
      dets = []
      for i in range(dataset.num_images):
        det, _ = utils.nms(coords[i], probs[i], 7, inter_thr)
        dets.append(det)

      f_score, tdr, fdr = val_utils.compute_statistics(pores, dets)

      # update best parameters
      if f_score > best_f_score:
        best_f_score = f_score
        best_tdr = tdr
        best_fdr = fdr
        best_inter_thr = inter_thr
        best_prob_thr = prob_thr

  return best_f_score, best_tdr, best_fdr, best_inter_thr, best_prob_thr


if __name__ == '__main__':
  import argparse
  import os
  import tensorflow as tf

  import polyu
  import models

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--polyu_dir_path',
      required=True,
      type=str,
      help='path to PolyU-HRF dataset')
  parser.add_argument(
      '--model_dir_path', type=str, required=True, help='logging directory')
  parser.add_argument(
      '--post',
      type=str,
      default='proposed',
      help="how to post-process detections. "
      "Can be either 'traditional' or 'proposed'")
  parser.add_argument(
      '--patch_size', type=int, default=17, help='pore patch size')
  parser.add_argument(
      '--results_path', type=str, help='path in which to save results')
  parser.add_argument('--seed', type=int, help='random seed')

  flags = parser.parse_args()

  # check specified post-processing method
  flags.post = flags.post.lower()
  if flags.post not in ('traditional', 'proposed'):
    raise ValueError("post-processing method not available. "
                     "Options are 'traditional' and 'proposed'")

  # set random seeds
  tf.set_random_seed(flags.seed)
  np.random.seed(flags.seed)

  # load polyu dataset
  print('Loading PolyU-HRF dataset...')
  polyu_path = os.path.join(flags.polyu_dir_path, 'GroundTruth',
                            'PoreGroundTruth')
  dataset = polyu.Dataset(
      os.path.join(polyu_path, 'PoreGroundTruthSampleimage'),
      os.path.join(polyu_path, 'PoreGroundTruthMarked'),
      split=(15, 5, 10),
      patch_size=flags.patch_size)
  print('Loaded')

  # gets placeholders for patches and labels
  patches_pl, labels_pl = utils.placeholder_inputs()

  with tf.Session() as sess:
    # build graph and restore model
    print('Restoring model...')
    net = models.FCN(patches_pl, training=False)
    utils.restore_model(sess, flags.model_dir_path)
    print('Done')

    # compute statistics
    f_score = None
    tdr = None
    fdr = None
    if flags.post == 'traditional':
      print('Generating proposals for test set...')
      pores, proposals = generate_proposals(sess, net.predictions, patches_pl,
                                            dataset.test)
      print('Done')

      print('Post-processing...')
      detections = []
      for proposal in proposals:
        detection = utils.post_process_traditional(proposal)
        detections.append(detection)
      detections = np.array(detections)
      print('Done')

      print('Computing statistics...')
      f_score, tdr, fdr = val_utils.compute_statistics(pores, detections)
      print('Done')
    else:
      print('Finding best thresholds in validation set...')
      _, _, _, inter_thr, prob_thr = statistics_for_proposed(
          sess, net.predictions, patches_pl, dataset.val)
      print('Done')

      print('Computing statistics...')
      f_score, tdr, fdr, _, _ = statistics_for_proposed(
          sess,
          net.predictions,
          patches_pl,
          dataset.test,
          inter_thrs=[inter_thr],
          prob_thrs=[prob_thr])

    # direct output according to user specification
    results_file = None
    if flags.results_path is None:
      results_file = sys.stdout
    else:
      # create path, if does not exist
      dirname = os.path.dirname(flags.results_path)
      dirname = os.path.abspath(dirname)
      if not os.path.exists(dirname):
        os.makedirs(dirname)

      results_file = open(flags.results_path, 'w')

    print('Whole image evaluation:', file=results_file)
    print('TDR = {}'.format(tdr), file=results_file)
    print('FDR = {}'.format(fdr), file=results_file)
    print('F score = {}'.format(f_score), file=results_file)
    if flags.post != 'traditional':
      print('inter_thr = {}'.format(inter_thr), file=results_file)
      print('prob_thr = {}'.format(prob_thr), file=results_file)
