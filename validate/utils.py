import numpy as np

import utils


def by_patches(sess,
               preds,
               batch_size,
               patches_pl,
               labels_pl,
               dataset,
               thrs=None):
  """
  Computes detection parameters that optimize the patch-based keypoint
  detection F-score in the dataset with a grid search. This is done
  efficiently, by sorting probability scores and iterating over them.

  Args:
    sess: tf session with loaded preds variables.
    preds: tf op for detection probability prediction.
    batch_size: size of mini-batch.
    patches_pl: patch input placeholder for preds op.
    labels_pl: label input placeholder to retrieve labels from
      tf input feed dict.
    dataset: dataset to perform grid-search on.
    thrs: grid-search threshold space. if None, defaults to
      {0.01, 0.02, ..., 1}.

  Returns:
    best_f_score: value of best found F-score.
    best_fdr: corresponding value of False Detection Rate.
    best_tdr: corresponding value of True Detection Rate.
  """
  # initialize dataset statistics
  true_preds = []
  false_preds = []
  total = 0

  steps_per_epoch = (dataset.num_samples + batch_size - 1) // batch_size
  for _ in range(steps_per_epoch):
    feed_dict = utils.fill_feed_dict(dataset, patches_pl, labels_pl,
                                     batch_size)

    # evaluate batch
    batch_preds = sess.run(preds, feed_dict=feed_dict)
    batch_labels = feed_dict[labels_pl]
    batch_total = np.sum(batch_labels)

    # update dataset statistics
    total += batch_total
    if batch_total > 0:
      true_preds.extend(batch_preds[batch_labels == 1].flatten())
    if batch_total < batch_labels.shape[0]:
      false_preds.extend(batch_preds[batch_labels == 0].flatten())

  # sort for efficient computation of tdr/fdr over thresholds
  true_preds.sort()
  true_preds.reverse()
  false_preds.sort()
  false_preds.reverse()

  # compute tdr/fdr score over thresholds
  best_f_score = 0
  best_fdr = None
  best_tdr = None

  true_pointer = 0
  false_pointer = 0

  eps = 1e-5
  if thrs is None:
    thrs = np.arange(1.01, -0.01, -0.01)
  for thr in thrs:
    # compute true positives
    while true_pointer < len(true_preds) and true_preds[true_pointer] >= thr:
      true_pointer += 1

    # compute false positives
    while false_pointer < len(
        false_preds) and false_preds[false_pointer] >= thr:
      false_pointer += 1

    # compute tdr and fdr
    tdr = true_pointer / (total + eps)
    fdr = false_pointer / (true_pointer + false_pointer + eps)

    # compute and update f score
    f_score = 2 * (tdr * (1 - fdr)) / (tdr + 1 - fdr)
    if f_score > best_f_score:
      best_tdr = tdr
      best_fdr = fdr
      best_f_score = f_score

  return best_f_score, best_fdr, best_tdr


def compute_statistics(pores, detections):
  """Computes true detection rate (TDR), false detection rate (FDR) and the corresponding F-score for the given groundtruth and detections.

  Args:
    pores: pore groundtruth coordinates in format [N, P, 2].
      Second dimension must be np.arrays.
    detections: detection coordinates in the same format as pores, ie
      [N, D, 2] and second dimension np.arrays.

  Returs:
    f_score: F-score for the given detections and groundtruth.
    tdr: TDR for the given detections and groundtruth.
    fdr: FDR for the given detections and groundtruth.
  """
  # find correspondences between detections and pores
  total_pores = 0
  total_dets = 0
  true_dets = 0
  for i in range(len(pores)):
    # update totals
    total_pores += len(pores[i])
    total_dets += len(detections[i])
    true_dets += len(utils.find_correspondences(pores[i], detections[i]))

  # compute tdr, fdr and f score
  eps = 1e-5
  tdr = true_dets / (total_pores + eps)
  fdr = (total_dets - true_dets) / (total_dets + eps)
  f_score = 2 * (tdr * (1 - fdr)) / (tdr + (1 - fdr))

  return f_score, tdr, fdr
