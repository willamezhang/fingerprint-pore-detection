import os
import tensorflow as tf
import numpy as np
import cv2

from scipy.ndimage.measurements import label as connected_comps


def to_patches(img, patch_size):
  patches = []
  for i in range(img.shape[0] - patch_size + 1):
    for j in range(img.shape[1] - patch_size + 1):
      patches.append(img[i:i + patch_size, j:j + patch_size])

  return patches


def placeholder_inputs(labels_dim=1):
  images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
  labels = tf.placeholder(tf.float32, [None, labels_dim], name='labels')
  return images, labels


def fill_feed_dict(dataset, patches_pl, labels_pl, batch_size):
  """
  Creates a tf feed_dict containing patches and corresponding labels from a
  mini-batch of size batch_size sampled from dataset.
  for online dataset augmentation.

  Args:
    dataset: dataset object satisfying polyu._Dataset.next_batch signature.
    patches_pl: tf placeholder for patches.
    labels_pl: tf placeholder for labels.
    batch_size: size of mini-batch to be sampled.

  Returns:
    tf feed_dict containing patches and corresponding labels.
  """
  patches_feed, labels_feed = dataset.next_batch(batch_size)

  feed_dict = {
      patches_pl: np.expand_dims(patches_feed, axis=-1),
      labels_pl: np.expand_dims(labels_feed, axis=-1)
  }

  return feed_dict


def create_dirs(log_dir_path, batch_size, learning_rate):
  import datetime
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
  log_dir = os.path.join(
      log_dir_path, 'bs-{}_lr-{:.0e}_t-{}'.format(batch_size, learning_rate,
                                                  timestamp))
  tf.gfile.MakeDirs(log_dir)

  return log_dir


def nms(centers, probs, bb_size, thr):
  """
  Converts each center in centers into center centered bb_size x bb_size
  bounding boxes and applies Non-Maximum-Suppression to them.

  Args:
    centers: centers of detection bounding boxes.
    probs: probabilities for each of the detections.
    bb_size: bounding box size.
    thr: NMS discarding intersection threshold.

  Returns:
    dets: np.array of NMS filtered detections.
    det_probs: np.array of corresponding detection probabilities.
  """
  area = bb_size * bb_size
  half_bb_size = bb_size // 2

  xs, ys = np.transpose(centers)
  x1 = xs - half_bb_size
  x2 = xs + half_bb_size
  y1 = ys - half_bb_size
  y2 = ys + half_bb_size

  order = np.argsort(probs)[::-1]

  dets = []
  det_probs = []
  while len(order) > 0:
    i = order[0]
    order = order[1:]
    dets.append(centers[i])
    det_probs.append(probs[i])

    xx1 = np.maximum(x1[i], x1[order])
    yy1 = np.maximum(y1[i], y1[order])
    xx2 = np.minimum(x2[i], x2[order])
    yy2 = np.minimum(y2[i], y2[order])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (2 * area - inter)

    inds = np.where(ovr <= thr)[0]
    order = order[inds]

  return np.array(dets), np.array(det_probs)


def pairwise_distances(x1, x2):
  # memory efficient implementation based on Yaroslav Bulatov's answer in
  # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
  sqr1 = np.sum(x1 * x1, axis=1, keepdims=True)
  sqr2 = np.sum(x2 * x2, axis=1)
  D = sqr1 - 2 * np.matmul(x1, x2.T) + sqr2

  return D


def restore_model(sess, model_dir):
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    raise IOError('No model found in {}.'.format(model_dir))


def load_image(image_path):
  image = cv2.imread(image_path, 0)
  image = np.array(image, dtype=np.float32) / 255
  return image


def load_images(folder_path):
  images = []
  for image_path in sorted(os.listdir(folder_path)):
    if image_path.endswith(('.jpg', '.png', '.bmp')):
      images.append(load_image(os.path.join(folder_path, image_path)))

  return images


def load_images_with_names(images_dir):
  images = load_images(images_dir)
  image_names = [
      path.split('.')[0] for path in sorted(os.listdir(images_dir))
      if path.endswith(('.jpg', '.bmp', '.png'))
  ]

  return images, image_names


def save_dets_txt(dets, filename):
  with open(filename, 'w') as f:
    for coord in dets:
      print(coord[0] + 1, coord[1] + 1, file=f)


def load_dets_txt(pts_path):
  pts = []
  with open(pts_path, 'r') as f:
    for line in f:
      row, col = [int(t) for t in line.split()]
      pts.append((row - 1, col - 1))

  return pts


def find_correspondences(descs1,
                         descs2,
                         pts1=None,
                         pts2=None,
                         euclidean_weight=0,
                         transf=None,
                         thr=None):
  """
  Finds bidirectional correspondences between descs1 descriptors and
  descs2 descriptors. If thr is provided, discards correspondences
  that fail a distance ratio check with threshold thr. If pts1, pts2,
  and transf are give, the metric considered when finding correspondences
  is
    d(i, j) = ||descs1(j) - descs2(j)||^2 + euclidean_weight *
      * ||transf(pts1(i)) - pts2(j)||^2

  Args:
    descs1: [N, M] np.array of N descriptors of dimension M each.
    descs2: [N, M] np.array of N descriptors of dimension M each.
    pts1: [N, 2] np.array of N coordinates for each descriptor in descs1.
    pts2: [N, 2] np.array of N coordinates for each descriptor in descs2.
    euclidean_weight: weight given to spatial constraint in comparison
      metric.
    transf: alignment transformation that aligns pts1 to pts2.
    thr: distance ratio check threshold.

  Returns:
    list of correspondence tuples (j, i, d) in which index j of
      descs2 corresponds with i of descs1 with distance d.
  """
  # compute descriptors' pairwise distances
  D = pairwise_distances(descs1, descs2)

  # add points' euclidean distance
  if euclidean_weight != 0:
    assert transf is not None
    assert pts1 is not None
    assert pts2 is not None

    # assure pts are np array
    pts1 = transf(np.array(pts1))
    pts2 = np.array(pts2)

    # compute points' pairwise distances
    euclidean_D = pairwise_distances(pts1, pts2)

    # add to overral keypoints distance
    D += euclidean_weight * euclidean_D

  # find bidirectional corresponding points
  pairs = []
  if thr is None or len(descs1) == 1 or len(descs2) == 1:
    # find the best correspondence of each element
    # in 'descs2' to an element in 'descs1'
    corrs2 = np.argmin(D, axis=0)

    # find the best correspondence of each element
    # in 'descs1' to an element in 'descs2'
    corrs1 = np.argmin(D, axis=1)

    # keep only bidirectional correspondences
    for i, j in enumerate(corrs2):
      if corrs1[j] == i:
        pairs.append((j, i, D[j, i]))
  else:
    # find the 2 best correspondences of each
    # element in 'descs2' to an element in 'descs1'
    corrs2 = np.argpartition(D.T, [0, 1])[:, :2]

    # find the 2 best correspondences of each
    # element in 'descs1' to an element in 'descs2'
    corrs1 = np.argpartition(D, [0, 1])[:, :2]

    # find bidirectional corresponding points
    # with second best correspondence 'thr'
    # worse than best one
    for i, (j, _) in enumerate(corrs2):
      if corrs1[j, 0] == i:
        # discard close best second correspondences
        if D[j, i] < D[corrs2[i, 1], i] * thr:
          if D[j, i] < D[j, corrs1[j, 1]] * thr:
            pairs.append((j, i, D[j, i]))

  return pairs


def detect_pores(image, image_pl, predictions, half_patch_size, prob_thr,
                 inter_thr, sess):
  """
  Detects pores in an image. First, a pore probability map is computed
  with the tf predictions op. This probability map is then thresholded
  and converted to coordinates, which are filtered with NMS.

  Args:
    image: image in which to detect pores.
    image_pl: tf placeholder holding net's image input.
    predictions: tf tensor op of net's output.
    half_patch_size: half the detection patch size. used for padding the
      predictions to the input's original dimensions.
    prob_thr: probability threshold.
    inter_thr: NMS intersection threshold.
    sess: tf session

  Returns:
    detections for image in shape [N, 2]
  """
  # predict probability of pores
  pred = sess.run(
      predictions,
      feed_dict={image_pl: np.reshape(image, (1, ) + image.shape + (1, ))})

  # add borders lost in convolution
  pred = np.reshape(pred, pred.shape[1:-1])
  pred = np.pad(pred, ((half_patch_size, half_patch_size),
                       (half_patch_size, half_patch_size)), 'constant')

  detections = post_process_proposed(predictions, prob_thr, inter_thr)

  return detections


def post_process_proposed(predictions, prob_thr, inter_thr):
  # threshold and convert into coodinates
  pick = predictions > prob_thr
  coords = np.argwhere(pick)
  probs = predictions[pick]

  # filter with nms
  dets, _ = nms(coords, probs, 7, inter_thr)

  return dets


def post_process_traditional(proposals, threshold=0.5):
  # threshold predictions
  thresholded = proposals > threshold

  # merge connected components
  comps, labels = connected_comps(thresholded)
  dets = []
  for label in range(1, labels + 1):
    coords = np.argwhere(comps == label)
    dets.append(np.mean(coords, axis=0))

  return np.array(dets)
