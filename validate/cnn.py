import sys
import numpy as np

import utils
import validate.utils as val_utils


def generate_proposals(sess,
                       pred_op,
                       patches_pl,
                       dataset,
                       batch_size,
                       discard=True):
  """Generates detection proposals with inference in mini-batches.

  Args:
    sess: tf session with loaded pred_op variables.
    pred_op: tf op for detection probability prediction.
    patches_pl: patch input placeholder for pred_op op.
    dataset: dataset for which to generate proposals.
    batch_size: size of batch to perform inference.
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
    img, label = dataset.next_image_batch(1)
    img = img[0]
    label = label[0]

    # convert 'img' to patches
    patches = np.array(utils.to_patches(img, patch_size))

    # predict for each patch
    pred = []
    for i in range((len(patches) + batch_size - 1) // batch_size):
      # sample batch
      batch_patches = patches[batch_size * i:batch_size * (i + 1)].reshape(
          [-1, patch_size, patch_size, 1])

      # predict for batch
      batch_preds = sess.run(pred_op, feed_dict={patches_pl: batch_patches})

      # update image pred
      pred.extend(batch_preds[..., 0])

    # put predictions in image format
    pred = np.array(pred).reshape(img.shape[0] - patch_size + 1,
                                  img.shape[1] - patch_size + 1)

    # add borders lost in convolution
    pred = np.pad(pred, ((half_patch_size, half_patch_size),
                         (half_patch_size, half_patch_size)), 'constant')

    # add image prediction to predictions
    preds.append(pred)

    # turn pore label image into list of pore coordinates
    pores.append(np.argwhere(label))
  print('Done.')

  return np.array(pores), np.array(preds)


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
      '--patch_size', type=int, default=17, help='pore patch size')
  parser.add_argument(
      '--results_path', type=str, help='path in which to save results')
  parser.add_argument('--seed', type=int, help='random seed')

  flags = parser.parse_args()

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
  patches_pl, _ = utils.placeholder_inputs()

  with tf.Session() as sess:
    # build graph and restore model
    print('Restoring model...')
    net = models.CNN(patches_pl)
    utils.restore_model(sess, flags.model_dir_path)
    print('Done')

    # compute statistics
    f_score = None
    tdr = None
    fdr = None
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
