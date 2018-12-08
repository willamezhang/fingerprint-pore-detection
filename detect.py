import argparse
import tensorflow as tf
import cv2

import utils
import models

FLAGS = None


def main():
  half_patch_size = FLAGS.patch_size // 2

  with tf.Graph().as_default():
    image_pl, _ = utils.placeholder_inputs()

    print('Building graph...')
    net = models.FCN(image_pl, training=False)
    print('Done')

    with tf.Session() as sess:
      print('Restoring model in {}...'.format(FLAGS.model_dir_path))
      utils.restore_model(sess, FLAGS.model_dir_path)
      print('Done')

      print('Loading image...')
      image = cv2.imread(FLAGS.image_path, 0)
      print('Done')

      print('Detecing pores...')
      detections = utils.detect_pores(image, image_pl, net.predictions,
                                      half_patch_size, FLAGS.prob_thr,
                                      FLAGS.inter_thr, sess)
      print('Done')

      print('Saving detections to {}...'.format(FLAGS.save_path))
      utils.save_dets_txt(detections, FLAGS.save_path)
      print('Done')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_path',
      required=True,
      type=str,
      help='path to image in which to detect pores')
  parser.add_argument(
      '--model_dir_path',
      type=str,
      required=True,
      help='path from which to restore trained model')
  parser.add_argument(
      '--patch_size', type=int, default=17, help='pore patch size')
  parser.add_argument(
      '--save_path',
      type=str,
      default='detections.txt',
      help='path to file in which detections should be saved')
  parser.add_argument(
      '--prob_thr',
      type=float,
      default=0.6,
      help='probability threshold to filter detections')
  parser.add_argument(
      '--inter_thr', type=float, default=0, help='nms intersection threshold')

  FLAGS = parser.parse_args()

  main()
