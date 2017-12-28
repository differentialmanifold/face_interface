import os
import argparse
import sys
import numpy as np
from scipy import misc
from validation import valid_utils
from face_exception import FaceException
import face_compare

dir_name = '~/share/data/CAS-PEAL-R1'
file_extend = 'tif'


def main(args):
    # Read the file containing the pairs used for testing
    pairs = valid_utils.read_pairs(os.path.expanduser(args.data_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = valid_utils.get_paths(os.path.expanduser(args.data_dir), pairs, args.data_file_ext)

    uncompleted, tp, fp, tn, fn, distances, dist_verify = calc_acc(paths, actual_issame)

    accurate = (len(tp) + len(tn)) / (len(tp) + len(tn) + len(fp) + len(fn))
    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))

    print('uncompleted data is {}'.format(uncompleted))
    print('false positive is {}'.format(fp))
    print('false negative is {}'.format(fn))
    print('accurate is {}, precision is {}, recall is {}'.format(accurate, precision, recall))

    max_acc, max_threshold = find_best_dist(distances, dist_verify)
    print('max accuracy is {} max threshold is {}'.format(max_acc, max_threshold))


def calc_acc(paths, actual_issame):
    uncompleted = []
    tp = []
    fp = []
    tn = []
    fn = []

    distances = []
    dist_verify = []

    for i in range(len(actual_issame)):
        img_arrs = [misc.imread(paths[2 * i], mode='RGB'), misc.imread(paths[2 * i + 1], mode='RGB')]

        try:
            compare_obj = face_compare.compare_face_in_image(img_arrs, show_distance=True)
            if compare_obj['issame'] and actual_issame[i]:
                tp.append(i)
            if (not compare_obj['issame']) and actual_issame[i]:
                fn.append(i)
            if compare_obj['issame'] and (not actual_issame[i]):
                fp.append(i)
            if (not compare_obj['issame']) and (not actual_issame[i]):
                tn.append(i)

            distances.append(compare_obj['dist'])
            dist_verify.append(actual_issame[i])

        except FaceException as err:
            uncompleted.append(i)
            print(err)
        print('finished line {}'.format(i))
    return uncompleted, tp, fp, tn, fn, distances, dist_verify


def find_best_dist(distances, dist_verify):
    thresholds = np.arange(0, 4, 0.01)
    distances = np.array(distances)
    dist_verify = np.array(dist_verify)
    max_acc = 0
    max_threshold = 0
    for threshold in thresholds:
        acc = np.mean(((distances < threshold) == dist_verify).astype(np.int))
        if acc > max_acc:
            max_acc = acc
            max_threshold = threshold
    return max_acc, max_threshold


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing images.', default=dir_name + '/raw')
    parser.add_argument('--data_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default=dir_name + '/pairs.txt')
    parser.add_argument('--data_file_ext', type=str,
                        help='The file extension for the dataset.', default=file_extend, choices=['jpg', 'png', 'tif'])
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
