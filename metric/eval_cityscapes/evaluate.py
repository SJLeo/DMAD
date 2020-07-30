import os
import caffe
import argparse
import numpy as np
import scipy.misc
from PIL import Image
from util import segrun, fast_hist, get_scores
from cityscapes import cityscapes

parser = argparse.ArgumentParser()
parser.add_argument("--cityscapes_dir", type=str, required=True, help="Path to the original cityscapes dataset")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the evaluation results")
parser.add_argument("--caffemodel_dir", type=str, default='./caffemodel/', help="Where the FCN-8s caffemodel stored")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
parser.add_argument("--split", type=str, default='val', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the FCN output images")
args = parser.parse_args()

# python evaluate.py \
#        --cityscapes_dir ~/datasets/cityscapes/ \
#        --result_dir /media/disk1/lishaojie/DMAD/experiments/cityscapes_pretrain_hinge/test_results/fake_B \
#        --output_dir ./output_dir \
#        --gpu_id 1 \
#        --caffemodel_dir ./caffemodel/

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_output_images > 0:
        output_image_dir = args.output_dir + 'image_outputs/'
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
    CS = cityscapes(args.cityscapes_dir)
    n_cl = len(CS.classes)
    label_frames = CS.list_label_frames(args.split)
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(args.caffemodel_dir + 'deploy.prototxt',
                    args.caffemodel_dir + 'fcn-8s-cityscapes.caffemodel',
                    caffe.TEST)

    table_path = os.path.join(args.cityscapes_dir, 'table.txt')
    map_source_list = {}
    table = []
    with open(table_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            table.append(line.strip().split(' '))
    for item in table:
        if item[2].endswith('.png'):
            map_source_list[str(item[2].split('/')[-1])] = str(item[0])

    hist_perframe = np.zeros((n_cl, n_cl))
    for i, idx in enumerate(label_frames):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, len(label_frames)))
        city = idx.split('_')[0]
        # idx is city_shot_frame
        label = CS.load_label(args.split, city, idx)
        im_file = args.result_dir + '/' + map_source_list[idx + '_leftImg8bit.png'] + '_fake_B.png'

        im = np.array(Image.open(im_file))
        im = scipy.misc.imresize(im, (label.shape[1], label.shape[2]))
        out = segrun(net, CS.preprocess(im))
        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
        if args.save_output_images > 0:
            label_im = CS.palette(label)
            pred_im = CS.palette(out)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_pred.jpg', pred_im)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_gt.jpg', label_im)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_input.jpg', im)

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    with open(args.output_dir + '/evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))


main()
