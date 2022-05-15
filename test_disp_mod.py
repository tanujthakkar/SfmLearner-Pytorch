import torch

from skimage.transform import resize as imresize
from scipy.ndimage import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import os
from imageio import imread, imsave
from matplotlib import pyplot as plt
import cv2

from models import DispNetS, PoseExpNet
from utils import *

EPSILON = 1E-6

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet0", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-dispnet1", required=False, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'stillbox'])
parser.add_argument("--gps", '-g', action='store_true',
                    help='if selected, will get displacement from GPS for KITTI. Otherwise, will integrate speed')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()

    disp_net0 = DispNetS().to(device)
    weights = torch.load(args.pretrained_dispnet0)
    disp_net0.load_state_dict(weights['state_dict'])
    disp_net0.eval()

    if(args.pretrained_dispnet1):
        disp_net1 = DispNetS().to(device)
        weights = torch.load(args.pretrained_dispnet1)
        disp_net1.load_state_dict(weights['state_dict'])
        disp_net1.eval()

    if args.pretrained_posenet is None:
        print('no PoseNet specified, scale_factor will be determined by median ratio, which is kiiinda cheating\
            (but consistent with original paper)')
        seq_length = 1
    else:
        weights = torch.load(args.pretrained_posenet)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
        pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()

    dataset_dir = Path(args.dataset_dir)
    frames = read_directory(os.path.join(dataset_dir, "Test"))
    gt = read_directory(os.path.join(dataset_dir, "GT"))

    errors = np.zeros((2, 9, len(frames)), np.float32)

    for i, frame in enumerate(tqdm(frames)):

        tgt_img = imread(frame)
        gt_img = imread(gt[i])

        h,w,_ = tgt_img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            tgt_img = imresize(tgt_img, (args.img_height, args.img_width)).astype(np.float32)
            gt_img = imresize(gt_img, (args.img_height, args.img_width)).astype(np.float32)
        tgt_img = np.transpose(tgt_img, (2, 0, 1))

        tgt_tensor = torch.from_numpy(tgt_img).unsqueeze(0)
        tgt_tensor = ((tgt_tensor - 0.5)/0.5).to(device)

        pred_disp0 = disp_net0(tgt_tensor)[0]
        disp0 = (255*tensor2array(pred_disp0, max_value=None, colormap='bone')).astype(np.uint8)
        disp0 = np.transpose(disp0, (1,2,0))[:, :, 0]
        pred_depth0 = 1/(disp0 + EPSILON)

        if(args.pretrained_dispnet1):
            pred_disp1 = disp_net1(tgt_tensor)[0]
            disp1 = (255*tensor2array(pred_disp1, max_value=None, colormap='bone')).astype(np.uint8)
            disp1 = np.transpose(disp1, (1,2,0))[:, :, 0]
            pred_depth1 = 1/(disp1 + EPSILON)

        # gt_depth = compute_depth_from_gt(gt[i])
        gt_depth = gt_img

        fig = plt.figure()

        fig.add_subplot(3, 1, 1)
        plt.imshow(np.transpose(tgt_img, (1,2,0)))
        plt.axis("off")
        plt.axis("tight")
        plt.axis("image")
        fig.tight_layout()

        fig.add_subplot(3, 1, 2)
        plt.imshow(pred_depth0, cmap='gray')
        plt.axis("off")
        plt.axis("tight")
        plt.axis("image")
        fig.tight_layout()

        # fig.add_subplot(3, 1, 3)
        # plt.imshow(pred_depth1, cmap='gray')
        # plt.axis("off")
        # plt.axis("tight")
        # plt.axis("image")
        # fig.tight_layout()

        fig.add_subplot(3, 1, 3)
        plt.imshow(gt_depth, cmap='gray')
        plt.axis("off")
        plt.axis("tight")
        plt.axis("image")
        fig.tight_layout()

        # plt.show()

        scale_factor0 = np.median(gt_depth)/np.median(pred_depth0)
        errors[0,:,i] = compute_errors(1 / scale_factor0 *gt_depth, scale_factor0 * pred_depth0)

        if(args.pretrained_dispnet1):
            scale_factor2 = np.median(gt_depth)/np.median(pred_depth2)
            errors[1,:,i] = compute_errors(1 / scale_factor2 * gt_depth, scale_factor2 * pred_depth2)

    mean_errors = errors.mean(2)
    print(mean_errors[0][:])
    print(mean_errors[1][:])


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_log = np.mean(np.abs(np.log(gt) - np.log(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, abs_log, a1, a2, a3

def compute_depth_from_gt(gt_file):
    gt = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # gt[gt > 0] = (gt[gt > 0] - 1) / 256
    # depth = (0.209313 * 2262.52) / gt

    return depth


if __name__ == '__main__':
    main()
