import argparse
import cv2
import numpy as np
import yaml
import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from datasets.kitti_odom import KittiOdometry
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2


def depth_to_space(data, block_size=8):
    # [B, N, h, w] -> [B, h, 2, N]
    merge = data.permute(1, 2, 3, 0)
    # -> [h, w, 8, 8, B]
    new_shape = (block_size, block_size) + merge.shape[1:]
    merge = merge.reshape(new_shape).permute(2, 3, 0, 1, 4)
    # -> [8, H, w, B]
    merge = torch.cat(list(merge), dim=1)
    # ->[H, W, B]
    merge = torch.cat(list(merge), dim=1)
    merge = merge.permute(2, 0, 1)
    return merge


def getPtsFromHeatmap(heatmap, conf_thresh=0.015):
    '''
    :param heatmap:
        np (H, W)
    :return:
    '''
    heatmap = heatmap.squeeze()
    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys  # abuse of ys, xs
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
    pts, _ = nms_fast(pts, H, W, dist_thresh=4)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = 20
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < 100 + bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def vis_result(image, wait_key=0, window_name='vis', title=''):
    """[summary]

    Args:
        image (np.array): [HxWxC] image of numpy array(OpenCV)
        wait_key(float): wait time for opencv window ([0] ms)

    Returns:
        [type]: [description]
    """

    # Visualize
    if not isinstance(image, np.ndarray):
        image = image.cpu().numpy()
    image = image.copy()

    font_scale = 0.5 if image.shape[1] < 100 else 2
    font_thick = 1 if image.shape[1] < 100 else 2
    org = (int(font_scale * 10), int(font_scale * 30))
    if isinstance(title, str) and len(title) > 0:
        cv2.putText(image,
                    text=title,
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=font_scale,
                    color=(0, 0, 255),
                    thickness=font_thick)

    if wait_key >= 0:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(wait_key)
        if key == 113:
            exit()
    return image


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
        3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
        0 : Empty or suppressed.
        1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
        in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
        nmsed_corners - 3xN numpy matrix with surviving corners.
        nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
        desc1 - MxN numpy matrix of N corresponding M-dimensional descriptors.
        desc2 - MxN numpy matrix of N corresponding M-dimensional descriptors.
        nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
        matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    # if desc1.shape[1] == 0 or desc2.shape[1] == 0:
    #     return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


def sample_desc_from_points(coarse_desc, pts):
    # --- Process descriptor.
    H, W = coarse_desc.shape[2] * 8, coarse_desc.shape[3] * 8
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.cuda()
        desc = torch.nn.functional.grid_sample(coarse_desc,
                                               samp_pts,
                                               align_corners=True)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc


def main():
    parser = argparse.ArgumentParser('test kitti odom')
    parser.add_argument('--data', dest='data_folder', type=str, default='')
    parser.add_argument('--kp_weight',
                        type=str,
                        default='',
                        help='path to keypoint model weight')
    parser.add_argument('--input_size',
                        type=str,
                        default='352x1216',
                        choices=['192x640', '352x1216'],
                        help='reshape image input if given [HxW]')
    parser.add_argument('--max_sample',
                        type=int,
                        default=1e10,
                        help='max number of samples of dataset[1e10]')
    args = parser.parse_args()
    args.input_size = list(map(int, args.input_size.split('x')))
    config = yaml.load('configs/superpoint_kitti_export.yaml',
                       Loader=yaml.FullLoader)
    print("check config!! ", config)
    model = SuperPointNet_gauss2()
    if os.path.isfile(args.kp_weight):
        print('Loading checkpoint', args.kp_weight)
        checkpoint = torch.load(args.kp_weight)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    dataset = KittiOdometry('train', args)
    loader = DataLoader(dataset)
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10,
                             (1216, 704))
    rgb_to_gray = transforms.Grayscale()
    prev_desc = None
    prev_img = None
    for data in loader:
        for key in data.keys():
            data[key] = data[key].cuda()

        img_gray = rgb_to_gray(data['rgb1'])
        # dense_d, conf = depth({'rgb': data['rgb1'], 'raw': data['raw1']})
        # out = model(dense_d.unsqueeze(1) / dense_d.max())
        out = model(img_gray)

        space = F.softmax(out['semi'], dim=1)
        # exclude dustbin for no keypoint
        # [B, C, H, W]
        merge = space[:, :-1]
        merge = depth_to_space(space[:, :-1]).detach().cpu().numpy()
        pts = getPtsFromHeatmap(merge[0])

        desc = out['desc']
        desc = sample_desc_from_points(desc, pts)

        img = data['rgb1'][0].detach().cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # # show depth
        # img = vis_depth(dense_d[0].detach().cpu().numpy(),
        #                 img,
        #                 weight_depth=.5,
        #                 scale=80,
        #                 wait_key=-1)
        # draw keypoints
        for idx in range(pts.shape[-1]):
            xx = int(pts[0, idx])
            yy = int(pts[1, idx])
            img = cv2.circle(img,
                             center=(xx, yy),
                             radius=2,
                             color=(0, 0, 255),
                             thickness=1)

        if prev_desc is None:
            prev_desc = desc
            prev_img = img
            prev_kp = pts
            continue
        matches = nn_match_two_way(prev_desc, desc, nn_thresh=1)

        vis_img = np.concatenate([prev_img, img], axis=0)
        # draw matches
        for m_idx in range(matches.shape[1]):
            i1 = int(matches[0, m_idx])
            i2 = int(matches[1, m_idx])
            x1 = int(prev_kp[0, i1])
            y1 = int(prev_kp[1, i1])
            x2 = int(pts[0, i2])
            y2 = int(pts[1, i2]) + img.shape[0]
            vis_img = cv2.line(vis_img,
                               pt1=(x1, y1),
                               pt2=(x2, y2),
                               color=(0, 255, 0),
                               thickness=1)

        vis_img = vis_result(vis_img)
        writer.write(vis_img)

        prev_desc = desc
        prev_img = img
        prev_kp = pts
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)
        # cv2.waitKey()


if __name__ == '__main__':

    with torch.no_grad():
        main()
