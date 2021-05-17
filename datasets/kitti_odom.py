from collections import Counter
from functools import partial
import glob
import os
import os.path
import random

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

# Original image size in odometry is [376, 1241]


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    (from https://github.com/nianticlabs/monodepth2)
    """
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file,
                       dtype=float) / 255.0  # scale pixels to the range [0,1]
    img_file.close()
    return rgb_png


def depth_read(filename, proj, shape):
    """Generate depth map from raw velodyne bin file
    Aligned with cam2 now

    Args:
        filename (str): file name of bin file
        proj (numpy array): [3x4] numpy array projecting 3d pts to 2d image coord
        shape (numpy.array): [H, W] image shape

    Returns:
        numpy array: generated depth map
    """

    assert os.path.exists(filename), "file not found: {}".format(filename)

    pts = load_velodyne_points(filename)
    # points in front
    pts = pts[pts[:, 0] > 0]
    # [N, 4] -> [N, 3]
    pts_im = (proj @ pts.T).T
    pts_im[:, :2] = pts_im[:, :2] / pts_im[:, 2][..., np.newaxis]
    pts_im[:, 2] = pts[:, 0]
    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    pts_im[:, 0] = np.round(pts_im[:, 0]) - 1
    pts_im[:, 1] = np.round(pts_im[:, 1]) - 1
    val_inds = (pts_im[:, 0] >= 0) & (pts_im[:, 1] >= 0)
    val_inds = (val_inds & (pts_im[:, 0] < shape[1]) &
                (pts_im[:, 1] < shape[0]))
    pts_im = pts_im[val_inds, :]

    # project to image
    depth = np.zeros(shape)
    ind_x = pts_im[:, 1].astype(np.int)
    ind_y = pts_im[:, 0].astype(np.int)
    depth[ind_x, ind_y] = pts_im[:, 2]
    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, pts_im[:, 1], pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(pts_im[pts[0], 0])
        y_loc = int(pts_im[pts[0], 1])
        depth[y_loc, x_loc] = pts_im[pts, 2].min()
    depth[depth < 0] = 0
    depth = depth[..., np.newaxis]

    return depth


def to_tensor(x):
    return torch.from_numpy(x).to(torch.float)


def train_transform(images, args, shape=None):
    """transform for training

    Args:
        images (list of numpy array): len=2k, [rgb1, sparse1, rgb2, sparse2, ..]
        args (argparser): args
        shape (list, optional): reshape image to fixed size. Defaults to None.

    Returns:
        list of Tensor: list of transformed tensor
    """
    # convert from numpy to torch
    images = list(map(lambda x: to_tensor(x).permute(2, 0, 1), images))
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        transforms.RandomHorizontalFlip(float(do_flip)),
    ])
    # transform sparse
    for i in range(1, len(images), 2):
        images[i] = transform_geometric(images[i])

    if shape is not None:
        transform_geometric = transforms.Compose([
            transform_geometric,
            transforms.Resize(shape),
        ])
    # fix jitter for pair of sample
    jitter = random.randrange(max(0, 1 - args.jitter), 1 + args.jitter)
    jitter_range = (jitter, jitter)
    transform_rgb = transforms.Compose([
        transforms.ColorJitter(jitter_range, jitter_range, jitter_range, 0),
        transform_geometric
    ])
    # transform rgb
    for i in range(0, len(images), 2):
        images[i] = transform_rgb(images[i])
    # sparse = drop_depth_measurements(sparse, 0.9)

    return images


def val_transform(images, args, shape=None):
    images = list(map(lambda x: to_tensor(x).permute(2, 0, 1), images))
    if shape is None:
        return images
    transform = transforms.Compose([
        transforms.Resize(shape),
    ])
    for i in range(0, len(images), 2):
        images[i] = transform(images[i])
    return images


class KittiOdometry(data.Dataset):
    """A data loader for the Kitti dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        self.shape = None
        # Reshape
        self.shape = args.input_size
        assert split in ['train', 'val', 'test']
        if split == 'train':
            self.sequence = range(9)
        elif split == 'val':
            self.sequence = range(9, 11)
        else:
            self.sequence = range(11, 22)

        self.load_calib()
        self.load_gt_pose()
        paths, transform = self.get_paths_and_transform()
        self.paths = paths
        self.transform = transform
        self.threshold_translation = 0.1
        self.max_sample = args.max_sample

        # index: [idx_seq, idx_sample_in_seq]
        self.num_to_index = []
        for seq, seq_list in enumerate(self.paths['rgb']):
            for idx_sample in range(len(seq_list)):
                self.num_to_index.append([seq, idx_sample])

    def __getitem__(self, index):
        seq, sample = self.num_to_index[index]
        # Get different sample
        sample2 = np.random.randint(-2, 3) + sample
        sample2 = max(0, min(sample2, len(self.paths['rgb'][seq]) - 1))
        # seq = 0
        # sample = 0
        # sample2 = sample + 2
        while sample == sample2:
            sample2 = np.random.randint(-2, 3) + sample
            sample2 = max(0, min(sample2, len(self.paths['rgb'][seq]) - 1))

        if self.split == 'test':
            pose1_to_2 = None
        else:
            pose1_to_2 = (torch.inverse(self.gt_poses[seq][sample2])
                          @ self.gt_poses[seq][sample])

        rgb = rgb_read(self.paths['rgb'][seq][sample])
        rgb2 = rgb_read(self.paths['rgb'][seq][sample2])
        inputs_shape = rgb.shape[:2]
        if self.shape is not None:
            inputs_shape = self.shape
        sparse = depth_read(self.paths['lid'][seq][sample],
                            self.velo_to_im2[seq], inputs_shape)
        sparse2 = depth_read(self.paths['lid'][seq][sample2],
                             self.velo_to_im2[seq], inputs_shape)
        rgb, sparse, rgb2, sparse2 = self.transform(
            [rgb, sparse, rgb2, sparse2], self.args)

        input_data = {
            "rgb1": rgb,
            "raw1": sparse,
            "rgb2": rgb2,
            "raw2": sparse2,
            "trs_1_to_2": pose1_to_2,
            "inv_K": to_tensor(self.inv_K[seq]),
            "cam_to_im": to_tensor(self.cam_to_im2[seq])
        }

        return input_data

    def __len__(self):
        return min(self.max_sample, len(self.num_to_index))

    def load_calib(self):
        """
        load from odometry calibration for every sequence
        """
        # self.intrinsic = []
        # self.velo_to_cam2 = []
        self.inv_K = []
        self.velo_to_im2 = []
        self.cam_to_im2 = []
        for seq in self.sequence:
            calib_file = os.path.join(
                self.args.data_folder,
                'odometry/sequences/{:02d}/calib.txt'.format(seq))
            calib = read_calib_file(calib_file)

            width, height = Image.open(
                calib_file.rsplit('/', 1)[0] + '/image_2/000000.png').size
            proj2 = calib["P2"].reshape(3, 4)
            # normed_k = proj2.copy()
            # normed_k[0] = normed_k[0]
            # normed_k[1] = normed_k[1]
            if self.shape is not None:
                # adjust projection matrix to given shape
                proj2[0] = proj2[0] / width * self.shape[1]
                proj2[1] = proj2[1] / height * self.shape[0]
            self.inv_K.append(np.linalg.inv(proj2[:3, :3]))
            self.cam_to_im2.append(proj2)

            # # note: we will take the center crop of the images during augmentation
            # # that changes the optical centers, but not focal lengths
            # # from width = 1242 to 1216, with a 13-pixel cut on both sides
            # proj2[0, 2] = proj2[0, 2] - 13
            # # from width = 375 to 352, with a 11.5-pixel cut on both sides
            # proj2[1, 2] = proj2[1, 2] - 11.5

            velo_to_cam0 = calib["Tr"].reshape(3, 4)
            velo_to_cam0 = np.vstack([velo_to_cam0, [0, 0, 0, 1]])
            # CAM0->CAM2(x += P2[0,3]/P2[0,0])
            T2 = np.eye(4)
            T2[0, 3] = proj2[0, 3] / proj2[0, 0]
            # self.intrinsic.append(K)
            # self.velo_to_cam2.append(T2 @ velo_to_cam0)
            self.velo_to_im2.append(proj2 @ T2 @ velo_to_cam0)

    def load_gt_pose(self):
        """Load gt pose for every used sequence

        Args:
            split (str): 'train' or 'val'
        """
        self.gt_poses = []
        if self.split == 'test':
            return
        for seq in self.sequence:
            self.gt_poses.append([])
            gt_file = os.path.join(self.args.data_folder,
                                   'odometry/poses/{:02d}.txt'.format(seq))
            with open(gt_file, 'r') as all_gt:
                for gt in all_gt:
                    cam0_to_world = list(map(float, gt.split(' ')))
                    cam0_to_world = np.array(cam0_to_world).reshape(3, 4)
                    cam0_to_world = np.vstack([cam0_to_world, [0, 0, 0, 1]])
                    self.gt_poses[-1].append(to_tensor(cam0_to_world))

    def get_paths_and_transform(self):
        if self.split == "train":
            transform = partial(val_transform, shape=self.shape)
        elif self.split in ["val", 'test']:
            transform = partial(val_transform, shape=self.shape)
        else:
            raise ValueError("Unrecognized split " + str(self.split))

        paths_lid = []
        paths_rgb = []
        for seq in self.sequence:
            glob_dir = os.path.join(
                self.args.data_folder,
                'odometry/sequences/{:02d}/image_2/*.png'.format(seq))
            paths_rgb.append(sorted(glob.glob(glob_dir)))
            glob_dir = os.path.join(
                self.args.data_folder,
                'odometry/sequences/{:02d}/velodyne/*.bin'.format(seq))
            paths_lid.append(sorted(glob.glob(glob_dir)))

        paths = {"rgb": paths_rgb, "lid": paths_lid}
        return paths, transform
