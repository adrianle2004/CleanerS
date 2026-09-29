import os
import logging
import numpy as np
from torch.utils.data import Dataset
from ..build import DATASETS
import cv2
import imageio

# <workspace>/data/NYU, next to the repo
_DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *['..'] * 4, 'data', 'NYU')


@DATASETS.register_module()
class NYU(Dataset):
    classes = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objs']
    num_classes = 12
    class2color = {'empty': [22, 191, 206],
                   'ceiling': [214, 38, 40],
                   'floor': [43, 160, 4],
                   'wall': [158, 216, 229],
                   'window': [114, 158, 206],
                   'chair': [204, 204, 91],
                   'bed': [255, 186, 119],
                   'sofa': [147, 102, 188],
                   'table': [30, 119, 181],
                   'tvs': [188, 188, 33],
                   'furn': [255, 127, 12],
                   'objects': [196, 175, 214],
                   'ignore': [153, 153, 153]}
    cmap = [*class2color.values()]

    def __init__(self,
                 data_root: str = _DEFAULT_ROOT,
                 img_H: int = 480,
                 img_W: int = 640,
                 split: str = 'train',
                 transform=None,
                 tsdf_dir: str = 'Custom_TSDF',
                 labelweight_dir: str = 'TSDF',
                 mapping_dir: str = 'Mapping',
                 ):
        """tsdf_dir / labelweight_dir / mapping_dir select which folder each
        array is read from, so the reference and a custom encoder can be
        compared without editing code between runs.

          tsdf_dir        -> arr_0, the MODEL INPUT (the thing under test)
          labelweight_dir -> arr_1, the EVAL MASK   (test_NYU.py:206/210)
          mapping_dir     -> arr_0, the SC selector (mapping == 307200)

        Defaults are the isolated-encoder setup: custom TSDF as input, but the
        reference label_weight and mapping as the yardstick. Pointing all three
        at the custom folders changes the input AND what is being scored, so
        the resulting number is no longer comparable to the published one --
        do that only to validate the full pipeline end to end.
        """

        super().__init__()
        self.split, self.transform, self.data_root = \
            split, transform, data_root
        self.img_H = img_H
        self.img_W = img_W
        self.tsdf_dir = tsdf_dir
        self.labelweight_dir = labelweight_dir
        self.mapping_dir = mapping_dir
        logging.info(f"NYU sources: tsdf={tsdf_dir}, label_weight={labelweight_dir}, "
                     f"mapping={mapping_dir}")

        data_list = os.listdir(os.path.join(data_root, split, 'RGB'))
        self.data_list = [item[:7] for item in data_list if '.png' in item]

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def load_img(self, item):
        rgb_path = os.path.join(self.data_root, self.split, 'RGB', '{}_rgb.png'.format(item))
        img = np.array(cv2.imread(rgb_path), dtype=np.float32)
        return img

    def load_mapping(self, item):
        mapping_path = os.path.join(self.data_root, self.mapping_dir, '{}.npz'.format(item[3:]))
        mapping = np.load(mapping_path)['arr_0'].astype(np.int32)

        mapping2d = (np.ones((self.img_H, self.img_W)) * -1).reshape(-1).astype(np.long)
        mapping2d[mapping[mapping != 307200]] = np.nonzero(mapping != 307200)[0]
        mapping2d = mapping2d.reshape(self.img_H, self.img_W)
        return mapping, mapping2d

    def load_tsdf(self, item, CAD=False):
        tsdf_path = os.path.join(self.data_root, self.tsdf_dir, '{}.npz'.format(item[3:]))

        if CAD:
            # NOTE: only consumed by the teacher during training. VoxelSSC.forward
            # (voxel_seg.py:32) takes img/mapping2d/tsdf and swallows the rest in
            # **kwargs, so at eval this load is dead weight -- kept for parity.
            cad_path = tsdf_path.replace('NYU', 'NYUCAD')
            if os.path.exists(cad_path):
                tsdf_path = cad_path

        tsdf = np.load(tsdf_path)['arr_0'].astype(np.float32).reshape(1, 60, 36, 60)
        return tsdf

    def load_label(self, item):
        label3d_path = os.path.join(self.data_root, 'Label', '{}.npz'.format(item[3:]))
        labelweight_path = os.path.join(self.data_root, self.labelweight_dir,
                                        '{}.npz'.format(item[3:]))

        # Force both arrays to be flat 1D structures (.flatten())
        label3d = np.load(label3d_path)['arr_0'].astype(np.long).flatten()
        label_weight = np.load(labelweight_path)['arr_1'].astype(np.float32).flatten()

        return label3d, label_weight

    def load_depth(self, item, CAD=False):
        data_root = self.data_root.replace('NYU', 'NYUCAD') if CAD else self.data_root
        depth_path = os.path.join(data_root, self.split, 'depth', '{}_0000.png'.format(item))
        depth = imageio.imread(depth_path) / 8000.0
        depth = np.array(depth)
        return depth

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        item = self.data_list[data_idx]

        try:
            img = self.load_img(item)
            label3d, label_weight = self.load_label(item)

            # Notes: the same mapping for both teacher and student models
            mapping, mapping2d = self.load_mapping(item)
            tsdf, tsdf_CAD = self.load_tsdf(item, CAD=False), self.load_tsdf(item, CAD=True)

            data = {'img': img,
                    'label3d': label3d, 'label_weight': label_weight,
                    'mapping': mapping, 'mapping2d': mapping2d,
                    'file': item, 'tsdf': tsdf, 'tsdf_CAD': tsdf_CAD}
            
            # pre-process.
            if self.transform is not None:
                data = self.transform(data)
            return data

        except FileNotFoundError as e:
            # Prints a clear diagnostic message to your console showing the exact missing file
            print(f"\n[WARNING] File missing for sample '{item}'. Details: {e}. Skipping to next scene...")
            
            # Recursive fallback: tries to load the next sample in line instead of crashing
            return self.__getitem__((idx + 1) % len(self.data_idx))

    def __len__(self):
        return len(self.data_idx)
