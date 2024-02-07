import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class RvssDataset(Dataset):
    def __init__(self, path, hist_len=1):
        super().__init__()
        self.path = path
        self.hist_len = hist_len
        self._prep(path)

    def _prep(self, path):
        file_metadata = {}
        num_stat = []
        global_index = 0
        for file in os.listdir(path):
            if file.startswith("."):
                continue
            demonstration_path = "{}/{}".format(path, file)
            file_names = ["{}/{}".format(demonstration_path, name) for name in \
                          os.listdir(demonstration_path) if name.endswith("jpg")]
            file_names.sort(key=os.path.getctime)

            file_metadata[global_index] = [file_path for file_path in file_names]
            num_stat.append(global_index)
            global_index += len(file_names) - self.hist_len + 1
        self.metadata = file_metadata
        self.num_stat = num_stat

    def __len__(self):
        return self.num_stat[-1]

    def search_index(self, index):
        # print(index, self.num_stat)
        idx = np.searchsorted(self.num_stat, index, side="right") - 1
        floor_value = self.num_stat[idx]
        return idx, floor_value
    
    def extract_action(self, path_string):
        res = path_string.rsplit("/", 1)[1].rsplit(".", 1)[0][6:].strip("_")
        return res
    
    def decode_image_action(self, index):
        traj_index, floor_cnt = self.search_index(index)
        # print("traj_index", traj_index)
        # print("index", index)
        image_index = index  - floor_cnt + self.hist_len
        # print("image index", image_index)
        image_paths = self.metadata[floor_cnt][(image_index - self.hist_len):image_index]
        im_arrs = []
        act_arrs = []
        for image_path in image_paths:
            im_arr = cv2.imread(image_path)
            im_arr = np.expand_dims(im_arr, axis=0)
            im_arrs.append(im_arr)
        # print(image_path)
            action_signal = np.array(float(self.extract_action(image_path))).reshape(1, )
            act_arrs.append(action_signal)
        # print(len(im_arrs))
        im_arrs = np.concatenate(im_arrs, axis=0)
        act_arrs = np.concatenate(act_arrs, axis=0)
        return im_arrs, act_arrs

    def __getitem__(self, index):
        return self.decode_image_action(index)

if __name__ == "__main__":
    directory = "/Users/jy/Documents/PD/research/projects/rvss/train"
    rvsd = RvssDataset(path=directory, hist_len=1)
    print(len(rvsd))
    print(rvsd.num_stat)
    for idx in range(len(rvsd)):
        print("========= {}/{} ============".format(idx, len(rvsd)))
        arr = rvsd[idx]
    print([tmp_arr.shape for tmp_arr in arr])