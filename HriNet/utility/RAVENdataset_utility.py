import os
import glob
import numpy as np
from scipy import misc
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

figure_configuration_names = ['center_single', 'distribute_four', 'distribute_nine', 'in_center_single_out_center_single', 'in_distribute_four_out_center_single', 'left_center_single_right_center_single', 'up_center_single_down_center_single']

class ToTensor(object):
    def __call__(self, sample):
        #return torch.tensor(sample, dtype=torch.float32)
        return torch.from_numpy(sample)

class RAVENdataset(Dataset):
    def __init__(self, root_dir, dataset_type, figure_configurations, img_size, transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = []
        for idx in figure_configurations:
            tmp = [f for f in glob.glob(os.path.join(root_dir, figure_configuration_names[idx], "*.npz")) if dataset_type in os.path.basename(f)]
            self.file_names += tmp
        self.img_size = img_size   
        self.shuffle = shuffle
        self.switch = [3,4,5,0,1,2,6,7]     

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        meta_target = data["meta_target"] 

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = np.arange(8)
            np.random.shuffle(indices)
            new_target = np.where(indices == target)[0][0]
            new_choices = choices[indices, :, :]
            switch_2_rows = np.random.rand()            
            if switch_2_rows < 0.5:                
                context = context[self.switch, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target

        print("target", target)

        resize_image_arr = []
        for idx in range(0, 16):
            img_to_resize = Image.fromarray(image[idx,:,:])
            resized_image = img_to_resize.resize((self.img_size, self.img_size))
            resized_arr = np.array(resized_image)
            resize_image_arr.append(resized_arr)
        
        resize_image_arr = np.stack(resize_image_arr)
    
        del data
        if self.transform:
            resize_image_arr = self.transform(resize_image_arr)
            target = torch.tensor(target, dtype=torch.float32)
            print("target", type(target))
            #target = self.transform(target)
            meta_target = self.transform(meta_target)

        return resize_image_arr, target, meta_target