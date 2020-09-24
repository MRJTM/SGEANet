"""
@File       : dataset_SG.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2019/11/17
@Desc       : This is for generating batch for DA
"""
from src.dataset.data_process_utils import *


# dataset
class SGDataset(Dataset):
    def __init__(self, image_paths, transform=None,train=True,cfg=None,data_copy_num=2):
        random.seed()

        # multiply data into sevaral copys
        image_paths = image_paths*data_copy_num

        self.nSamples = len(image_paths)
        self.image_paths = image_paths
        if not train:
            self.image_paths.sort()
        self.transform = transform
        self.train = train
        self.cfg=cfg


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.image_paths[index]
        """----------load image, counting_gt----------"""
        # load data
        img, synthetic, gt = load_data(img_path,self.cfg['image_folder_name'],
                                       self.cfg['gt_folder_name'],self.cfg['syn_folder_name'],
                                       train=self.train,dataset_type="SG")

        """-----------------------preprocess------------------------"""
        # 1.resize,train model:same resize,test:just resize img
        if self.cfg['fix_shape']:
            target_size=(int(self.cfg['fix_w']),int(self.cfg['fix_h']))
        else:
            target_size = get_target_size(img, short_size=512)

        if self.train:
            img=cv2.resize(img,target_size)
            synthetic=cv2.resize(synthetic,target_size)
            gt=resize_gt(gt,target_size[1],target_size[0])
        else:
            img=cv2.resize(img,target_size)

        """train mode need to data aug"""
        if self.train:
            # 2.crop img and gt
            if self.cfg['crop']:
                crop_H = int(self.cfg['crop_size'])
                crop_W = int(self.cfg['crop_size'])
                x1, y1 = get_crop_start_point(img.shape[0], img.shape[1], crop_H=crop_H, crop_W=crop_W)
                img = img[y1:y1 + crop_H, x1:x1 + crop_W]
                gt = gt[y1:y1 + crop_H, x1:x1 + crop_W]
                synthetic=synthetic[y1:y1 + crop_H, x1:x1 + crop_W]

            # 3.random flip img and gt
            random_flip = np.random.uniform(0, 1)
            if random_flip >= 0.5:
                img = cv2.flip(img, 1)
                gt = cv2.flip(gt, 1)
                synthetic=cv2.flip(synthetic,1)

            # 4.change brightness hue...
            img =Image.fromarray(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)(img)
            img = np.array(img)

        # 5.normalize img
        if self.transform is not None:
            img = self.transform(img)
            if self.train:
                synthetic=self.transform(synthetic)


        # if train mode, gt need to be resize and to tensor
        if self.train:
            # 6.shrink gt
            if int(self.cfg['gt_downsample_rate'])>1:
                gt = resize_gt(gt, h=gt.shape[0] // int(self.cfg['gt_downsample_rate']),
                               w=gt.shape[1] // int(self.cfg['gt_downsample_rate']))

            # 7. gt to tensor
            gt = torch.from_numpy(gt.copy()).float()
            gt = gt.unsqueeze(0)

        if self.train:
            return img, synthetic, gt, img_path
        else:
            return img,gt,img_path