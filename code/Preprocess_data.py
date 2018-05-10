from os.path import join
from os import listdir

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def save_out(source_image_tensor, tearget_image_tensor, filename):
    image_numpy = tearget_image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_numpy.resize()
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, image_size, masked_size, resize_ratio, crop_mode=True):
        super(DatasetFromFolder, self).__init__()

        self.image_size = image_size
        self.masked_size =masked_size
        self.resize_ratio =resize_ratio
        self.mode = crop_mode

        self.image_path = join(image_dir)
        self.image_filenames = [x for x in listdir(self.image_path) if is_image_file(x)]


        self.reszie = transforms.Compose([transforms.RandomCrop((image_size,image_size))])
        self.centercrop = transforms.Compose([transforms.CenterCrop((image_size, image_size))])
        #self.reszie = transforms.Compose([transforms.Scale((356, 436)), transforms.RandomCrop((image_size,image_size))])
        #self.centercrop = transforms.Compose([transforms.Scale((356, 436)), transforms.CenterCrop((image_size, image_size))])
        self.reszie32 = transforms.Compose([transforms.Scale((image_size // resize_ratio, image_size // resize_ratio))])

        dist_list = [transforms.CenterCrop((masked_size,masked_size)),
                     transforms.Scale((masked_size//resize_ratio, masked_size//resize_ratio)),
                     transforms.Scale((masked_size, masked_size),interpolation=Image.NEAREST)
                     ]
        self.dist_trans = transforms.Compose(dist_list)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.Totensor = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, index):
        # Load Image
        img = load_img(join(self.image_path, self.image_filenames[index]))
        if(self.mode) :
            trans_img = self.reszie(img)
        else :
            trans_img = self.centercrop(img)
        target = self.transform(trans_img)

        target_stage1 = self.transform(self.reszie32(trans_img))

        target_mosaic = trans_img
        disted_img = self.dist_trans(target_mosaic)
        startx = (self.image_size - self.masked_size) // 2
        starty = (self.image_size - self.masked_size) // 2
        target_mosaic.paste(disted_img, (startx, startx))
        target_mosaic = self.transform(target_mosaic)

        blank_img = Image.new("L",(self.masked_size, self.masked_size))
        trans_img.paste(blank_img, (startx, starty))

        mask_img = Image.new("L", (self.image_size, self.image_size), 255)
        mask_img.paste(blank_img, (startx, starty))
        #mask_img.save("maks.png")



        input = self.transform(trans_img)
        input_stage1 = self.transform(self.reszie32(trans_img))

        #mask = torch.zeros(self.image_size, self.image_size)
        #mask_1 = torch.ones(self.masked_size, self.masked_size)
        #mask[startx:startx + self.masked_size, starty:starty + self.masked_size] = mask_1
        #mask = torch.unsqueeze(mask, 0)
        mask = self.Totensor(mask_img)
        input_masked = torch.cat((input, mask), 0)

        mask = self.Totensor(mask_img)
        input_masaic_masked = torch.cat((target_mosaic, mask), 0)

        mask = self.Totensor(self.reszie32(mask_img))
        input_stage1_masked =  torch.cat((input_stage1, mask), 0)

        return input, target, input_masked, target_mosaic, input_stage1, target_stage1, input_stage1_masked, input_masaic_masked

    def __len__(self):
        return len(self.image_filenames)


def get_training_set(root_dir, image_size, masked_size, resize_ratio):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir, image_size, masked_size, resize_ratio)

def get_test_set(root_dir, image_size, masked_size, resize_ratio, randomCrop=False):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir, image_size, masked_size, resize_ratio, randomCrop)