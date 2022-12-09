# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/7 23:41
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import numpy as np
import nibabel as nib


def load_nii_file(file_path):
    image_obj1 = nib.load(file_path)
    print(f'Type of the image {type(image_obj1)}')

    image_data1 = image_obj1.get_fdata()
    type(image_data1)

    height1, width1, depth1 = image_data1.shape
    print(f"The image object height: {height1}, width:{width1}, depth:{depth1}")

    print(f'image value range: [{image_data1.min()}, {image_data1.max()}]')

    print(image_obj1.header)




if __name__ == '__main__':
    load_nii_file(r"./datasets/NC-release-data/label/1000813648_20180116.nii.gz")



