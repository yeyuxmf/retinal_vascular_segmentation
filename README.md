# retinal_vascular_segmentation
Retinal vascular segmentation，Transformer，CNN， Pytorch,  Python

# Introduce
1. This is a retinal vessel segmentation, although its accuracy surpasses all current papers on the HRF, DRIVE, STARE, and CHASEDB1 datasets, it is not innovative and is a semi-finished paper.
2. The paper only wrote the experimental part and recorded some experimental data in the Word document in the Comparative Literature folder.


# Method
Reference 3D medical image segmentation model:
1. Shaker A, Maaz M, Rasheed H, et al. UNETR++: delving into efficient and accurate 3D medical image segmentation[J]. arXiv preprint arXiv:2212.04497, 2022.   
![企业微信截图_17018485728081](https://github.com/huang229/retinal_vascular_segmentation/assets/29627190/09a2da2f-5509-45a3-bd11-9779dadfab64)

This is the model structure diagram in the UNETR++paper. I directly modified it from 3D to 2D image application.

# Train
python main.py 
# Test 
python test.py  and python test_HRF.py
HRF is a high-resolution dataset with different testing methods.

# Experimental data
![1](https://github.com/huang229/retinal_vascular_segmentation/assets/29627190/7964b7b2-76d6-4085-aa58-993e97ba9276)

![2](https://github.com/huang229/retinal_vascular_segmentation/assets/29627190/421d8ed1-df73-453a-8453-04237b1cefdd)

![3](https://github.com/huang229/retinal_vascular_segmentation/assets/29627190/0b13c26b-a7d9-44fb-983c-1bcdcdb40cec)

![4](https://github.com/huang229/retinal_vascular_segmentation/assets/29627190/8c3fe53f-4eab-42bb-aace-497e8d16d83c)


# License permissions
1. Please comply with the relevant medical image data usage license for usage permissions.
2. Anyone can continue mathematical research on this.















