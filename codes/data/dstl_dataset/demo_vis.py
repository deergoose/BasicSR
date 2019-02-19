import matplotlib.pyplot as plt
import numpy as np

from dataset import DstlDataset


opt = {
    "name": "dstl",
    "mode": "LRHR",
    "dataroot_HR": "/Users/zhoufang/Desktop/lu/dstl",
    "dataroot_LR": None,
    "subset_file": None,
    "use_shuffle": None,
    "n_workers": 8,
    "batch_size": 16,
    "HR_size": 128,
    "use_flip": True,
    "use_rot": True,
    "scale": 4
}

dataset = DstlDataset(opt)

pixel_average = np.zeros(3)
pixel_max = 0

print(len(dataset))
for i in range(len(dataset)):
    im = dataset[i]
    #print im.shape
    #print np.mean(np.mean(im, axis=1), axis=1)
    #print np.max(im)
    #pixel_average += np.mean(np.mean(im, axis=1), axis=1)
    #pixel_max = max(pixel_max, np.max(im))
    
    fig, ax = plt.subplots()
    im = ax.imshow(im)
    plt.show()

print pixel_average/float(len(dataset))
print pixel_max
