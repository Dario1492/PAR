import os
import pickle
import numpy as np
from easydict import EasyDict
from scipy.io import loadmat

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
group_order = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,1,2,3]
def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    mivia_data = loadmat(os.path.join(save_dir, 'annotation.mat')) #forse da cambiare annotation.mat
    dataset = EasyDict()
    dataset.description = 'mivia'
    dataset.reorder = 'group_order' #forse non serve
    dataset.root = os.path.join(save_dir, 'data')
    train_image_name = [mivia_data['train_images_name'][i][0][0] for i in range(68082)] #da contrallare se vi vuole il for
    val_image_name = [mivia_data['val_images_name'][i][0][0] for i in range(25000)]
    test_image_name = [mivia_data['test_images_name'][i][0][0] for i in range(13162)]
    dataset.image_name = train_image_name + val_image_name + test_image_name
    dataset.label = np.concatenate((mivia_data['train_label'], mivia_data['val_label'], mivia_data['test_label']), axis=0) #controllare axis=0
    dataset.attr_name = [mivia_data['attributes'][i][0][0] for i in range(25)]

   # if reorder:
   #     dataset.label = dataset.label[:, np.array(group_order)]
   #     dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 68082)  # np.array(range(80000))
    dataset.partition.val = np.arange(68082, 93082)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(93082, 105244)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0,93082)  # np.array(range(90000))
    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

    if __name__ == "__main__":

      save_dir = './data/mivia/'
      #reoder = False
      generate_data_description(save_dir, reorder=True)