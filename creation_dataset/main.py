import scipy.io
import numpy as np
import pandas as pd

if __name__ == '__main__':


    #f = open("training_set.txt", "r")
#
    #for line in f:
    #    splits = line.split(",")
    #    print(splits)
#
    #f.close()

    df_train = pd.read_csv("training_set.csv", header=None)

    df_validate = pd.read_csv("validation_set.csv", header=None)


    color = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]


    up_color = ["up" + c for c in color]
    lower_color = ["lo" + c for c in color]
    attributes = up_color + lower_color + ["gender", "bag", "hat"]
    #attributes = [[element] for element in attributes]

    attributes = pd.DataFrame(attributes)
    print("attributes",attributes)
    #print(color)
    attributes = np.array(attributes[0]).reshape(25,1)

    #print(attributes)

    #print("4 columns", df_train[[1, 2, 3, 4, 5]])
    
    train_label = np.array(df_train[[1, 2, 3, 4, 5]])

    #attributes = attributes.reshape(1,25)
    print(attributes.shape)
    print(attributes)
    images_name = np.array(df_train[0]).reshape(1,93082)
    images_name = images_name.transpose()
    #images_name = images_name
    print(images_name.shape)
    print(images_name)

    data = {'attributes': attributes,
            'test_images_name': [],
            'test_label': [[0, 0, 0],[1, 1, 1], [2, 2, 2]],
            'train_images_name': images_name,
            'train_label': train_label,
            'val_images_name': [],
            'val_label': [[]]
    }
    
    #attributes = attributes.transpose()
    
    print(df_train[0])
    print(attributes, attributes.shape)
    print(data['train_images_name'].shape)
    print(data["train_images_name"])
    # Specify the file path
    file_path = 'example.mat'

    # Save the data to a .mat file
    scipy.io.savemat(file_path, data)

    print(f'Data has been saved to {file_path}')

