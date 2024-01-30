import os

import pandas as pd
import numpy as np
import scipy
import json, argparse
import pickle


def nan_index(df):
    df_nan_upp = df.loc[df['upper_-1'] == True]
    df_nan_low = df.loc[df['lower_-1'] == True]
    df_nan_gender = df.loc[df['gender_-1'] == True]

    df.loc[df_nan_upp.index, 'upper_1':'upper_11'] = -1
    df.loc[df_nan_low.index, 'lower_1':'lower_11'] = -1
    df.loc[df_nan_gender.index, 'gender_0':'gender_1'] = -1

    print("\nsono qui\n")
    df.drop(columns=['upper_-1', 'lower_-1', 'gender_-1'], inplace=True)

    return df


def conversion(filename, type="train", without_nan=False):
    df = pd.read_csv(filename, header=None, names=['filename', 'upper', 'lower', 'gender', 'bag', 'hat'])

    # print(len(df.index))
    # df.replace(-1, pd.NA, inplace=True)
    # print("valori nulli", df.isnull().sum())

    if without_nan:
        all_values = df.loc[(df['upper'] != -1) & (df['lower'] != -1) & (df['gender'] != -1) & (df['bag'] != -1) & (df['hat'] != -1)]
        df = all_values.copy()
        print("without nan")

    df['upper'] = df['upper'].astype('category')
    df['lower'] = df['lower'].astype('category')
    df['gender'] = df['gender'].astype('category')

    one_hot_encoded_color = pd.get_dummies(df, columns=['upper', 'lower', 'gender'])
    print(one_hot_encoded_color.head())
    one_hot_encoded_color = nan_index(one_hot_encoded_color) if without_nan is False and type=="train" else one_hot_encoded_color

    one_hot_encoded_color.replace(True, 1, inplace=True)
    one_hot_encoded_color.replace(False, 0, inplace=True)

    if without_nan is False and type=="train":
        one_hot_encoded_color.replace(-1, 0, inplace=True)
    print(one_hot_encoded_color.head())
    return one_hot_encoded_color


def create_file_mat(dataframe, name_columns):
    color = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]

    up_color = ["up" + c for c in color]
    lower_color = ["lo" + c for c in color]
    attributes = ["bag", "hat"] + up_color + lower_color + ["male", "female"]

    attributes = pd.DataFrame(attributes)
    print("attributes", attributes)
    attributes = np.array(attributes[0]).reshape(len(attributes), 1)

    labels = np.array(dataframe[name_columns])

    images_name = np.array(dataframe['filename']).reshape(1, len(dataframe))
    images_name = images_name.transpose()

    return attributes, labels, images_name

def create_dict_data(dataframe):

    dict_data_filename = pd.DataFrame(dataframe['filename'])
    dict_data = dict()
    for i in dict_data_filename['filename']:
        dict_data[i] = (0, 'front')
    
    for i, key in enumerate(dict_data.keys()):

        print(key, dict_data[key])
        if i == 10:
            break

        i+=1

    return dict_data
    


if __name__ == '__main__':
    """
    0 -> male
    1 -> female
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_nan", default=False)
    parser.add_argument("--data", default=True)
    args = parser.parse_args()


    path = os.path.join(os.getcwd(), "creation_dataset")
    ### Training set
    df_train = conversion(f"{path}/training_set_fixed.csv", without_nan=args.no_nan)

    if args.data:
        dataset = create_dict_data(df_train)
        with open(os.path.join(path, 'dict_data_mivia.pkl'), 'wb+') as f:
            pickle.dump(dataset, f) 
    # columns = df_train.columns[1:]

    # print("\n\n",columns, "\n\n")

    # attributes, train_label, images_name_train = create_file_mat(df_train, columns)
    # print(attributes.shape)
    # print(train_label.shape, type(train_label))
    # print(images_name_train.shape)

    # ### Validation  and Test
    # df_validation = conversion(f"{path}/validation_set.csv", type="val", without_nan=args.no_nan)

    # if args.no_nan:
    #     ### added image to train
    #     df_added_training = df_validation.sample(n = 4000, random_state=4000)
    #     _, added_train_label, added_images_name_train = create_file_mat(df_added_training, columns)

    #     train_label = np.concatenate((train_label, added_train_label), axis=0)
    #     images_name_train = np.concatenate((images_name_train, added_images_name_train), axis=0)

    #     df_validation = df_validation.drop(df_added_training.index)

    # df_test = df_validation.sample(frac=0.3, random_state=200)
    # df_validation = df_validation.drop(df_test.index)
    # _, val_label, images_name_val = create_file_mat(df_validation, columns)
    # print(val_label.shape)
    # print(images_name_val.shape)

    # _, test_label, images_name_test = create_file_mat(df_test, columns)
    # # print(attributes_val.shape)
    # print(test_label.shape)
    # print(images_name_test.shape)

    # data = {'attributes': attributes,
    #         'test_images_name': images_name_test,
    #         'test_label': test_label,
    #         'train_images_name': images_name_train,
    #         'train_label': train_label,
    #         'val_images_name': images_name_val,
    #         'val_label': val_label
    #         }

    # size_data = {
    #     'attributes': attributes.shape,
    #     'test_images_name': images_name_test.shape,
    #     'test_label': test_label.shape,
    #     'train_images_name': images_name_train.shape,
    #     'train_label': train_label.shape,
    #     'val_images_name': images_name_val.shape,
    #     'val_label': val_label.shape
    # }
    # json_file = f"{path}/size_dataset.json"

    # output_file = ""

    # if args.csv:
    #     outfile = os.path.join(path, "")

    # if args.no_nan:
    #     outfile = os.path.join(path, "annotation_without_nan.mat")
    #     json_file = f"{path}/size_dataset_no_nan.json"
    # else:
    #     outfile = os.path.join(path, "annotation_fixed_values.mat")

    # with open(json_file, 'w') as output:
    #     json.dump(size_data, output, indent=4)

    # scipy.io.savemat(outfile, data)

