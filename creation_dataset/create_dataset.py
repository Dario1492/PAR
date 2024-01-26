import pandas as pd
import numpy as np
import scipy

def nan_index(df):
    df_nan_upp = df.loc[df['upper_-1'] == True]

    df_nan_low = df.loc[df['lower_-1'] == True]
    df.loc[df_nan_upp.index, 'upper_1':'upper_11'] = -1
    df.loc[df_nan_low.index, 'lower_1':'lower_11'] = -1
    print("\nsono qui\n")
    df.drop(columns=['upper_-1', 'lower_-1'], inplace=True)

    return df

def conversion(filename, type="train"):
    df = pd.read_csv(filename, header=None, names=['filename', 'upper', 'lower', 'gender', 'bag', 'hat'])
    #print(len(df.index))
    #df.replace(-1, pd.NA, inplace=True)
    #print("valori nulli", df.isnull().sum())
    df['upper'] = df['upper'].astype('category')
    df['lower'] = df['lower'].astype('category')

    one_hot_encoded_color = pd.get_dummies(df, columns=['upper', 'lower'])
    print(one_hot_encoded_color.head())
    one_hot_encoded_color = nan_index(one_hot_encoded_color) if type == "train" else one_hot_encoded_color
    print(one_hot_encoded_color.head())
    one_hot_encoded_color.replace(True, 1, inplace=True)
    one_hot_encoded_color.replace(False, 0, inplace=True)

    return one_hot_encoded_color

def create_file_mat(dataframe):
    color = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]

    up_color = ["up" + c for c in color]
    lower_color = ["lo" + c for c in color]
    attributes = ["gender", "bag", "hat"] + up_color + lower_color

    attributes = pd.DataFrame(attributes)
    print("attributes",attributes)
    attributes = np.array(attributes[0]).reshape(25,1)

    train_label = np.array(dataframe[columns])

    images_name = np.array(dataframe['filename']).reshape(1,len(dataframe))
    images_name = images_name.transpose()

    return attributes, train_label, images_name

columns = ["gender", "bag", "hat"]
for i in range(1, 12):
    columns.append("upper_" + str(i))

for i in range(1,12):
    columns.append("lower_" + str(i))

df_train = conversion("training_set.csv")
attributes, train_label, images_name_train = create_file_mat(df_train)
print(attributes.shape)
print(train_label.shape)
print(images_name_train.shape)
df_validation = conversion("validation_set.csv", type="val")

df_test = df_validation.sample(frac=0.3, random_state=200,)
df_validation = df_validation.drop(df_test.index)
_, val_label, images_name_val = create_file_mat(df_validation)
print(val_label.shape)
print(images_name_val.shape)

_, test_label, images_name_test = create_file_mat(df_test)
#print(attributes_val.shape)
print(test_label.shape)
print(images_name_test.shape)

data = {'attributes': attributes,
       'test_images_name': images_name_test,
       'test_label': test_label,
       'train_images_name': images_name_train,
       'train_label': train_label,
       'val_images_name': images_name_val,
       'val_label': val_label
       }

outfile = "annotation_prova.mat"
scipy.io.savemat(outfile, data)