import os
import glob
import pandas as pd
path = '/home/dario1492/Desktop/training_set/training_set'
extension = 'jpg'
project_path = os.getcwd()
os.chdir(path)
print(os.getcwd())
result = glob.glob(f'*.jpg')
print(len(result))

#compare
os.chdir(project_path)
print(os.getcwd())
df = pd.read_csv("creation_dataset/training_set.csv", header=None, names=['filename', 'upper', 'lower', 'gender', 'bag', 'hat'])

filename = df['filename']
print(filename.shape, type(filename))

set_pandas = set(filename.unique())
print(type(set_pandas), len(set_pandas))
set_result = set(result)
print(type(result), len(set_result), result[:10])

differences = set_pandas ^ set_result

print("\n\n", differences)


