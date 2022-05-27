import pandas
import numpy as np
import os


batch_size = 50
num_step = 50

source_folder = 'Dynamic model data-20s/Data 0525'
file_list = os.listdir(source_folder)
file_list.sort()
file = np.random.randint(low = 0, high = len(file_list),size = 1)[0]
file = file_list[file]
file = os.path.join(source_folder,file)
df = pandas.read_csv(file)
print(df.loc)

class V_dynamic_data():
    def __init__(self):
        1