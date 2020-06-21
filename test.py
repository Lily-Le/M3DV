import os
import numpy as np
import pandas as pd
#import tensorflow as tf
from tensorflow.keras.models import load_model
'''path '''
test_info_path="../input/train-data/test_result.csv"
test_nodule_path="../input/train-data/test/test/"

model1_path="../input/train-data/finalModels/den-seg-base5arg1-epo-14-acc-0.72-val-0.73.hdf5"
model2_path="../input/train-data/finalModels/den-seg-base5arg1-epo-10-acc-0.71-val-0.73.hdf5"
model3_path="../input/train-data/finalModels/den-seg-acc0.70-val0.70-20val0.71.hdf5"

'''path end'''



#处理测试数据-------------------------------------------------------------------------------
test_info=pd.read_csv(test_info_path)
name = test_info.loc[:, 'name']
test_num=len(name)
xs = np.empty((test_num,*(32,32,32), 1))
segs=np.empty((test_num,*(32,32,32), 1))

for i in range(test_num):
        with np.load(os.path.join(test_nodule_path, '%s.npz' % name[i])) as npz:
            voxel=npz['voxel'][34:66,34:66,34:66]
            seg=npz['seg'][34:66,34:66,34:66]
            voxel=np.expand_dims(voxel,axis=-1)
            seg=np.expand_dims(seg,axis=-1)           
            xs[i,]=voxel
            segs[i,]=seg
x_test=xs
x_test_seg=segs
print(x_test.shape)
print('seg shape',x_test_seg.shape)
#np.save('x_test.npy',x_test)
xseg_test=x_test*x_test_seg 
#----------------------------------------------------------------------------------------

#加载模型并预测---------------------------------------------------------------------------

model1=load_model(model1_path)
model2=load_model(model2_path)
model3=load_model(model3_path)

y_pre1=model1.predict(xseg_test)
y_pre2=model2.predict(xseg_test)
y_pre3=model3.predict(xseg_test)
y_pre=(y_pre1+y_pre2+y_pre3)/3
result=y_pre[:,[1]]
#---------------------------------------------------------------------------------------

#写入文件--------------------------------------------------------------------------------

test_num=len(xseg_test)

file_name="submission.csv"
test_list = pd.read_table(test_info_path,sep = ",")['name']
test_name = np.array(test_list).reshape(test_num)

load_predicted = np.array(result).reshape(test_num)
load_test_dict = {'name':test_name, 'Predicted':load_predicted}
load_result = pd.DataFrame(load_test_dict, index = [0 for _ in range(test_num)])
load_result.to_csv(file_name, index = False, sep = ',')
