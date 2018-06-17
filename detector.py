import numpy as np
import pandas as pd


input_data = [[4.0, 2.0, 0.60], [4.2, 2.1, 0.59], [3.9, 2.0, 0.58], [4.3, 2.1, 0.62], [4.1, 2.2, 0.63]]
det_mat=[[3,8],[4,6]]
image_data=[0.1,0.20,0.30]
cov_exdata=[[3,0,2],[2,0,-2],[0,1,1]]

mean_data=np.mean(input_data,axis=0)
print(mean_data)
cov_data=np.cov(np.transpose(input_data))
print(cov_data)
det_mat=np.linalg.det(cov_data)
print(det_mat)
len_image_data=len(image_data)
print(len_image_data)
formula_1=np.subtract(image_data,mean_data)
print(formula_1)
formula_1_T=np.array(formula_1)[np.newaxis]
formula_1_T=formula_1_T.T
print(formula_1_T)
formula_inv=np.linalg.inv(cov_data)
print(formula_inv)
formula_multi=np.dot(formula_1,formula_inv)
print(formula_multi)
formula_multi=np.dot(formula_multi,formula_1_T)
print(formula_multi)
formula_multi=-0.5*formula_multi
print(formula_multi)
exp_data=np.exp(formula_multi)
print(exp_data)
det_cov_product=exp_data*np.power(det_mat,-0.5)
print(det_cov_product)
final_value=((2 *np.pi)**(len_image_data/2))*det_cov_product
print('----------------')
print(final_value)







