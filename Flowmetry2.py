#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ## image_5 (박재윤)

# In[119]:


image_5=[]

for i in range(10):
    image_5.append('image_5_f0{}.tif'.format(i))
    
for i in range(10, 25):
    image_5.append('image_5_f{}.tif'.format(i))
    
image_5


# In[120]:


img=cv2.imread('image_5_f15.tif')
plt.imshow(img)


# In[121]:


img


# In[122]:


img.shape


# In[184]:


result_K=[]
result_BFI=[]

for i in range(25):
    
    #이미지 파일 불러오기
    image_i=cv2.imread(image_5[i]) 
    
    #100 by 100 의 이미지를 추출
    #픽셀 좌표는 (y, x) 의 순서임을 유의
    kernel=image_i[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    #K 와 BFI 계산
    std_result = np.std(re_shape)
    mean_result = np.mean(re_shape)
    K= std_result/mean_result
    BFI= 1/(K*K)
    
    result_K.append(K)
    result_BFI.append(BFI)
    
    print("이미지 {}".format(i))
    print('std_result = ', std_result)
    print('mean_result = ', mean_result)
    print('K = ', K)
    print('BFI = ', BFI)
    print()


# In[187]:


print('result_K = ', result_K)
print()
print('result_BFI = ', result_BFI)
print()
print('K의 최댓값: ', max(result_K), ', K의 최솟값', min(result_K))
plt.plot(result_K)
plt.axvline(x=8, color='r', linestyle='--', linewidth=3)
plt.axvline(x=15, color='r', linestyle='--', linewidth=3)
plt.show()
print('BFI의 최댓값: ', max(result_BFI), ', BFI의 최솟값', min(result_BFI))
plt.plot(result_BFI)
plt.axvline(x=8, color='r', linestyle='--', linewidth=3)
plt.axvline(x=15, color='r', linestyle='--', linewidth=3)
plt.show()

# mean_result-mean_back=0 인 경우가 존재 
# outlier 취급


# ### *Background Subtract

# In[99]:


kernel=np.zeros((100, 100))
result_K=[]
result_BFI=[]


# In[100]:


for i in range(25):
    image_i=pilimg.open(image[i])
    width, height=image_i.size
    image_a=np.zeros((width, height)) #np.zeros 는 0으로 가득 찬 width by height array 를 생성
    
    #이미지 파일 불러오기
    image_i=cv2.imread(image[i]) 
    
    image_a=image_i #각 자리에 값들을 할당
    
    #100 by 100 의 이미지를 추출
    #픽셀 좌표는 (y, x) 의 순서임을 유의
    kernel=image_a[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    for k in range(len(re_shape)):
        re_shape[i]=re_shape[i]-8.5583  ### 왜 값이 변하지 않지?
        
    #K 와 BFI 계산
    std_result = np.std(re_shape)
    mean_result = np.mean(re_shape)
    K= std_result/mean_result
    BFI= 1/(K*K)
    
    result_K.append(K)
    result_BFI.append(BFI)
    
    print("이미지 {}".format(i))
    print('std_result = ', std_result)
    print('mean_result = ', mean_result)
    print('K = ', K)
    print('BFI = ', BFI)
    print()


# In[83]:


# 리스트 값 치환하기

a=[3, 5, 6, 1, 3]

mean_a=np.mean(a)

for i in range(len(a)):
    a[i]=a[i]-mean_a
    
a


# In[102]:


print('result_K = ', result_K)
print()
print('result_BFI = ', result_BFI)
print()
print('K의 최댓값: ', max(result_K), ', K의 최솟값', min(result_K))
plt.plot(result_K)
plt.show()
print('BFI의 최댓값: ', max(result_BFI), ', BFI의 최솟값', min(result_BFI))
plt.plot(result_BFI)
plt.show()


# In[188]:


result_K=[]
result_BFI=[]

for i in range(17):
    image_i=cv2.imread(image_5[i]) 
    
    #100 by 100 의 이미지를 추출
    #픽셀 좌표는 (y, x) 의 순서임을 유의
    kernel=image_i[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    #K 와 BFI 계산
    std_result = np.std(re_shape)
    mean_result = np.mean(re_shape)
    mean_back=8.558
    K= std_result/(mean_result-mean_back)
    BFI= 1/(K*K)
    
    result_K.append(K)
    result_BFI.append(BFI)
    
    print("이미지 {}".format(i))
    print('std_result = ', std_result)
    print('mean_result = ', mean_result)
    print('K = ', K)
    print('BFI = ', BFI)
    print()


# In[189]:


print('result_K = ', result_K)
print()
print('result_BFI = ', result_BFI)
print()
print('K의 최댓값: ', max(result_K), ', K의 최솟값', min(result_K))
plt.plot(result_K)
plt.axvline(x=8, color='r', linestyle='--', linewidth=3)
plt.axvline(x=15, color='r', linestyle='--', linewidth=3)
plt.show()
print('BFI의 최댓값: ', max(result_BFI), ', BFI의 최솟값', min(result_BFI))
plt.plot(result_BFI)
plt.axvline(x=8, color='r', linestyle='--', linewidth=3)
plt.axvline(x=15, color='r', linestyle='--', linewidth=3)
plt.show()

# mean_result-mean_back=0 인 경우가 존재 
# outlier 취급


# ## image_0 (양규원)

# In[177]:


#C:/Users/양규원/Desktop/flowmetry./

image_0= []

for i in range(10):
    image_0.append('image_0_f0{}.tif'.format(i))
    
for j in range(10, 25):
    image_0.append('image_0_f{}.tif'.format(j))
    
image_0


# In[178]:


result_K=[]
result_BFI=[]

for i in range(25):
    image_i=pilimg.open(image_0[i])
    width, height=image_i.size
    image_a=np.zeros((width, height)) #np.zeros 는 0으로 가득 찬 width by height array 를 생성
    
    #이미지 파일 불러오기
    image_i=cv2.imread(image_0[i]) 
    
    image_a=image_i #각 자리에 값들을 할당
    
    #100 by 100 의 이미지를 추출
    #픽셀 좌표는 (y, x) 의 순서임을 유의
    kernel=image_a[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    #K 와 BFI 계산
    std_result = np.std(re_shape)
    mean_result = np.mean(re_shape)
    K= std_result/mean_result
    BFI= 1/(K*K)
    
    result_K.append(K)
    result_BFI.append(BFI)
    
    print("이미지 {}".format(i))
    print('std_result = ', std_result)
    print('mean_result = ', mean_result)
    print('K = ', K)
    print('BFI = ', BFI)
    print()


# In[179]:


print('result_K = ', result_K)
print()
print('result_BFI = ', result_BFI)
print()
print('K의 최댓값: ', max(result_K), ', K의 최솟값', min(result_K))
plt.plot(result_K)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()
print('BFI의 최댓값: ', max(result_BFI), ', BFI의 최솟값', min(result_BFI))
plt.plot(result_BFI)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()


# ### *Background Subtract

# In[180]:


result_K=[]
result_BFI=[]

for i in range(25):
    image_i=cv2.imread(image_0[i]) 
    
    #100 by 100 의 이미지를 추출
    #픽셀 좌표는 (y, x) 의 순서임을 유의
    kernel=image_i[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    #K 와 BFI 계산
    std_result = np.std(re_shape)
    mean_result = np.mean(re_shape)
    mean_back=8.558
    K= std_result/(mean_result-mean_back)
    BFI= 1/(K*K)
    
    result_K.append(K)
    result_BFI.append(BFI)
    
    print("이미지 {}".format(i))
    print('std_result = ', std_result)
    print('mean_result = ', mean_result)
    print('K = ', K)
    print('BFI = ', BFI)
    print()


# In[181]:


print('result_K = ', result_K)
print()
print('result_BFI = ', result_BFI)
print()
print('K의 최댓값: ', max(result_K), ', K의 최솟값', min(result_K))
plt.plot(result_K)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()
print('BFI의 최댓값: ', max(result_BFI), ', BFI의 최솟값', min(result_BFI))
plt.plot(result_BFI)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()


# ## image_2 (윤수연)

# In[31]:


image_2= []

for i in range(10):
    image_2.append('image_2_f0{}.tif'.format(i))
    
for j in range(10, 25):
    image_2.append('image_2_f{}.tif'.format(j))
    
image_2


# In[32]:


kernel=np.zeros((100, 100))
result_K=[]
result_BFI=[]


# In[33]:


for i in range(25):
    image_i=pilimg.open(image_2[i])
    width, height=image_i.size
    image_a=np.zeros((width, height)) #np.zeros 는 0으로 가득 찬 width by height array 를 생성
    
    #이미지 파일 불러오기
    image_i=cv2.imread(image_2[i]) 
    
    image_a=image_i #각 자리에 값들을 할당
    
    #100 by 100 의 이미지를 추출
    #픽셀 좌표는 (y, x) 의 순서임을 유의
    kernel=image_a[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    #K 와 BFI 계산
    std_result = np.std(re_shape)
    mean_result = np.mean(re_shape)
    K= std_result/mean_result
    BFI= 1/(K*K)
    
    result_K.append(K)
    result_BFI.append(BFI)
    
    print("이미지 {}".format(i))
    print('std_result = ', std_result)
    print('mean_result = ', mean_result)
    print('K = ', K)
    print('BFI = ', BFI)
    print()


# In[174]:


print('result_K = ', result_K)
print()
print('result_BFI = ', result_BFI)
print()
print('K의 최댓값: ', max(result_K), ', K의 최솟값', min(result_K))
plt.plot(result_K)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()
print('BFI의 최댓값: ', max(result_BFI), ', BFI의 최솟값', min(result_BFI))
plt.plot(result_BFI)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()


# ### *Background Subtract

# In[175]:


result_K=[]
result_BFI=[]

for i in range(25):
    image_i=cv2.imread(image_2[i]) 
    
    #100 by 100 의 이미지를 추출
    #픽셀 좌표는 (y, x) 의 순서임을 유의
    kernel=image_i[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    #K 와 BFI 계산
    std_result = np.std(re_shape)
    mean_result = np.mean(re_shape)
    mean_back=8.558
    K= std_result/(mean_result-mean_back)
    BFI= 1/(K*K)
    
    result_K.append(K)
    result_BFI.append(BFI)
    
    print("이미지 {}".format(i))
    print('std_result = ', std_result)
    print('mean_result = ', mean_result)
    print('K = ', K)
    print('BFI = ', BFI)
    print()


# In[176]:


print('result_K = ', result_K)
print()
print('result_BFI = ', result_BFI)
print()
print('K의 최댓값: ', max(result_K), ', K의 최솟값', min(result_K))
plt.plot(result_K)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()
print('BFI의 최댓값: ', max(result_BFI), ', BFI의 최솟값', min(result_BFI))
plt.plot(result_BFI)
plt.axvline(x=6, color='r', linestyle='--', linewidth=3)
plt.axvline(x=12, color='r', linestyle='--', linewidth=3)
plt.show()


# ## image_3  (Background Image)

# In[198]:


image_3= []

for i in range(10):
    image_3.append('image_3_f0{}.tif'.format(i))
    
image_3


# In[199]:


img_3_00=cv2.imread('image_3_f00.tif')
img_3_00


# In[200]:


img_3_00.shape


# In[201]:


#average background level 8.5583

mean=[]

for i in range(10):
    
    #이미지 파일 불러오기
    img=cv2.imread(image_3[i]) 
    
    #이미지 자르기
    img_seg=img[80:180, 100:200] 
    re_shape=kernel.reshape(-1,) #변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
    
    #mean 계산
    mean_result=np.mean(re_shape)
    
    mean.append(mean_result)
    
mean

#왜 구할때마다 값이 달라지지?


# In[ ]:




