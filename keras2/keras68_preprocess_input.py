
from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
img_path = 'D:\study_data\_data\image\dog.PNG'
img = image.load_img(img_path, target_size=(224,224)) #저장한 이미지 크기에 상관없이 224 로 잡혀져 있음
print(img)

x = image.img_to_array(img)
# print("==============image.img_to_array=================")
# print(x, '\n', x.shape) # (224, 224, 3)

#차원늘리기 (이미지 기반 데이터는 4차원으로 늘려줘야함)
x = np.expand_dims(x, axis=0)
print("==============np.expnd_dims(x, axis=0)=================")
print(x, '\n', x.shape) # (224, 224, 3)

x = preprocess_input(x)
print("==============preprocess_input(x)=================")
print(x, '\n', x.shape) # (224, 224, 3)
print(np.min(x),np.max(x)) # -105.779 151.061

print("========= model.predict(x) ===========")
preds = model.predict(x)
print(preds, '\n ', preds.shape)
print("결과는 : ", decode_predictions(preds, top=5)[0])

# ImageNet 데이터셋에 대한 top 매개변수 만큼의 최상위 항목을 반환