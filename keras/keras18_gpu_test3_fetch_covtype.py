#[과제] 속도 비교 과제
#gpu와 cpu

print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus):
    print("쥐피유 돈다")
    aaa = 'gpu'
else:
    print("쥐피유 안돈다") 
    aaa = 'cpu'   