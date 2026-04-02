from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator()

data = gen.flow_from_directory('training_Dog_decease_Detection_system/dataset/train')
print("DOG:", data.class_indices)

data = gen.flow_from_directory('training_Cat_decease_Detection_system/dataset/train')
print("CAT:", data.class_indices)

data = gen.flow_from_directory('training_Cow_decease_Detection_s/dataset/train')
print("COW:", data.class_indices)