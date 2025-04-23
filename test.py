
import Discriminator_SVM
import numpy as np

data = np.array([[ 0, 4,0,12,0,5,2,5,  60,10,0,1]])

model = Discriminator_SVM.build_discriminator()

predict = model.predict(data)
print(f"predict: {predict}")