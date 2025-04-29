
import Discriminator_SVM
import numpy as np
features=[
        'fin count', 'syn count', 'psh count', 'ack count',
        'rst count', 'urg count', 'packet count', 'pkt_length', 'max pkt_length', 'flow duration', 'min iat',
        'max iat'
        ]
data = np.array([[0,16,60,54,0,0,2,0,1,	0]])

model = Discriminator_SVM.build_discriminator()

predict = model.predict(data)
print(f"predict: {predict}")
print(model.decision_function(data))