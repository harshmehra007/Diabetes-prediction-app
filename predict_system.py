import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.sav','rb'))


input_data = (4,110,92,0,0,36.6,0.191,30)

input_data_array = np.asarray(input_data)

input_reshape = input_data_array.reshape(1,-1)

pridiction  = loaded_model.predict(input_reshape)
print(pridiction)

if (pridiction[0] == 0):
    print('sugar nhi h')
else:
    print('sugar h')
