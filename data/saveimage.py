import imageio
import numpy as np
import os


working_dir = os.path.split(os.path.realpath(__file__))[0]
pos_data = np.load(os.path.join(working_dir,'positive_data.npy'))
neg_data = np.load(os.path.join(working_dir,'negitive_data.npy'))

for i in range(40):
    imageio.imwrite(os.path.join(working_dir,'PNG','positive_{}.png'.format(i)),pos_data[i][1])
    imageio.imwrite(os.path.join(working_dir,'PNG','negitive_{}.png'.format(i)),neg_data[i][1])
