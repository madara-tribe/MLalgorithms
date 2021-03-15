import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Sequence_5DGenerator(object):
    def __init__(self, image, nt, batch_size, output_mode="train", color=True, shuffle=True):
        self.image = image
        self.H = image.shape[1]
        self.W = image.shape[2]
        self.nt = nt
        self.batch_size= batch_size
        self.color = color
        self.index_array = None
        assert output_mode in {'train', 'prediction'}, 'output_mode must be {error or prediction}'
        self.output_mode = output_mode
        if shuffle:
            possible_start = np.array([i for i in range(self.image.shape[0]-self.nt) if i%self.nt==0])
            self.possible_starts = np.random.permutation(possible_start)
        else:
            self.possible_starts = np.array([i for i in range(self.image.shape[0]-self.nt) if i%self.nt==0])
        self.num_5dtensor = len(self.possible_starts)
        
    def preprocess(self, X):
        return X.astype(np.float32)/255

    def use_all_img(self):
        if self.color:
            X_all = np.zeros((self.num_5dtensor, self.nt) + (self.H, self.W, 3), np.float32)
        else:
            X_all = np.zeros((self.num_5dtensor, self.nt) + (self.H, self.W, 1), np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.image[idx:idx+self.nt])
        return X_all

    def flow_from_img(self):
        if self.color:
            batch_x = np.zeros((self.batch_size, self.nt) + (self.H, self.W, 3), np.float32)
        else:
            batch_x = np.zeros((self.batch_size, self.nt) + (self.H, self.W, 1), np.float32)
        while True:
            self.index_array = np.array([self.possible_starts[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(self.possible_starts))])
            for total, idxs in enumerate(self.index_array):
                for i, idx in enumerate(idxs):
                    batch_x[i] = self.preprocess(self.image[idx:idx+self.nt])
                if len(batch_x)==self.batch_size:
                    input_img=batch_x
                    # plt.imshow(input_img[1][1].reshape(128, 160), "gray"),plt.show()
                    if self.output_mode == 'train':
                        input_label = np.zeros(self.batch_size, np.float32)
                    elif self.output_mode == 'prediction':
                        input_label = input_img
                    yield input_img, input_label

def get_5dloader(img_dir, load=True):
    images = [cv2.imread(os.path.join(img_dir, f)) for idx, f in enumerate(os.listdir(img_dir)) if idx<100]
    images = np.array([cv2.resize(img, (224, 224)) for img in images])
    print(images.shape)
    if load:
        loader = Sequence_5DGenerator(images, nt=24, batch_size=4, color=True).flow_from_img()
    else:
        loader = Sequence_5DGenerator(images, nt=24, batch_size=4, color=True).use_all_img()
    return loader

if __name__=='__main__':
    img_dir='/Users/train/'
    #loaders = get_5dloader(img_dir, load=None)
    #for img in loaders:
        #print(img.shape)
    loaders = get_5dloader(img_dir, load=True)
    for img, label in loaders:
        print(img.shape, label.shape)
    