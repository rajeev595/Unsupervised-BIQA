import cv2
import numpy as np
import tensorflow as tf

import random
from random import shuffle

from scipy.io import savemat, loadmat

from alexnet import AlexNet

np.random.seed(2018)
random.seed(2018)

class ImageDataGenerator:

    def __init__(self, 
                 data_dir = '../tid2013/', 
                 inp_size = [227, 227], 
                 scale_size = [256, 256],
                 mean = np.array([104., 117., 124.]),
                 horizontal_flip = True
                 ):
        self.DATA_DIR = data_dir
        self.INP_SIZE = inp_size
        self.SCALE_SIZE = scale_size
        self.mean = mean
        self.horizontal_flip = horizontal_flip
        self.pointer = 0
        self.NUM_EXAMPLES = 24 * 25

        self.get_data_paths()
        self.shuffle_data_paths()
        self.read_mos()
        
    def get_data_paths(self):
        img_paths_ref = [] # A list of paths to reference images.
        img_paths_l1 = []  # A list of paths to images with level-1 distortion.
        img_paths_l2 = []  # A list of paths to images with level-2 distortion.
        img_paths_l3 = []  # A list of paths to images with level-3 distortion.
        img_paths_l4 = []  # A list of paths to images with level-4 distortion.
        img_paths_l5 = []  # A list of paths to images with level-5 distortion.

        for img_no in range(1, 26):
            for distortion in range(1, 25):
                img_path_ref = str("I%.2d.bmp" % img_no)
                img_path_l1 = str("I%.2d_%.2d_%d.bmp" % (img_no, distortion, 1))
                img_path_l2 = str("I%.2d_%.2d_%d.bmp" % (img_no, distortion, 2))
                img_path_l3 = str("I%.2d_%.2d_%d.bmp" % (img_no, distortion, 3))
                img_path_l4 = str("I%.2d_%.2d_%d.bmp" % (img_no, distortion, 4))
                img_path_l5 = str("I%.2d_%.2d_%d.bmp" % (img_no, distortion, 5))

            # Appending the path to list.
                img_paths_ref.append(img_path_ref)
                img_paths_l1.append(img_path_l1)
                img_paths_l2.append(img_path_l2)
                img_paths_l3.append(img_path_l3)
                img_paths_l4.append(img_path_l4)
                img_paths_l5.append(img_path_l5)
                
        self.image_paths = {'ref': img_paths_ref,
                            'level1': img_paths_l1,
                            'level2': img_paths_l2,
                            'level3': img_paths_l3,
                            'level4': img_paths_l4,
                            'level5': img_paths_l5
                            }
        
    def shuffle_data_paths(self):
        indcs = np.arange(self.NUM_EXAMPLES)
        shuffle(indcs)
        
        image_paths_shuffled = {'ref': [], 
                                'level1': [],
                                'level2': [],
                                'level3': [],
                                'level4': [],
                                'level5': []
                               }
        
        for idx in indcs:
            image_paths_shuffled['ref'].append(self.image_paths['ref'][idx])
            image_paths_shuffled['level1'].append(self.image_paths['level1'][idx])
            image_paths_shuffled['level2'].append(self.image_paths['level2'][idx])
            image_paths_shuffled['level3'].append(self.image_paths['level3'][idx])
            image_paths_shuffled['level4'].append(self.image_paths['level4'][idx])
            image_paths_shuffled['level5'].append(self.image_paths['level5'][idx])
            
        self.image_paths = {'ref': image_paths_shuffled['ref'],
                            'level1': image_paths_shuffled['level1'],
                            'level2': image_paths_shuffled['level2'],
                            'level3': image_paths_shuffled['level3'],
                            'level4': image_paths_shuffled['level4'],
                            'level5': image_paths_shuffled['level5']
                           }

    def read_mos(self):
        mos_file = open(self.DATA_DIR + "mos_with_names.txt", "r")

        mos = {}
        for line in mos_file:
            img_score, img_name = line.split()
            mos[img_name] = float(img_score)

        self.mos = mos
            
    def get_next_batch(self, batch_size):
        
        if self.pointer + batch_size < self.NUM_EXAMPLES:
            first = self.pointer
            last = first + batch_size
            self.pointer = last
        else:
            first = self.NUM_EXAMPLES - batch_size
            last = first + batch_size
            self.pointer = 0
            
        self.first = first
        self.last = last

        imgs_batch = {}
        mos_batch = {}
        for i in range(6):
            imgs_batch['level' + str(i)] = np.ndarray([batch_size, self.INP_SIZE[0], self.INP_SIZE[1], 3])
            mos_batch['level' + str(i)] = np.ndarray([batch_size,])

        for i in range(first, last):
            h = np.random.randint(self.SCALE_SIZE[0] - self.INP_SIZE[0], size = 1)[0] # Crop horizontal position.
            w = np.random.randint(self.SCALE_SIZE[1] - self.INP_SIZE[1], size = 1)[0] # Crop vertical position.

            if self.horizontal_flip and np.random.random() < 0.5:
                hflip = 1
            else:
                hflip = 0
            
            for d in range(6):
                if d == 0: # Reference image.
                    img_path = self.DATA_DIR + "reference_images/" + self.image_paths['ref'][i]
                else: # Distorted image.
                    img_path = self.DATA_DIR + "distorted_images/" + self.image_paths['level' + str(d)][i]

                img = cv2.imread(img_path)
                img = cv2.resize(img, tuple(self.SCALE_SIZE))
                img_crop = img[h : h + self.INP_SIZE[0], w : w + self.INP_SIZE[0], :]

                if hflip:
                    img_crop = cv2.flip(img_crop, 1)
            
                j = i - first
                imgs_batch['level' + str(d)][j] = img_crop - self.mean
                
                if d == 0: # Original images.
                    mos_batch['level' + str(d)][j] = 10.
                else:
                    mos_batch['level' + str(d)][j] = self.mos[self.image_paths['level' + str(d)][i]]

        return imgs_batch, mos_batch

######################### ENCODING DATA UTILS ##############################
class EncodingDataGenerator:
    def __init__(self, 
                 encoding_dim, 
                 alexnet_batch_size,
                 saved_encodings
                 ):
        self.NUM_EXAMPLES = 25 * 24
        self.dim = encoding_dim
        self.pointer = 0
        self.num_train = int(0.8 * self.NUM_EXAMPLES)
        self.num_test = self.NUM_EXAMPLES - self.num_train
        self.alexnet_batch_size = alexnet_batch_size
        
        if saved_encodings:
            self.retrieve_encodings()
        else:
            self.get_encodings()
            
        self.get_train_test_split() # Obtaining train and test sets.
        
        print ("No. of reference images used for training: %d" % self.num_train)
        print ("No. of reference images used for testing: %d" % self.num_test)

    def retrieve_encodings(self):
        data = loadmat('encoded_data.mat')
        img_encs = data['encodings']
        mos = data['scores']
        
        self.img_encs = {}
        self.mos = {}
        for d in range(6):
            self.img_encs['level' + str(d)] = \
            img_encs['level' + str(d)][0, 0]
            
            self.mos['level' + str(d)] = \
            mos['level' + str(d)][0, 0].T

    def get_encodings(self):
        train_layers = []
        batch_size = self.alexnet_batch_size

        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])

        model = AlexNet(X, 1.0, 1000, train_layers)
        encoding = model.fc7 # A 4096 x 1 encoding from AlexNet

        # Initializing the data-generator.
        img_utils = ImageDataGenerator(data_dir = "tid2013/", 
                                       inp_size = [227, 227], 
                                       scale_size = [256, 256], 
                                       horizontal_flip = True)

        img_encs = {}
        mos = {}
        for d in range(6):
            img_encs['level' + str(d)] = np.ndarray([self.NUM_EXAMPLES, self.dim])
            mos['level' + str(d)] = np.ndarray([self.NUM_EXAMPLES, ])

        with tf.Session() as sess:
        # Initializing the variables.
            tf.global_variables_initializer()

        # Initializing the weights.
            model.load_initial_weights(sess)

            for _ in range(self.NUM_EXAMPLES // batch_size + 1):
            # Sampling a batch
                imgs_batch, mos_batch = img_utils.get_next_batch(batch_size)
                first = img_utils.first
                last = img_utils.last

                for d in range(6):
                    img_enc = sess.run(encoding,
                                       feed_dict = {X: imgs_batch['level' + str(d)]})
                    img_encs['level' + str(d)][first : last] = img_enc
                    mos['level' + str(d)][first : last] = mos_batch['level' + str(d)]

        sess.close()
        
        self.img_encs = img_encs
        self.mos = mos
        
        # Saving the data.
        data = {
            'encodings': img_encs,
            'scores': mos
        }
        savemat('encoded_data.mat', data)
        
    def get_train_test_split(self):
        
        self.img_encs_train = {}
        self.img_encs_test = {}
        self.mos_train = {}
        self.mos_test = {}
        
        for d in range(6):
            self.img_encs_train['level' + str(d)] = \
            self.img_encs['level' + str(d)][ : self.num_train]
            
            self.img_encs_test['level' + str(d)] = \
            self.img_encs['level' + str(d)][self.num_train: ]
            
            self.mos_train['level' + str(d)] = \
            self.mos['level' + str(d)][: self.num_train]
            
            self.mos_test['level' + str(d)] = \
            self.mos['level' + str(d)][self.num_train: ]
        
    def shuffle_encs(self): # Shuffles the train set.
        indcs = np.arange(self.num_train)
        shuffle(indcs)
        
        img_encs_shuffled = self.img_encs_train
        mos_shuffled =self.mos_train
        for d in range(6):
            img_encs_shuffled['level' + str(d)] = \
            self.img_encs_train['level' + str(d)][indcs]
            
            mos_shuffled['level' + str(d)] = \
            self.mos_train['level' + str(d)][indcs]

        self.img_encs_train = img_encs_shuffled
        self.mos_train = mos_shuffled

    def get_next_batch_train(self, data, batch_size):
        
        if data == 'train':
            q0, q1, q2, q3 = 'level0', 'level1', 'level2', 'level3'
        elif data == 'val':
            q0, q1, q2, q3 = 'level0', 'level1', 'level4', 'level5'
        else:
            print ("ERROR: data should be either train or val")
        
        if self.pointer + batch_size < self.num_train:
            first = self.pointer
            last = self.pointer + batch_size
            self.pointer = last
        else:
            first = self.num_train - batch_size
            last = first + batch_size
            self.pointer = 0
            
        encs_batch = {
            'q0': self.img_encs_train[q0][first : last],
            'q1': self.img_encs_train[q1][first : last],
            'q2': self.img_encs_train[q2][first : last],
            'q3': self.img_encs_train[q3][first : last]
        }
        mos_batch = {
            'q0': self.mos_train[q0][first : last],
            'q1': self.mos_train[q1][first : last],
            'q2': self.mos_train[q2][first : last],
            'q3': self.mos_train[q3][first : last]
        }
        
        return encs_batch, mos_batch
        
    def get_next_batch_test(self, batch_size):
        
        if self.pointer + batch_size < self.num_test:
            first = self.pointer
            last = self.pointer + batch_size
            self.pointer = last
        else:
            first = self.num_test - batch_size
            last = first + batch_size
            self.pointer = 0
            
        self.first = first
        self.last = last

        encs_batch = np.concatenate([
            self.img_encs_test['level1'][first : last],
            self.img_encs_test['level2'][first : last],
            self.img_encs_test['level3'][first : last],
            self.img_encs_test['level4'][first : last],
            self.img_encs_test['level5'][first : last]
            ], axis = 0)
            
        mos_batch = np.concatenate([
            self.mos_test['level1'][first : last],
            self.mos_test['level2'][first : last],
            self.mos_test['level3'][first : last],
            self.mos_test['level4'][first : last],
            self.mos_test['level5'][first : last]
            ], axis = 0)

        return encs_batch, mos_batch