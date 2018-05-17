#!/usr/bin python3
""" The script to run the training process of faceswap """

import os
import sys
import threading

import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from DNNTraining import Train_DNN

from lib.training_data import add_images_to_numpy_array

from lib.utils import get_folder, get_image_paths, set_system_verbosity


class Train(object):
    """ The training process.  """
    def __init__(self, arguments):
        self.args = arguments
        self.images = self.get_images()
        self.warped_images = []
        self.original_images = []
        for image_list in self.images:
            warped_image_batch, original_image_batch = add_images_to_numpy_array(image_list)
            self.warped_images.append(warped_image_batch)
            self.original_images.append(original_image_batch)
        self.stop = False
        self.save_now = False
        self.preview_buffer = dict()
        self.lock = threading.Lock()

    def process(self):
        """ Call the training process object """
        print("Training data directory: {}".format(self.args.model_dir))
        lvl = '0' if self.args.verbose else '2'
        set_system_verbosity(lvl)
        thread = self.start_thread()

        if self.args.preview:
            self.monitor_preview()
        else:
            self.monitor_console()

        self.end_thread(thread)

    def get_images(self):
        """ Check the image dirs exist, contain images and return the image
        objects """
        images = []
        if self.args.pre_training:
            if not os.path.isdir(self.args.input_P):
                print('Error: {} does not exist'.format(self.args.input_P))
                exit(1)
            images.append(get_image_paths(self.args.input_P))
            print("Model Pre-Training Directory: {}".format(self.args.input_P))
        else:
            for image_dir in [self.args.input_A, self.args.input_B]:
                if not os.path.isdir(image_dir):
                    print('Error: {} does not exist'.format(image_dir))
                    exit(1)

                if not os.listdir(image_dir):
                    print('Error: {} contains no images'.format(image_dir))
                    exit(1)

                images.append(get_image_paths(image_dir))
            print("Model A Directory: {}".format(self.args.input_A))
            print("Model B Directory: {}".format(self.args.input_B))
        return images

    def start_thread(self):
        """ Put the training process in a thread so we can keep control """
        thread = threading.Thread(target=self.process_thread)
        thread.start()
        return thread

    def end_thread(self, thread):
        """ On termination output message and join thread back to main """
        print("Exit requested! The trainer will complete its current cycle, "
              "save the models and quit (it can take up a couple of seconds "
              "depending on your training speed). If you want to kill it now, "
              "press Ctrl + c")
        self.stop = True
        thread.join()
        sys.stdout.flush()

    def process_thread(self):
        """ The training process to be run inside a thread """
        try:
            print("Loading data, this may take a while...")

            if self.args.allow_growth:
                self.set_tf_allow_growth()
            for dimm in [256, 512, 1024]:
                for lr in [5e-4, 5e-5, 5e-6]:
                    trainer = Train_DNN(DIM_ENCODER=dimm, lr=lr)
                    trainer.preTraining(self.warped_images[0], self.original_images[0], self.args.batch_size)
                    del(trainer)
            # trainer = Train_DNN()
            # if (self.args.pre_training):
            #     trainer.preTraining(self.warped_images[0], self.original_images[0])
            # else:
            #     trainer.train_on_A_and_B(self.warped_images[0], self.original_images[0], self.warped_images[1],
            #                              self.original_images[1])  # Here the actual training starts!

        except KeyboardInterrupt:
            print("Training was cancelled by the user!")
            exit(0)
        except Exception as err:
            raise err

    def monitor_preview(self):
        """ Generate the preview window and wait for keyboard input """
        print("Using live preview.\n"
              "Press 'ENTER' on the preview window to save and quit.\n"
              "Press 'S' on the preview window to save model weights "
              "immediately")
        while True:
            try:
                with self.lock:
                    for name, image in self.preview_buffer.items():
                        cv2.imshow(name, image)

                key = cv2.waitKey(1000)
                if key == ord("\n") or key == ord("\r"):
                    break
                if key == ord("s"):
                    self.save_now = True
                if self.stop:
                    break
            except KeyboardInterrupt:
                break

    @staticmethod
    def monitor_console():
        """ Monitor the console for any input followed by enter or ctrl+c """
        print("Starting. Press 'ENTER' to stop training and save model")
        try:
            input()
        except KeyboardInterrupt:
            pass

    @staticmethod
    def set_tf_allow_growth():
        """ Allow TensorFlow to manage VRAM growth """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))

    def show(self, image, name=""):
        """ Generate the preview and write preview file output """
        try:
            scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self.args.write_image:
                img = "_sample_{}.jpg".format(name)
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)

            if self.args.redirect_gui:
                img = ".gui_preview.png"
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)
            elif self.args.preview:
                with self.lock:
                    self.preview_buffer[name] = image
        except Exception as err:
            print("could not preview sample")
            raise err
