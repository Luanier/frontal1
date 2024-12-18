"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import streamlit as st
import os
import time
import torch
import numpy as np
from os.path import join, basename
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import AverageMeter
import warnings

warnings.filterwarnings('ignore')

# Define Streamlit App
def main():
    st.title("Image Testing with FFWM")
    
    # Input fields for parameters
    datamode = st.selectbox("Select Data Mode", ["multipie", "lfw"], index=1)
    dataroot = st.text_input("Data Root Directory", "./my_data/")
    preload = st.checkbox("Preload Data", value=True)
    save_image = st.checkbox("Save Test Results", value=False)
    crop = st.checkbox("Center Crop Face (for identity loss)", value=False)
    lightcnn_path = st.text_input("LightCNN Model Path", "./checkpoints/lightCNN_10_checkpoint.pth")
    
    # Run the testing script
    if st.button("Run Test"):
        with st.spinner("Running the test..."):
            try:
                # Set up options
                train_opt = TestOptions()
                train_opt.parser.add_argument('--save_image', action='store_true', help='save test result?')
                train_opt.parser.add_argument('--datamode', type=str, default=datamode, help='data mode: multipie or lfw')
                train_opt.parser.add_argument('--crop', action='store_true', help='center crop face, for calculate the identity loss')
                train_opt.parser.add_argument('--lightcnn', type=str, default=lightcnn_path, help='the path to pretrained lightcnn model')
                
                opt = train_opt.parse()  # get training options
                opt.batch_size = 1
                opt.dataroot = dataroot
                opt.preload = preload
                opt.save_image = save_image
                opt.crop = crop
                
                # Prepare dataset and model
                dataset_val = create_dataset(opt, is_val=True)
                dataset_size_val = len(dataset_val)
                st.write(f'The number of test images: {dataset_size_val}')
                
                torch.set_num_threads(4)
                opt.isTrain = False
                model = create_model(opt)
                model.setup4test(opt)
                visualizer = Visualizer(opt)
                model.set_eval()
                
                if datamode == 'multipie':
                    visual_list_path = join(opt.dataroot, 'multipie', 'test', 'visual_list.npy')
                    if os.path.exists(visual_list_path):
                        visual_list = np.load(visual_list_path)
                    else:
                        visual_list = []

                    visual_list = set(visual_list)
                    gallery_dict = dataset_val.dataset.gallery_dict
                    gallery_keys = list(gallery_dict.keys())
                    gallery_feas = model.get_gallery_fea(gallery_keys, gallery_dict)
                    metric = AverageMeter()
                    
                    for i, data in enumerate(dataset_val):
                        files = data['input_path']
                        model.set_input(data)
                        feas = model.test()
                        model.visual_names = ['img_S', 'img_F', 'fake_F128']
                        metric.update(feas, files, gallery_feas, gallery_keys)
                        
                        for idx, name in enumerate(files):
                            if name in visual_list or (len(visual_list) == 0 and opt.save_image):
                                prefix = os.path.splitext(name)[0]
                                visualizer.display_test_results(model.get_current_visuals(), 0, True, prefix, idx=idx)
                    
                    visualizer.print_test_results(metric)
                else:
                    t = time.time()
                    for i, data in enumerate(dataset_val):
                        files = data['input_path']
                        model.set_input(data)
                        model.test(return_fea=False)
                        
                        for idx, name in enumerate(files):
                            prefix = os.path.splitext(name)[0]
                            visualizer.display_test_results(model.get_current_visuals(), 0, True, prefix, idx=idx)
                    
                    t_end = time.time()
                    st.write(f"Total time: {t_end-t} seconds")
                    st.write(f"Average time: {(t_end-t)/dataset_size_val} seconds/image")
                
                st.success("Test completed successfully!")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
