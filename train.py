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
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import os, json

def create_val_functions(opt):
    """
    Creates a list of evaluation functions to be computed at evaluation time. The evaluation functions has each to be passed as a action store true
      argument (default is mse). If a new eval function is implemented, this function must corespondlingly be updated.
    """
    val_functions = []
    if opt.mse_val_function:
        val_functions.append(("mse", mse_val_function)) # when appending your function, append a tuple with (function name (str), function)
    # implement further val functions here if needed
    return val_functions

def mse_val_function(pred, gt):
    return (((gt - pred) ** 2) / gt.numel()).sum()

def save_results(out_dir, results):
    """
    Save dictionary-like objects of the form {epoch: {metric_name: value}}
    at the specified location.

    Args:
        out_dir (str): Directory where the results will be saved.
        results (dict): Dictionary of results in the form {epoch: {metric_name: value}}.
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Convert tensors to serializable types (e.g., float)
    serializable_results = {
        epoch: {metric: (value.item() if isinstance(value, torch.Tensor) else value)
                for metric, value in metrics.items()}
        for epoch, metrics in results.items()
    }

    # Define the output file path
    out_file = os.path.join(out_dir, "results.json")

    # Save the results as a JSON file
    with open(out_file, "w") as f:
        json.dump(serializable_results, f, indent=4)

    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    if opt.use_val_dataset:
        val_dataset = create_dataset(opt, is_validation_data=True)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    val_functions = create_val_functions(opt) # create a list of the val functions to be computed for the evaluations. These are given as store true arguments to the parser.
    val_results = {} # the evaluation metric results, have the form: epoch:metric:value
    best_loss = 99999999

    for epoch in range(opt.epoch_count, opt.n_epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in tqdm(enumerate(dataset)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if opt.use_val_dataset and epoch % opt.eval_epoch_freq == 0:
            # measures MSE over the image
            for eval_func in val_functions:
                val_results.setdefault(epoch, {})
                # Set the specific eval_func[0] key to 0.0
                val_results[epoch][eval_func[0]] = 0.0

            with torch.no_grad():
                for i, data in enumerate(val_dataset):
                    model.set_input(data)
                    pred = model.forward_and_return()
                    gt = data["B"].to("cuda")
                    for eval_func in val_functions:
                        val_results[epoch][eval_func[0]] += eval_func[1](pred, gt)
                print(f"Eval Scores of Epoch {epoch}")
                for eval_func in val_functions:
                    val_results[epoch][eval_func[0]] /= len(val_dataset)
                    for item in val_results[epoch].items():
                        print(f"{item[0]} : {item[1]}")
            if val_results[epoch]['mse'] < best_loss:              # cache our model every <save_epoch_freq> epochs
                print('New Best Model! Epoch %d, MSE %lf' % (epoch, val_results[epoch]['mse']))
                model.save_networks('model_best')
                best_loss = val_results[epoch]['mse']
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))
    
    if opt.use_val_dataset:
        save_results(opt.out_val_results, val_results)






