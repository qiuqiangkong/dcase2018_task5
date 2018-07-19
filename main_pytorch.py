import os
import numpy as np
import argparse
import h5py
import math
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_accuracy, calculate_f1_score)
from models import move_data_to_gpu, init_layer, init_bn, BaselineCnn
import config


def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate. 
    
    Args:
      model: object
      generator: object
      data_type: string, 'train' | 'validate'
      max_iteration: int, maximum iteration for validation
      cuda: bool
      
    Returns:
      accuracy: float
      f1_score: float
      loss: float
    """

    generate_func = generator.generate_validate(data_type=data_type, 
                                                max_iteration=max_iteration)

    # Inference
    (outputs, targets, audio_names) = forward(model=model,
                                              generate_func=generate_func,
                                              cuda=cuda, 
                                              has_target=True)

    # Statistics
    loss = F.nll_loss(torch.Tensor(outputs), torch.LongTensor(targets)).numpy()
    loss = float(loss)

    predictions = np.argmax(outputs, axis=-1)

    accuracy = calculate_accuracy(targets, predictions)

    f1_score = calculate_f1_score(targets, predictions, average='macro')

    return accuracy, f1_score, loss


def forward(model, generate_func, cuda, has_target):
    """Forward data to a model.

    Args:
      model: object
      generate_func: function
      cuda: bool
      has_target: bool
      
    Returns:
      (outputs, targets, audio_names) | (outputs, audio_names)
    """

    outputs = []
    targets = []
    audio_names = []    

    iteration = 0

    # Evaluate on mini-batch
    for data in generate_func:

        if has_target:
            (batch_x, batch_y, bach_audio_names) = data
            targets.append(batch_y)

        else:
            (batch_x, bach_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        batch_output = model(batch_x)
        batch_output = batch_output.data.cpu().numpy()

        outputs.append(batch_output)
        audio_names.append(bach_audio_names)

        iteration += 1

    outputs = np.concatenate(outputs, axis=0)
    audio_names = np.concatenate(audio_names, axis=0)
    
    if has_target:
        targets = np.concatenate(targets, axis=0)
        return outputs, targets, audio_names
        
    else:
        return outputs, audio_names


def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    validate = args.validate
    holdout_fold = args.holdout_fold
    mini_data = args.mini_data
    cuda = args.cuda
    filename = args.filename

    batch_size = 64
    classes_num = len(config.labels)
    max_validate_iteration = 100

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'mini_development.h5')
        
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel',
                                 'development.h5')
                                 
    if validate:
        train_txt = os.path.join(dataset_dir, 'evaluation_setup', 
                                'fold{}_train.txt'.format(holdout_fold))
                                
        validate_txt = os.path.join(dataset_dir, 'evaluation_setup', 
                                    'fold{}_evaluate.txt'.format(holdout_fold))
                                    
    else:
        train_txt = None
        validate_txt = None
                             
    if validate:
        models_dir = os.path.join(workspace, 'models', filename, 
                                'holdout_fold={}'.format(holdout_fold))
                                
    else:
        models_dir = os.path.join(workspace, 'models', filename, 'full_train')
                              
    create_folder(models_dir)
    
    # Model
    model = BaselineCnn(classes_num)

    if cuda:
        model.cuda()

    generator = DataGenerator(hdf5_path=hdf5_path,
                        batch_size=batch_size, 
                        train_txt=train_txt, 
                        validate_txt=validate_txt)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    iteration = 0
    train_bgn_time = time.time()

    # Train on mini-batch
    for (batch_x, batch_y) in generator.generate_train():
        
        # Evaluate
        if iteration % 200 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_f1_score, tr_loss) = evaluate(
                model=model,
                generator=generator,
                data_type='train',
                max_iteration=max_validate_iteration,
                cuda=cuda)

            if validate:
                (va_acc, va_f1_score, va_loss) = evaluate(
                    model=model,
                    generator=generator,
                    data_type='validate',
                    max_iteration=max_validate_iteration,
                    cuda=cuda)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            # Print info
            logging.info(
                "iteration: {}, train time: {:.3f} s, validate time: {:.3f} s".format(
                    iteration, train_time, validate_time))

            logging.info("tr_acc: {:.3f}, tr_f1_score: {:.3f}, tr_loss: {:.3f}".format(
                tr_acc, tr_f1_score, tr_loss))

            if validate:
                logging.info("va_acc: {:.3f}, va_f1_score: {:.3f}, va_loss: {:.3f}".format(
                    va_acc, va_f1_score, va_loss))

            logging.info("")

            train_bgn_time = time.time()

        # Move data to gpu
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        # Train
        model.train()
        output = model(batch_x)
        loss = F.nll_loss(output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
        
        # Save model
        if iteration % 1000 == 0 and iteration > 0:

            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info("Model saved to {}".format(save_out_path))
            
        # Reduce learning rate
        if iteration % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
                
        # Stop learning
        if iteration == 10000:
            break
            
            
def inference_validation(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    cuda = args.cuda
    filename = args.filename

    validation = True
    batch_size = 64
    labels = config.labels
    classes_num = len(config.labels)
    
    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'development.h5')
                             
    train_txt = os.path.join(dataset_dir, 'evaluation_setup', 
                            'fold{}_train.txt'.format(holdout_fold))
                            
    validate_txt = os.path.join(dataset_dir, 'evaluation_setup', 
                                'fold{}_evaluate.txt'.format(holdout_fold))
                             
    model_path = os.path.join(workspace, 'models', filename, 
                              'holdout_fold={}'.format(holdout_fold), 
                              'md_{}_iters.tar'.format(iteration))
                       
    # Model
    model = BaselineCnn(classes_num)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size, 
                              train_txt=train_txt, 
                              validate_txt=validate_txt)

    generate_func = generator.generate_validate(data_type='validate', 
                                                max_iteration=-1)
    
    # Inference
    inference_time = time.time()

    (outputs, targets, audio_names) = forward(model=model,
                                              generate_func=generate_func,
                                              cuda=cuda, 
                                              has_target=True)
                                              
    logging.info("Inference time: {:.3f} s".format(
        time.time() - inference_time))

    predictions = np.argmax(outputs, axis=-1)

    # Calculate statistics
    accuracy = calculate_accuracy(targets, predictions)

    class_wise_f1_score = calculate_f1_score(targets, predictions, average=None)
    
    # Print statistics
    logging.info('Averaged accuracy: {:.3f}'.format(accuracy))
    
    logging.info('{:<30}{}'.format('class_name', 'f1_score'))
    logging.info('---------------------------------------')
    
    for (n, lb) in enumerate(labels):
        logging.info('{:<30}{:.3f}'.format(lb, class_wise_f1_score[n]))
    
    logging.info('---------------------------------------')
    logging.info('{:<30}{:.3f}'.format('Average', np.mean(class_wise_f1_score)))    
    
    
def inference_testing_data(args):

    # Arugments & parameters
    workspace = args.workspace
    iteration = args.iteration
    cuda = args.cuda
    filename = args.filename

    validation = True
    batch_size = 64
    labels = config.labels
    classes_num = len(config.labels)
    ix_to_lb = config.ix_to_lb
    
    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'development.h5')
    
    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'test.h5')
                             
    model_path = os.path.join(workspace, 'models', filename, 'full_train', 
                              'md_{}_iters.tar'.format(iteration))
                              
    submission_path = os.path.join(workspace, 'submissions', filename, 
                                   'iteration={}'.format(iteration), 
                                   'submission.csv')
                       
    create_folder(os.path.dirname(submission_path))
                       
    # Model
    model = BaselineCnn(classes_num)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path,
                                  test_hdf5_path=test_hdf5_path, 
                                  batch_size=batch_size)

    generate_func = generator.generate_test()

    # Inference
    inference_time = time.time()
    
    (outputs, audio_names) = forward(model=model,
                                              generate_func=generate_func,
                                              cuda=cuda, 
                                              has_target=False)
                                              
    logging.info("Inference time: {:.3f} s".format(
        time.time() - inference_time))

    predictions = np.argmax(outputs, axis=-1)

    # Write out submission file
    f = open(submission_path, 'w')

    for (n, audio_name) in enumerate(audio_names):
        f.write('{}\t{}\n'.format(audio_name, ix_to_lb[predictions[n]]))

    f.close()
    
    print('Write out submission file to {}'.format(submission_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int, choices=[1, 2, 3, 4])
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--holdout_fold', type=int, choices=[1, 2, 3, 4])
    parser_inference_validation.add_argument('--iteration', type=int)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_testing_data = subparsers.add_parser('inference_testing_data')
    parser_inference_testing_data.add_argument('--workspace', type=str, required=True)
    parser_inference_testing_data.add_argument('--iteration', type=int)
    parser_inference_testing_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)
    
    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_folder(os.path.dirname(logs_dir))
    logging = create_logging(logs_dir, filemode='w')

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)
        
    elif args.mode == 'inference_testing_data':
        inference_testing_data(args)
        
    else:
        raise Exception("Incorrect argument!")