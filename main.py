#!/usr/bin/env python3
import sys

sys.path.append('./lib')
import argparse
import os
import datetime
import numpy as np
import time
import pickle
import torch
from torch import optim

import utils
from param_stamp import get_param_stamp, get_param_stamp_from_args
import evaluate
from data import get_multitask_experiment
from lib.encoder import Classifier
from lib.vae_models import AutoEncoder
import lib.callbacks as cb
from lib.train import train_cl
from lib.continual_learner import ContinualLearner
from lib.exemplars import ExemplarHandler
from lib.replayer import Replayer
import lib.visual_plt

EXPERIMENT = 'mydataset'
VISDOM = VISDOM_EXEMPLARS = None
SEED = 7
RESULT_DIR = './results1'
SCENARIO = 'domain'
# use binary (instead of multi-class) classication loss
#BCE = True
# size of latent representation
Z_DIM = 100

parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
parser.add_argument('--get-stamp', action='store_true')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--factor', type=str, default='clutter', dest='factor')
parser.add_argument('--savepath', type=str, default='./results', dest='savepath')
parser.add_argument('--cumulative', type=int, default=0, dest='cul')
parser.add_argument('--bce', action='store_true')

parser.add_argument('--tasks', type=int, default=9)

parser.add_argument('--fc-layers', type=int, default=3, dest='fc_lay')
parser.add_argument('--fc-units', type=int, default=400, metavar="N")
parser.add_argument('--fc-drop', type=float, default=0.)
parser.add_argument('--fc-bn', type=str, default="no")
parser.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])

parser.add_argument('--iters', type=int, default=3000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

parser.add_argument('--feedback', action="store_true")
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
parser.add_argument('--replay', type=str, default='none', choices=replay_choices)
parser.add_argument('--distill', action='store_true')
parser.add_argument('--temp', type=float, default=2., dest='temp')

parser.add_argument('--g-z-dim', type=int, default=100)
parser.add_argument('--g-fc-lay', type=int)
parser.add_argument('--g-fc-uni', type=int)

parser.add_argument('--g-iters', type=int)
parser.add_argument('--lr-gen', type=float)

parser.add_argument('--ewc', action='store_true')
parser.add_argument('--lambda', type=float, default=5240., dest="ewc_lambda")
parser.add_argument('--fisher-n', type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--gamma', type=float, default=1.)
parser.add_argument('--emp-fi', action='store_true')
parser.add_argument('--si', action='store_true')
parser.add_argument('--c', type=float, default=0.3, dest="si_c")
parser.add_argument('--epsilon', type=float, default=0.2, dest="epsilon")
parser.add_argument('--xdg', type=float, default=0., dest="gating_prop")

parser.add_argument('--icarl', action='store_true')
parser.add_argument('--use-exemplars', action='store_true')
parser.add_argument('--add-exemplars', action='store_true')
parser.add_argument('--budget', type=int, default=2500, dest="budget")
parser.add_argument('--herding', action='store_true')
parser.add_argument('--norm-exemplars', action='store_true')

parser.add_argument('--log-per-task', action='store_true')
parser.add_argument('--loss-log', type=int, default=200, metavar="N")
parser.add_argument('--prec-log', type=int, default=200, metavar="N")
parser.add_argument('--prec-n', type=int, default=1024)
parser.add_argument('--sample-log', type=int, default=500, metavar="N")
parser.add_argument('--sample-n', type=int, default=64)


def run(args):
    result_path = os.path.join('./precision_onEachTask', args.savepath)
    savepath = result_path + '/' + str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) + '.csv'
    os.makedirs(result_path, exist_ok=True)

    # Set default arguments
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -if [log_per_task], reset all logs
    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
    # -if [iCaRL] is selected, select all accompanying options
    if hasattr(args, "icarl") and args.icarl:
        args.use_exemplars = True
        args.add_exemplars = True

    # -if EWC, SI or XdG is selected together with 'feedback', give error
    if args.feedback and (args.ewc or args.si or args.gating_prop > 0 or args.icarl):
        raise NotImplementedError("EWC, SI, XdG and iCaRL are not supported with feedback connections.")
    # -if binary classification loss is selected together with 'feedback', give error
    if args.feedback and args.bce:
        raise NotImplementedError("Binary classification loss not supported with feedback connections.")
    # -if XdG is selected together with both replay and EWC, give error (either one of them alone with XdG is fine)
    if args.gating_prop > 0 and (not args.replay == "none") and (args.ewc or args.si):
        raise NotImplementedError("XdG is not supported with both '{}' replay and EWC / SI.".format(args.replay))
        # --> problem is that applying different task-masks interferes with gradient calculation
        #    (should be possible to overcome by calculating backward step on EWC/SI-loss also for each mask separately)
    # -create plots- and results-directories if needed
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    scenario = SCENARIO
    # (but note that when XdG is used, task-identity information is being used so the actual scenario is still Task-IL)

    # If only want param-stamp, get it printed to screen and exit
    if hasattr(args, "get_stamp") and args.get_stamp:
        _ = get_param_stamp_from_args(args=args)
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    if args.factor == 'sequence':
        args.tasks = 12

    # -------------------------------------------------------------------------------------------------#

    # ----------------#
    # ----- DATA -----#
    # ----------------#

    # Prepare data for chosen experiment
    with open(args.factor + '.pk', 'rb') as f:
        ((train_datasets, test_datasets), config, classes_per_task) = pickle.load(f)

    if args.cul == 1:
        for i in range(1, len(train_datasets)):
            train_datasets[i].imgs.extend(train_datasets[i - 1].imgs)
            train_datasets[i].labels.extend(train_datasets[i - 1].labels)
    # -------------------------------------------------------------------------------------------------#

    # ------------------------------#
    # ----- MODEL (CLASSIFIER) -----#
    # ------------------------------#

    # Define main model (i.e., classifier, if requested with feedback connections)
    if args.feedback:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.g_fc_uni, z_dim=Z_DIM,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
        ).to(device)
        model.lamda_pl = 1.  # --> to make that this VAE is also trained to classify
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
            fc_bn=True if args.fc_bn == "yes" else False, excit_buffer=True if args.gating_prop > 0 else False,
            binaryCE=args.bce, binaryCE_distill=True,
        ).to(device)

    # Define optimizer (only include parameters that "requires_grad")
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type == "sgd":
        model.optimizer = optim.SGD(model.optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))

    # ----------------------------------#
    # ----- CL-STRATEGY: EXEMPLARS -----#
    # ----------------------------------#

    # Store in model whether, how many and in what way to store exemplars
    if isinstance(model, ExemplarHandler) and (args.use_exemplars or args.add_exemplars or args.replay == "exemplars"):
        model.memory_budget = args.budget
        model.norm_exemplars = args.norm_exemplars
        model.herding = args.herding

    # -----------------------------------#
    # ----- CL-STRATEGY: ALLOCATION -----#
    # -----------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        if args.ewc:
            model.fisher_n = args.fisher_n
            model.gamma = args.gamma
            model.online = args.online
            model.emp_FI = args.emp_fi

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner):
        model.si_c = args.si_c if args.si else 0
        if args.si:
            model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and args.gating_prop > 0:
        mask_dict = {}
        excit_buffer_list = []
        for task_id in range(args.tasks):
            mask_dict[task_id + 1] = {}
            for i in range(model.fcE.layers):
                layer = getattr(model.fcE, "fcLayer{}".format(i + 1)).linear
                if task_id == 0:
                    excit_buffer_list.append(layer.excit_buffer)
                n_units = len(layer.excit_buffer)
                gated_units = np.random.choice(n_units, size=int(args.gating_prop * n_units), replace=False)
                mask_dict[task_id + 1][i] = gated_units
        model.mask_dict = mask_dict
        model.excit_buffer_list = excit_buffer_list

    # -------------------------------------------------------------------------------------------------#

    # -------------------------------#
    # ----- CL-STRATEGY: REPLAY -----#
    # -------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, Replayer):
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = True if (args.replay == "generative" and not args.feedback) else False
    if train_gen:
        # -specify architecture
        generator = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=100, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
        ).to(device)
        # -set optimizer(s)
        generator.optim_list = [
            {'params': filter(lambda p: p.requires_grad, generator.parameters()), 'lr': args.lr_gen}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
    else:
        generator = None

    # ---------------------#
    # ----- REPORTING -----#
    # ---------------------#

    # Get parameter-stamp (and print on screen)
    param_stamp = get_param_stamp(
        args, model.name, verbose=True, replay=True if (not args.replay == "none") else False,
        replay_model_name=generator.name if (args.replay == "generative" and not args.feedback) else None,
    )

    # Prepare for plotting in visdom
    # -define [precision_dict] to keep track of performance during training for storing and for later plotting in pdf
    precision_dict = evaluate.initiate_precision_dict(args.tasks)
    precision_dict_exemplars = evaluate.initiate_precision_dict(args.tasks) if args.use_exemplars else None

    # ---------------------#
    # ----- CALLBACKS -----#
    # ---------------------#

    # Callbacks for reporting on and visualizing loss
    generator_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log, visdom=VISDOM, model=model if args.feedback else generator, tasks=args.tasks,
                        iters_per_task=args.iters if args.feedback else args.g_iters,
                        replay=False if args.replay == "none" else True)
    ] if (train_gen or args.feedback) else [None]
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=VISDOM, model=model, tasks=args.tasks,
                           iters_per_task=args.iters, replay=False if args.replay == "none" else True)
    ] if (not args.feedback) else [None]

    # Callbacks for evaluating and plotting generated / reconstructed samples
    sample_cbs = [
        cb._sample_cb(log=args.sample_log, visdom=VISDOM, config=config, test_datasets=test_datasets,
                      sample_size=args.sample_n, iters_per_task=args.iters if args.feedback else args.g_iters)
    ] if (train_gen or args.feedback) else [None]

    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [prec_log]
    eval_cb = cb._eval_cb(
        log=args.prec_log, test_datasets=test_datasets, visdom=VISDOM, precision_dict=None, iters_per_task=args.iters,
        test_size=args.prec_n, classes_per_task=classes_per_task, scenario=SCENARIO,
    )
    # -pdf / reporting: summary plots (i.e, only after each task)
    eval_cb_full = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, precision_dict=precision_dict,
        iters_per_task=args.iters, classes_per_task=classes_per_task, scenario=SCENARIO,
    )
    # -with exemplars (both for visdom & reporting / pdf)
    eval_cb_exemplars = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, visdom=VISDOM_EXEMPLARS, classes_per_task=classes_per_task,
        precision_dict=precision_dict_exemplars, scenario=SCENARIO, iters_per_task=args.iters,
        with_exemplars=True,
    ) if args.use_exemplars else None
    # -collect them in <lists>
    eval_cbs = [eval_cb, eval_cb_full]
    eval_cbs_exemplars = [eval_cb_exemplars]

    # -------------------------------------------------------------------------------------------------#

    # --------------------#
    # ----- TRAINING -----#
    # --------------------#

    print("--> Training:")
    # Keep track of training-time
    start = time.time()
    # Train model
    train_cl(
        model, train_datasets, test_datasets, replay_mode=args.replay, scenario=SCENARIO,
        classes_per_task=classes_per_task,
        iters=args.iters, batch_size=args.batch, savepath=savepath,
        generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
        sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
        eval_cbs_exemplars=eval_cbs_exemplars, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
    )


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
