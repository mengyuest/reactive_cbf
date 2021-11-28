import tensorflow as tf
import numpy as np
import h5py
import random
import argparse
import os
import hgail.misc.utils as h_utils
import hgail.misc.tf_utils
import rllab.misc.logger as logger
from datetime import datetime
import time
from os.path import join as ospj
import sys

class Recorder:
    def __init__(self, larger_is_better=True):
        self.history = []
        self.larger_is_better = larger_is_better
        self.best_at = None
        self.best_val = None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x > y
        else:
            return x < y

    def update(self, val):
        self.history.append(val)
        if len(self.history) == 1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history) - 1

    def is_current_best(self):
        return self.best_at == len(self.history) - 1


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return np.array(lengths)

def get_policy_var_list():
    return [v for v in tf.trainable_variables() if "policy" in v.name]

def get_exp_home():
    return "../../../train_exps/"

def get_exp_name(exp_name):
    return exp_name

def write_cmd_to_file(log_dir, argv):
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write(dt_string + "\n")
        seen = set()
        argv_list = []
        for i, x in enumerate(argv):
            if len(x) > 0:
                argv_list.append(x)
            if x.startswith("--"):
                if x not in seen:
                    seen.add(x)
                else:
                    exit("duplicate flag: %s" % x)
            if x == "--params_filepath":
                if len(argv[i + 1]) == 0:
                    argv_list.append("\"\"")
        f.write("python " + " ".join(argv_list))

def get_ext_indices(cbf_intervals):
    intervals = [int(xx) for xx in cbf_intervals.split(",")]
    ext_indices = []
    # "a,b" means [a-1,b-1], i.e. [a-1,b)
    for i in range(0, len(intervals), 2):
        i_start = intervals[i]
        i_end = intervals[i + 1]
        for j in range(i_start - 1, i_end):
            ext_indices.append(j)
    return ext_indices


class Logger(object):
    def __init__(self, path):
        self._terminal=sys.stdout
        self._log = open(path,"w")
    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)
    def flush(self):
        pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Trajectory Prediction Network")
    parser.add_argument('--exp_name', type=str, default="exp")
    parser.add_argument('--dataset_path', type=str, default="../../../data/trajectories/ngsim_all_attrmlc_new.h5")
    parser.add_argument('--hiddens', nargs='+', default=(256,256,256,64))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--local', action='store_true')

    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # logger and setup dir
    args.exp_name = get_exp_name(args.exp_name)
    exp_dir = ospj(get_exp_home(), args.exp_name)
    log_dir = exp_dir
    os.makedirs(log_dir, exist_ok=True)
    # logger.add_text_output(ospj(exp_dir, 'log.txt'))
    logger = Logger(ospj(exp_dir, 'log.txt'))
    sys.stdout = logger
    write_cmd_to_file(log_dir, sys.argv)
    args.full_log_dir = log_dir

    # data
    handle = h5py.File(args.dataset_path, 'r')
    keys = list(handle.keys())
    data_list=[]
    for key in keys:
        data_list.append(np.array(handle[key]))
        print(key,data_list[-1].shape)
    # data = np.array(handle['1'])
    data = np.concatenate(data_list, axis=-1)

    if "ngsim" not in args.exp_name:
        data = np.swapaxes(data, 0, 2)
    lengths = compute_lengths(data)
    xs = []
    for i, l in enumerate(lengths):
        xs.append(data[i, :l])
    data = np.concatenate(xs)

    print(data.shape)
    for i in range(data.shape[1]):
        # print(handle.attrs["feature_names"][i], np.min(data[:, i]), np.max(data[:, i]), np.mean(data[:, i]))
        print(np.min(data[:, i]), np.max(data[:, i]), np.mean(data[:, i]))
    # exit()
    if "ngsim" in args.exp_name:
        ext_intervals=get_ext_indices("6,8,15,23,67,128")
        cbf_intervals=get_ext_indices("1,3,11,12,15,74")
        obs_array = data[:, ext_intervals][:, cbf_intervals]
        act_array = data[:, [8, 10]]
    else:
        obs_array = data[:, :-2]
        act_array = data[:, [-2, -1]]

    print("shape", obs_array.shape, act_array.shape)

    N = obs_array.shape[0]
    arr = np.arange(N)
    np.random.shuffle(arr)
    split = int(arr.shape[0] * args.train_split)
    train_idx = arr[:split]
    val_idx = arr[split:]

    train_obs_array = obs_array[train_idx]
    train_act_array = act_array[train_idx]

    val_obs_array = obs_array[val_idx]
    val_act_array = act_array[val_idx]

    # build computation graph
    x_pl = tf.placeholder(tf.float32, shape=[None, obs_array.shape[1]], name="x")
    u_pl = tf.placeholder(tf.float32, shape=[None, 2], name="u")
    cat_x = tf.expand_dims(x_pl, axis=1)
    for i, hidden_num in enumerate(args.hiddens):
        cat_x = tf.contrib.layers.conv1d(inputs=cat_x,
                                         num_outputs=hidden_num,
                                         kernel_size=1,
                                         reuse=tf.AUTO_REUSE,
                                         scope='policy/conv%d' % i,
                                         activation_fn=tf.nn.relu)

    cat_x = tf.contrib.layers.conv1d(inputs=cat_x,
                                     num_outputs=2,
                                     kernel_size=1,
                                     reuse=tf.AUTO_REUSE,
                                     scope='policy/conv%d' % len(args.hiddens),
                                     activation_fn=None)
    est_u = tf.squeeze(cat_x, axis=1)  # TODO(modified)
    print(est_u.shape, u_pl.shape)
    loss = tf.reduce_mean(tf.square(tf.clip_by_value(u_pl, -10, 10)-est_u))
    # loss = tf.reduce_mean(tf.square(u_pl - est_u))

    optimizer = tf.train.AdamOptimizer(args.lr)
    joint_params = get_policy_var_list()
    global_step = tf.Variable(0, name='joint/global_step', trainable=False)
    gradients = tf.gradients(loss, joint_params)
    grad_norm_rescale = 40.
    grad_norm_clip = 10000.
    clipped_grads = hgail.misc.tf_utils.clip_gradients(gradients, grad_norm_rescale, grad_norm_clip)
    train_op = optimizer.apply_gradients(list(zip(clipped_grads, joint_params)), global_step=global_step)

    train_losses = AverageMeter()
    val_losses = AverageMeter()
    val_loss_record = Recorder(larger_is_better=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epo in range(args.epochs):
            num_itrs = train_obs_array.shape[0] // args.batch_size

            train_print_freq = num_itrs // 3

            # train
            for itr in range(num_itrs):
                train_obs_batch = train_obs_array[itr*args.batch_size: (itr+1)*args.batch_size]
                train_act_batch = train_act_array[itr * args.batch_size: (itr + 1) * args.batch_size]

                _, u_np, loss_np = sess.run([train_op, est_u, loss], feed_dict={x_pl: train_obs_batch, u_pl: train_act_batch})

                train_losses.update(loss_np)
                if itr % train_print_freq==0:
                    print("[%d] Train %04d/%04d %.4f(%.4f)"%(epo, itr, num_itrs, train_losses.val, train_losses.avg))
            # validate
            num_val_itrs = val_obs_array.shape[0] // args.batch_size
            val_print_freq = num_val_itrs // 3
            for itr in range(num_val_itrs):
                val_obs_batch = val_obs_array[itr * args.batch_size: (itr + 1) * args.batch_size]
                val_act_batch = val_act_array[itr * args.batch_size: (itr + 1) * args.batch_size]

                u_np, loss_np = sess.run([est_u, loss],
                                            feed_dict={x_pl: val_obs_batch, u_pl: val_act_batch})
                val_losses.update(loss_np)
                if itr % val_print_freq==0:
                    print("[%d] Val %04d/%04d %.4f(%.4f)"%(epo, itr, num_val_itrs, val_losses.val, val_losses.avg))


            val_loss_record.update(val_losses.avg)
            print("[%d] train loss:%.4f  val loss:%.4f  val-best:%.4f ( =%d)" % (epo, train_losses.avg, val_losses.avg,
                                                           val_loss_record.best_val,
                                                           val_loss_record.best_at))
            if epo % args.save_freq == 0:
                params = dict()
                policy_var_list = get_policy_var_list()
                session = tf.get_default_session()
                params['policy'] = [session.run(v) for v in policy_var_list]
                h_utils.save_params(log_dir, params, epo, max_to_keep=10000)