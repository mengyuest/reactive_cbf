import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from os.path import join as ospj
import sys
import time
import random
import numpy as np
import tensorflow as tf
import configs
import utils
import rllab.misc.logger as logger
from rllab.misc import tensor_utils
import itertools

import hgail.misc.utils as h_utils
import hgail.misc.tf_utils
import sandbox.rocky.tf.algos.utils_cbf as utils_cbf
import geo_check
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy

import rllab.spaces
import scipy
import scipy.linalg as la

def load_pretrained_weights(args, policy1, policy2, policy_ref):

    if not args.zero_reference and args.dest_controller_type=="none":
        params = h_utils.load_params(args.policy_reference_path)
        print("Load referenced policy pretrained weights...")
        policy_ref.set_param_values(params['policy'])
        obs_mean = params['normalzing']['obs_mean']
        obs_var = params['normalzing']['obs_var']
    else:
        obs_mean=None
        obs_var=None

    if not args.zero_policy:
        if args.cbf_pretrained_path is not None:
            print("Load CBF pretrained weights...")
            cbf_params = h_utils.load_params(args.cbf_pretrained_path)
            utils_cbf.set_cbf_param_values(cbf_params['jcbfer'], args)

        if args.policy_pretrained_path1 is not None:
            print("Load policy pretrained weights...")
            policy_params1 = h_utils.load_params(args.policy_pretrained_path1)
            policy1.set_param_values(policy_params1['policy1'])

        if args.policy_pretrained_path2 is not None:
            print("Load policy pretrained weights...")
            policy_params2 = h_utils.load_params(args.policy_pretrained_path2)
            policy2.set_param_values(policy_params2['policy2'])

    return obs_mean, obs_var

def build_sub_graph(policy, state, uref, safe_mask, dang_mask, medium_mask, tf_dict, tag, args, refinement=False):
    if args.include_u_ref_feat:
        u = policy.get_action_tf_flat_with_uref(tf_dict, state, uref)
    else:
        u = policy.get_action_tf_flat(tf_dict, state)
    if args.debug_easy:
        tf_dict["debug_original_u%s"%tag] = u
        tf_dict["debug_uref%s"%tag] = uref

    if refinement:
        if tag=="1":
            u_res_n=args.n_envs1
        elif tag=="2":
            u_res_n=args.n_envs2
        u_res = tf.Variable(tf.zeros([u_res_n, 2]), name='u_res%s'%tag)
        u_init = tf.assign(u_res, tf.zeros_like(u_res))
        if args.pre_action_clip is not None:
            u = tf.clip_by_value(u + uref, -args.pre_action_clip, args.pre_action_clip)
            u = u + u_res
        else:
            u = u + uref + u_res
    else:
        u_res=None
        u_init=None
        u = u + uref

    if args.debug_easy:
        tf_dict["debug_u%s"%tag] = u
    score = network_cbf(state, args, tf_dict, dbg_name="first%s"%tag)
    if args.debug_easy:
        tf_dict["debug_score%s"%tag] = score
    if tag == "1":
        state_tp1 = dynamics1(args, state, u, u * 0.0)
    else:
        state_tp1 = dynamics2(args, state, u, u * 0.0)
    score_tp1 = network_cbf(state_tp1, args, tf_dict, dbg_name="second%s"%tag)

    if args.debug_easy:
        tf_dict["debug_score_next%s"%tag] = score_tp1
        tf_dict["debug_score%s"%tag] = score

    num_safe = tf.reduce_sum(safe_mask)
    num_dang = tf.reduce_sum(dang_mask)
    num_medium = tf.reduce_sum(medium_mask)

    # barrier loss
    loss_safe_full = tf.math.maximum(-score + args.h_safe_thres, 0) * safe_mask / (1e-5 + num_safe)
    loss_dang_full = tf.math.maximum(score + args.h_dang_thres, 0) * dang_mask / (1e-5 + num_dang)
    loss_safe = tf.reduce_sum(loss_safe_full)
    loss_dang = tf.reduce_sum(loss_dang_full)
    acc_safe = tf.reduce_sum(tf.cast(tf.greater_equal(score, 0), tf.float32) * safe_mask) / (1e-12 + num_safe)
    acc_dang = tf.reduce_sum(tf.cast(tf.less_equal(score, 0), tf.float32) * dang_mask) / (1e-12 + num_dang)
    acc_safe = tf.cond(tf.greater(num_safe, 0), lambda: acc_safe, lambda: -tf.constant(1.0))
    acc_dang = tf.cond(tf.greater(num_dang, 0), lambda: acc_dang, lambda: -tf.constant(1.0))

    # derivative loss
    deriv_safe_thres = args.grad_safe_thres
    deriv_medium_thres = args.grad_medium_thres
    deriv_dang_thres = args.grad_dang_thres

    lamda = 1 - 0.1 * args.alpha_cbf
    if not refinement:
        d_safe_full = tf.math.maximum(deriv_safe_thres - score_tp1 + lamda * score, 0) * safe_mask / (
                1e-12 + num_safe)
        d_medium_full = tf.math.maximum(deriv_medium_thres - score_tp1 + lamda * score, 0) * medium_mask / (
                1e-12 + num_medium)
        d_dang_full = tf.math.maximum(deriv_dang_thres - score_tp1 + lamda * score, 0) * dang_mask / (
                1e-12 + num_dang)
    else:
        d_safe_full = tf.math.maximum(- score_tp1 + lamda * score, 0) * safe_mask / (1e-12 + num_safe)
        d_medium_full = tf.math.maximum(- score_tp1 + lamda * score, 0) * medium_mask / (1e-12 + num_medium)
        d_dang_full = tf.math.maximum(- score_tp1 + lamda * score, 0) * dang_mask / (1e-12 + num_dang)
    loss_d_safe = tf.reduce_sum(d_safe_full)
    loss_d_medium = tf.reduce_sum(d_medium_full)
    loss_d_dang = tf.reduce_sum(d_dang_full)

    d_acc_safe = tf.reduce_sum(tf.cast(tf.greater_equal(score_tp1 - lamda * score, 0), tf.float32) * safe_mask) / (
            1e-12 + num_safe)
    d_acc_medium = tf.reduce_sum(
        tf.cast(tf.greater_equal(score_tp1 - lamda * score, 0), tf.float32) * medium_mask) / (1e-12 + num_medium)
    d_acc_dang = tf.reduce_sum(tf.cast(tf.greater_equal(score_tp1 - lamda * score, 0), tf.float32) * dang_mask) / (
            1e-12 + num_dang)

    acc_d_safe = tf.cond(tf.greater(num_safe, 0), lambda: d_acc_safe, lambda: -tf.constant(1.0))
    acc_d_medium = tf.cond(tf.greater(num_medium, 0), lambda: d_acc_medium, lambda: -tf.constant(1.0))
    acc_d_dang = tf.cond(tf.greater(num_dang, 0), lambda: d_acc_dang, lambda: -tf.constant(1.0))

    # regularization loss
    if tag=="1":
        loss_reg_policy_full = tf.reduce_sum(tf.math.square((u - uref) / tf.constant([[4.0, 4.0]])), axis=[1])
    else:  # TODO, bicycle case, larger norm for omega
        loss_reg_policy_full = tf.reduce_sum(tf.math.square((u - uref) / tf.constant([[4.0, .15]])), axis=[1])

    loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [tf.shape(loss_reg_policy_full)[0], 1])
    loss_reg_policy_full = (loss_reg_policy_full * safe_mask) / num_safe
    loss_reg_policy = tf.reduce_sum(loss_reg_policy_full)

    # total loss
    loss_crit = loss_safe * args.safe_loss_weight + loss_dang * args.dang_loss_weight
    loss_grad = loss_d_safe * args.safe_deriv_loss_weight + \
                loss_d_medium * args.medium_deriv_loss_weight + \
                loss_d_dang * args.dang_deriv_loss_weight
    loss_reg = loss_reg_policy * args.reg_policy_loss_weight
    total_loss = loss_crit + loss_grad + loss_reg

    tf_dict["loss_safe"] = loss_safe * args.safe_loss_weight
    tf_dict["loss_dang"] = loss_dang * args.dang_loss_weight
    tf_dict["loss_d_safe"] = loss_d_safe * args.safe_deriv_loss_weight
    tf_dict["loss_d_medium"] = loss_d_medium * args.medium_deriv_loss_weight
    tf_dict["loss_d_dang"] = loss_d_dang * args.dang_deriv_loss_weight

    tf_dict["loss_crit"] = loss_crit
    tf_dict["loss_grad"] = loss_grad
    tf_dict["loss_reg"] = loss_reg
    tf_dict["total_loss"] = total_loss

    tf_dict["acc_safe"] = acc_safe
    tf_dict["acc_dang"] = acc_dang
    tf_dict["acc_d_safe"] = acc_d_safe
    tf_dict["acc_d_medium"] = acc_d_medium
    tf_dict["acc_d_dang"] = acc_d_dang

    return loss_safe, loss_dang, loss_d_safe, loss_d_medium, loss_dang, loss_crit, loss_grad, loss_reg, total_loss,\
            acc_safe, acc_dang, acc_d_safe, acc_d_medium, acc_d_dang, num_safe, num_medium, num_dang, u_res, u_init


def build_computation_graph(policy1, policy2, policy_ref, args, refinement=False):
    state1 = tf.placeholder(tf.float32, shape=[None, args.num_neighbors*4+2], name="state1")
    uref1 = tf.placeholder(tf.float32, shape=[None, 2], name="uref1")
    state2 = tf.placeholder(tf.float32, shape=[None, args.num_neighbors * 4 + 2], name="state2")
    uref2 = tf.placeholder(tf.float32, shape=[None, 2], name="uref2")

    safe_mask1 = tf.placeholder(tf.float32, [None, args.num_neighbors], name="safe_mask1")
    dang_mask1 = tf.placeholder(tf.float32, [None, args.num_neighbors], name="dang_mask1")
    medium_mask1 = tf.placeholder(tf.float32, [None, args.num_neighbors], name="medium_mask1")
    safe_mask2 = tf.placeholder(tf.float32, [None, args.num_neighbors], name="safe_mask2")
    dang_mask2 = tf.placeholder(tf.float32, [None, args.num_neighbors], name="dang_mask2")
    medium_mask2 = tf.placeholder(tf.float32, [None, args.num_neighbors], name="medium_mask2")

    pl_dict=dict(
        state1=state1,
        uref1=uref1,
        safe_mask1=safe_mask1,
        dang_mask1=dang_mask1,
        medium_mask1=medium_mask1,
        state2=state2,
        uref2=uref2,
        safe_mask2=safe_mask2,
        dang_mask2=dang_mask2,
        medium_mask2=medium_mask2,
    )

    tf_dict = {}


    if policy1 is not None:
        loss_safe1, loss_dang1, loss_d_safe1, loss_d_medium1, loss_d_dang1, loss_crit1, loss_grad1, loss_reg1, total_loss1, \
        acc_safe1, acc_dang1, acc_d_safe1, acc_d_medium1, acc_d_dang1, num_safe1, num_medium1, num_dang1, u_res1, u_init1\
            = build_sub_graph(policy1, state1, uref1, safe_mask1, dang_mask1, medium_mask1,
                               tf_dict, tag="1", args=args, refinement=refinement)

    if policy2 is not None:
        loss_safe2, loss_dang2, loss_d_safe2, loss_d_medium2, loss_d_dang2, loss_crit2, loss_grad2, loss_reg2, total_loss2, \
        acc_safe2, acc_dang2, acc_d_safe2, acc_d_medium2, acc_d_dang2, num_safe2, num_medium2, num_dang2, u_res2, u_init2 \
            = build_sub_graph(policy2, state2, uref2, safe_mask2, dang_mask2, medium_mask2,
                              tf_dict, tag="2", args=args, refinement=refinement)


    # optimizer
    if not refinement:
        cbf_params = utils_cbf.get_var_list()
        if policy1 is not None:
            policy_params1 = policy1.get_params(trainable=True)
            # print(policy_params1)
            optimizer1 = tf.train.AdamOptimizer(args.joint_learning_rate)
            if args.joint_for_policy_only:
                joint_params1 = policy_params1
            elif args.joint_for_cbf_only:
                joint_params1 = cbf_params
            else:
                joint_params1 = cbf_params + policy_params1
            global_step1 = tf.Variable(0, name='joint/global_step1', trainable=False)
            gradients1 = tf.gradients(total_loss1, joint_params1)
            grad_norm_rescale = 40.
            grad_norm_clip = 10000.
            clipped_grads1 = hgail.misc.tf_utils.clip_gradients(gradients1, grad_norm_rescale, grad_norm_clip)
            train_op1 = optimizer1.apply_gradients(list(zip(clipped_grads1, joint_params1)), global_step=global_step1)
        else:
            train_op1 = None
        if policy2 is not None:
            policy_params2 = policy2.get_params(trainable=True)
            # print(policy_params2)
            optimizer2 = tf.train.AdamOptimizer(args.joint_learning_rate)
            if args.joint_for_policy_only:
                joint_params2 = policy_params2
            elif args.joint_for_cbf_only:
                joint_params2 = cbf_params
            else:
                joint_params2 = cbf_params + policy_params2
            global_step2 = tf.Variable(0, name='joint/global_step2', trainable=False)
            gradients2 = tf.gradients(total_loss2, joint_params2)
            grad_norm_rescale = 40.
            grad_norm_clip = 10000.
            clipped_grads2 = hgail.misc.tf_utils.clip_gradients(gradients2, grad_norm_rescale, grad_norm_clip)
            train_op2 = optimizer2.apply_gradients(list(zip(clipped_grads2, joint_params2)), global_step=global_step2)
        else:
            train_op2 = None

    else:
        if policy1 is not None:
            gradients1 = tf.gradients(loss_grad1, u_res1)
            u_res_new1 = tf.assign(u_res1, u_res1 - gradients1[0] * args.refine_learning_rate)
        if policy2 is not None:
            gradients2 = tf.gradients(loss_grad2, u_res2)
            u_res_new2 = tf.assign(u_res2, u_res2 - gradients2[0] * args.refine_learning_rate)

    # monitoring
    if policy1 is not None:
        tf_dict["loss_safe1"] = loss_safe1 * args.safe_loss_weight
        tf_dict["loss_dang1"] = loss_dang1 * args.dang_loss_weight
        tf_dict["loss_d_safe1"] = loss_d_safe1 * args.safe_deriv_loss_weight
        tf_dict["loss_d_medium1"] = loss_d_medium1 * args.medium_deriv_loss_weight
        tf_dict["loss_d_dang1"] = loss_d_dang1 * args.dang_deriv_loss_weight

        tf_dict["loss_crit1"] = loss_crit1
        tf_dict["loss_grad1"] = loss_grad1
        tf_dict["loss_reg1"] = loss_reg1
        tf_dict["total_loss1"] = total_loss1

        tf_dict["acc_safe1"] = acc_safe1
        tf_dict["acc_dang1"] = acc_dang1
        tf_dict["acc_d_safe1"] = acc_d_safe1
        tf_dict["acc_d_medium1"] = acc_d_medium1
        tf_dict["acc_d_dang1"] = acc_d_dang1

        tf_dict["num_safe1"] = num_safe1
        tf_dict["num_dang1"] = num_dang1
        tf_dict["num_medium1"] = num_medium1

    if policy2 is not None:
        tf_dict["loss_safe2"] = loss_safe2 * args.safe_loss_weight
        tf_dict["loss_dang2"] = loss_dang2 * args.dang_loss_weight
        tf_dict["loss_d_safe2"] = loss_d_safe2 * args.safe_deriv_loss_weight
        tf_dict["loss_d_medium2"] = loss_d_medium2 * args.medium_deriv_loss_weight
        tf_dict["loss_d_dang2"] = loss_d_dang2 * args.dang_deriv_loss_weight

        tf_dict["loss_crit2"] = loss_crit2
        tf_dict["loss_grad2"] = loss_grad2
        tf_dict["loss_reg2"] = loss_reg2
        tf_dict["total_loss2"] = total_loss2

        tf_dict["acc_safe2"] = acc_safe2
        tf_dict["acc_dang2"] = acc_dang2
        tf_dict["acc_d_safe2"] = acc_d_safe2
        tf_dict["acc_d_medium2"] = acc_d_medium2
        tf_dict["acc_d_dang2"] = acc_d_dang2

        tf_dict["num_safe2"] = num_safe2
        tf_dict["num_dang2"] = num_dang2
        tf_dict["num_medium2"] = num_medium2

    if refinement:
        return pl_dict, tf_dict, u_init1, u_res_new1, u_init2, u_res_new2

    else:
        for key, value in tf_dict.items():
            if "debug" not in key:
                tf.summary.scalar(key, value)
        summary_op = tf.summary.merge_all()

        return pl_dict, tf_dict, train_op1, train_op2, summary_op


def dynamics1(args, state, control, primal_control):
    n_nei_feat = 4
    N = args.num_neighbors
    next_state = state * 1.0
    simT = 1.0 / args.fps
    dt = simT / args.cbf_discrete_num

    ax = control[:, 0] + primal_control[:, 0]
    ay = control[:, 1] + primal_control[:, 1]

    ax_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ax, [N]), [N, -1])), [-1])
    ay_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ay, [N]), [N, -1])), [-1])

    for t in range(args.cbf_discrete_num):
        nei_feat = tf.reshape(next_state[:, :-2], [-1, n_nei_feat])
        # x = x + vx * dt
        # y = y + vy * dt
        # vx = vx + ax * dt
        # vy = vy + ay * dt
        new_x = nei_feat[:, 0] + nei_feat[:, 2] * dt
        new_y = nei_feat[:, 1] + nei_feat[:, 3] * dt
        new_vx = nei_feat[:, 2] - ax_cp * dt
        new_vy = nei_feat[:, 3] - ay_cp * dt

        new_ego_vx = next_state[:, -2] + ax * dt
        new_ego_vy = next_state[:, -1] + ay * dt

        # clipping
        x_max, x_min = 30.0, -30.0
        y_max, y_min = 30.0, -30.0

        new_x = tf.clip_by_value(new_x, x_min, x_max)
        new_y = tf.clip_by_value(new_y, y_min, y_max)

        new_nei = tf.stack([new_x, new_y, new_vx, new_vy], axis=-1)

        next_state = tf.concat([tf.reshape(new_nei, [-1, N * n_nei_feat]),
                                tf.expand_dims(new_ego_vx, axis=-1),
                                tf.expand_dims(new_ego_vy, axis=-1),
                                ], axis=-1)
    return next_state


def dynamics2(args, state, control, primal_control):
    n_nei_feat = 4
    N = args.num_neighbors
    next_state = state * 1.0
    simT = 1.0 / args.fps
    dt = simT / args.cbf_discrete_num

    accel = control[:, 0] + primal_control[:, 0]
    omega = control[:, 1] + primal_control[:, 1]

    for t in range(args.cbf_discrete_num):
        nei_feat = tf.reshape(next_state[:, :-2], [-1, n_nei_feat])
        # x = x + vx * dt
        # y = y + vy * dt
        # vx = vx + ax * dt
        # vy = vy + ay * dt
        new_x = nei_feat[:, 0] + nei_feat[:, 2] * dt
        new_y = nei_feat[:, 1] + nei_feat[:, 3] * dt
        # new_vx = nei_feat[:, 2] - ax_cp * dt
        # new_vy = nei_feat[:, 3] - ay_cp * dt

        new_ego_th = next_state[:, -2] + omega * dt
        new_ego_v = next_state[:, -1] + accel * dt
        new_ego_th_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(new_ego_th, [N]), [N, -1])), [-1])
        new_ego_v_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(new_ego_v, [N]), [N, -1])), [-1])

        old_th_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(next_state[:, -2], [N]), [N, -1])), [-1])
        old_v_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(next_state[:, -1], [N]), [N, -1])), [-1])

        new_vx = nei_feat[:, 2] + old_v_cp * tf.cos(old_th_cp) - new_ego_v_cp * tf.cos(new_ego_th_cp)
        new_vy = nei_feat[:, 3] + old_v_cp * tf.sin(old_th_cp) - new_ego_v_cp * tf.sin(new_ego_th_cp)


        # clipping
        x_max, x_min = 30.0, -30.0
        y_max, y_min = 30.0, -30.0

        new_x = tf.clip_by_value(new_x, x_min, x_max)
        new_y = tf.clip_by_value(new_y, y_min, y_max)

        new_nei = tf.stack([new_x, new_y, new_vx, new_vy], axis=-1)

        next_state = tf.concat([tf.reshape(new_nei, [-1, N * n_nei_feat]),
                                tf.expand_dims(new_ego_th, axis=-1),
                                tf.expand_dims(new_ego_v, axis=-1),
                                ], axis=-1)
    return next_state

def clip_state(state, args):
    state[:, 0] = np.clip(state[:, 0], -1.0, 1.0)
    state[:, 1] = np.clip(state[:, 1], -1.0, 5.0)
    state[:, 2] = np.clip(state[:, 2], -1.0, 5.0)
    state[:, 3] = np.clip(state[:, 3], -1.0, 30.0)
    state[:, 4] = np.clip(state[:, 4], -1.0, 30.0)

    state[:, 5] = np.clip(state[:, 5], -3.2, 3.2)
    state[:, 6] = np.clip(state[:, 6], -30.0, 30.0)

    return state


def precompute_trajs(args, s1, s2, dest1, dest2):
    # s    (n_envs, 4) x,y,vx,vy
    # dest (n_envs, 4) x,y,vx,vy
    # trajs(n_envs, T, 4) x,y,vx,vy in each timestep

    dt=1.0/args.fps
    T = args.env_H * dt

    trajs1 = np.zeros((args.n_envs1, args.env_H, 4))
    vx1=(dest1[:,0]-s1[:,0])/T
    vy1=(dest1[:,1]-s1[:,1])/T

    if args.n_envs1>0:
        for t in range(args.env_H):
            trajs1[:, t, 0] = s1[:, 0] + vx1 * (t+1) * dt
            trajs1[:, t, 1] = s1[:, 1] + vy1 * (t+1) * dt
            trajs1[:, t, 2] = vx1
            trajs1[:, t, 3] = vy1

    trajs2 = np.zeros((args.n_envs2, args.env_H, 4))

    if args.n_envs2>0:
        # l2 = ((dest2[:, 0] - s2[:, 0])**2 + (dest2[:, 1] - s2[:, 1])**2)**0.5
        # th2 = dest2[:, 2] - s2[:, 2]
        # curve2 = l2 * 1
        # for i in range(args.n_envs2):
        #     if np.abs(th2[i])>1e-3:
        #         curve2[i] = th2[i] * l2[i] / np.sin(th2[i])
        #
        # omega2 = th2 / T
        # v2 = curve2 / T
        #
        # trajs2[:, 0, 0] = s2[:, 0] + v2 * np.cos(s2[:, 2]) * dt
        # trajs2[:, 0, 1] = s2[:, 1] + v2 * np.sin(s2[:, 2]) * dt
        # trajs2[:, 0, 2] = s2[:, 2] + omega2 * dt
        # trajs2[:, 0, 3] = v2
        #
        # for t in range(1, args.env_H):
        #     trajs2[:, t, 0] = trajs2[:, t-1, 0] + trajs2[:, t-1, 3] * np.cos(trajs2[:, t-1, 2]) * dt
        #     trajs2[:, t, 1] = trajs2[:, t-1, 1] + trajs2[:, t-1, 3] * np.sin(trajs2[:, t-1, 2]) * dt
        #     trajs2[:, t, 2] = s2[:, 2] + omega2 * (t+1) * dt
        #     trajs2[:, t, 3] = v2
        curr_s = np.array(s2)
        for t in range(0, args.env_H):
            l2 = ((dest2[:, 0] - curr_s[:, 0])**2 + (dest2[:, 1] - curr_s[:, 1])**2)**0.5
            v2 = l2 / (args.env_H-t)/dt

            new_sx = curr_s[:, 0] + curr_s[:, 3] * np.cos(curr_s[:, 2]) * dt
            new_sy = curr_s[:, 1] + curr_s[:, 3] * np.sin(curr_s[:, 2]) * dt
            new_th = np.arctan2(dest2[:,1] - curr_s[:,1], dest2[:, 0] - curr_s[:, 0])
            new_v = v2

            curr_s = np.stack((new_sx, new_sy, new_th, new_v), axis=-1)

            trajs2[:, t, 0] = curr_s[:, 0]
            trajs2[:, t, 1] = curr_s[:, 1]
            trajs2[:, t, 2] = curr_s[:, 2]
            trajs2[:, t, 3] = curr_s[:, 3]
            # print(t, curr_s[:, 0], curr_s[:, 1], curr_s[:, 2], curr_s[:, 3])
    return trajs1, trajs2


def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    x = Q
    x_next = Q
    max_iter = 150
    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next

def dlqr_steer(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_dare(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]


def dlqr(A, B, Q, R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.array(scipy.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A))
    return -K


def lqr_1d_controller(d, v0, v1, dt):
    a_max = 2
    a_min = -2

    # # LQR 1d
    # A = np.array([[1.0, dt], [0, 1.0]])
    # B = np.array([[0.0], [dt]])
    # Q = np.array([[1.0, 0.0], [0.0, 0.1]])
    # R = np.array([0.0001])
    # K = dlqr(A, B, Q, R)

    if dt==0.1:
        K = np.array([[-24.83505377, -11.86672548]])
    else:
        raise NotImplementedError

    # print(K.shape)

    a = K @ np.array([[-d], [v0 - v1]])
    return max(min(a, a_max), a_min)

def dest_controller(args, s, trajs, dest, t, tag, cached, k_table):  # TODO
    # s    (n_envs, 4) x,y,vx,vy
    # trajs(n_envs, T, 4) x,y,vx,vy in each timestep
    # dest (n_envs, 4) x,y,vx,vy
    # t    current time, also should consider total time left
    # uref (n_envs, 2) ax, ay

    dt = 1.0 / args.fps
    if tag=="1":
        xt=s[:, 0]
        yt=s[:, 1]
        vxt=s[:, 2]
        vyt=s[:, 3]

        xn=dest[:, 0]
        yn=dest[:, 1]

        left_t = (args.env_H - t) * dt

        if args.dest_controller_type=="dest":
            if left_t==0:
                uref = np.zeros((args.n_envs1, 2))
            else:
                ax = 2 * (xn - xt - vxt * left_t) / left_t / left_t
                ay = 2 * (yn - yt - vyt * left_t) / left_t / left_t
                uref = np.stack((ax,ay), axis=-1)

        elif args.dest_controller_type=="even":
            sum_dt1 = 0
            sum_dt2 = 0
            ttt1=time.time()

            uref = np.zeros((args.n_envs1, 2))
            for i in range(args.n_envs1):
                s1 = trajs[i, min(t , args.env_H-1), :]
                ctrl_ax = float(lqr_1d_controller(s1[0] - xt[i], vxt[i], s1[2], dt))
                ctrl_ay = float(lqr_1d_controller(s1[1] - yt[i], vyt[i], s1[3], dt))
                uref[i, 0] = ctrl_ax
                uref[i, 1] = ctrl_ay
                # if i==0:
                #     print(s1, ctrl_ax, ctrl_ay)


            ttt2=time.time()
            sum_dt2 = ttt2-ttt1
        else:
            raise NotImplementedError
    elif tag=="2":
        xt=s[:,0]
        yt=s[:,1]
        tht=s[:,2]
        vt=s[:,3]

        xn=dest[:,0]
        yn=dest[:,1]
        thn=dest[:,2]

        left_t = (args.env_H - t) * dt

        if args.dest_controller_type=="dest":
            if left_t==0:
                uref = np.zeros((args.n_envs2, 2))
            else:
                l=((xn-xt)**2+(yn-yt)**2)**0.5
                dth=thn-tht
                accel=2*(l-vt*left_t)/left_t/left_t
                omega=dth/left_t
                uref=np.stack((accel, omega), axis=-1)
        elif args.dest_controller_type=="even":
            uref = np.zeros((args.n_envs2, 2))


            sum_dt1=0
            sum_dt2=0

            for i in range(args.n_envs2):
                ttt1=time.time()
                ind, e = calc_nearest_index(s[i, 0:2], trajs[i, :, 0], trajs[i, :, 1], trajs[i, :, 2], i)
                ttt2=time.time()
                th_e = pi_2_pi(s[i, 2] - trajs[i, ind, 2])
                v = s[i, 3]

                pe = cached[i]["pe"]
                pth_e= cached[i]["pth_e"]

                x = np.zeros((5, 1))
                x[0, 0] = e
                x[1, 0] = (e - pe) / dt
                x[2, 0] = th_e
                x[3, 0] = (th_e - pth_e) / dt
                x[4, 0] = v - trajs[i, ind, 3]

                # if i==0:
                #     print("x=",x.flatten())

                A = np.zeros((5, 5))
                A[0, 0] = 1.0
                A[0, 1] = dt
                A[1, 2] = v
                A[2, 2] = 1.0
                A[2, 3] = dt
                A[4, 4] = 1.0

                B = np.zeros((5, 2))
                B[3, 1] = 1
                B[4, 0] = dt

                lqr_Q = np.eye(5)
                lqr_R = np.eye(2)

                # print(A, B)
                ttt3=time.time()
                # K, _, _ = dlqr_steer(A, B, lqr_Q, lqr_R)


                assert args.fps==10
                v_str="%.3f"%(v)
                if v_str in k_table:
                    K = k_table[v_str]
                else:
                    if v<-25:
                        print("v too small: %.4f"%v)
                        K=k_table["-25.000"]
                    else:
                        print("v too big: %.4f" % v)
                        K=k_table["24.999"]

                ustar = -K @ x
                uref[i, 0] = ustar[0, 0]
                uref[i, 1] = ustar[1, 0]

                ttt4=time.time()
                cached[i]["pe"]=e
                cached[i]["pth_e"]=th_e

                sum_dt1 += ttt2-ttt1
                sum_dt2 += ttt4-ttt3

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # if tag=="2":
    #     print("ref-ctrl",sum_dt1, sum_dt2)

    return uref

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
def calc_nearest_index(state, cx, cy, cyaw, i):
    dx = [state[0] - icx for icx in cx]
    dy = [state[1] - icy for icy in cy]

    # if i==0:
    #     print("debug current state: %.4f %.4f traj:"%(state[0], state[1]))
    #     for j in range(len(cx)):
    #         print(j,cx[j],cy[j],cyaw[j])

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = np.sqrt(mind)

    dxl = cx[ind] - state[0]
    dyl = cy[ind] - state[1]

    angle = pi_2_pi(cyaw[ind] - np.arctan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def collect_samples(itr, env, policy1, policy2, policy_ref, args, debug_args, obs_mean, obs_var,
                    refinement=False, refinement_cache=None, k_table=None, saved_data=None):
    running_paths = [None] * (args.n_envs1 + args.n_envs2)
    paths=[]

    ps_intervals = get_ext_indices(args.ps_intervals)
    cbf_intervals = get_ext_indices(args.cbf_intervals)
    obses=env.reset()

    dones = [True] * (args.n_envs1 + args.n_envs2)
    bs = args.batch_size // (args.n_envs1 + args.n_envs2) // args.env_H
    for i in range(bs):
        # tt1=time.time()
        if not args.zero_reference:
            if args.dest_controller_type=="none":
                policy_ref.reset(dones)
            else:
                ctrl_trajs1, ctrl_trajs2 = precompute_trajs(args,
                        env._wrapped_env.start_state1, env._wrapped_env.start_state2,
                        env._wrapped_env.end_state1, env._wrapped_env.end_state2)

                env._wrapped_env.ref_traj = ctrl_trajs2

        for idx in range(args.n_envs1 + args.n_envs2):
            running_paths[idx] = dict(
                observations=[],
                actions=[],
                env_infos=[],
                agent_infos=[],
            )

        cached = [{"pe":0.0, "pth_e":0.0} for _ in range(args.n_envs2)]

        # print("before launching", env._wrapped_env.curr_state2[:, :2])

        # tt2 = time.time()

        # sum_dt1=0
        # sum_dt2=0
        # sum_dt3=0
        # sum_dt4 = 0

        # saved_data init
        if args.save_traj_data:  # TODO(video)
            saved_data["reactive_ids1"] = env._wrapped_env.egoids1
            saved_data["reactive_ids2"] = env._wrapped_env.egoids2


        for j in range(args.env_H):
            # ttt1=time.time()

            state = obses[:, cbf_intervals]
            if args.input_clip:
                state = clip_state(state, args)
                obses[:, cbf_intervals] = state

            state1 = state[:args.n_envs1]
            state2 = state[args.n_envs1:]

            if not args.zero_reference:
                if args.dest_controller_type=="none":  # TODO
                    ps_input = obses[:, ps_intervals]
                    ps_input = (ps_input - obs_mean) / (np.sqrt(obs_var) + 1e-8)
                    uref, uref_info = policy_ref.get_actions(ps_input)
                    uref = uref * np.array([[4.0, 4.0]])
                else:
                    # curr_state = env._wrapped_env.curr_state
                    # print("inner", j, curr_state)
                    curr_state1 = env._wrapped_env.curr_state1
                    curr_state2 = env._wrapped_env.curr_state2
                    # print("get current state2", curr_state2[:, :2])
                    uref1 = dest_controller(args, curr_state1, ctrl_trajs1, env._wrapped_env.end_state1, j, tag="1",
                                            cached=None, k_table=None)
                    # print("collect_smaples",len(k_table.keys()))
                    uref2 = dest_controller(args, curr_state2, ctrl_trajs2, env._wrapped_env.end_state2, j, tag="2",
                                            cached=cached, k_table=k_table)

                    uref = np.concatenate((uref1, uref2), axis=0)
                    # print("current_state", curr_state2[:, :2])
                    # print("current ref", uref)

            else:
                uref = np.zeros((args.n_envs1 + args.n_envs2, 2))

            # ttt2 = time.time()

            if args.zero_policy:
                agent_infos={}
                actions = np.zeros((args.n_envs1 + args.n_envs2, 2))
            else:
                if policy1 is not None:
                    if args.include_u_ref_feat:
                        actions1, agent_infos1 = policy1.get_actions_flat_with_uref(state1, uref1)
                    else:
                        actions1, agent_infos1 = policy1.get_actions_flat(state1)
                else:
                    actions1 = np.zeros((0, 2))
                if policy2 is not None:
                    if args.include_u_ref_feat:
                        actions2, agent_infos2 = policy2.get_actions_flat_with_uref(state2, uref2)
                    else:
                        actions2, agent_infos2 = policy2.get_actions_flat(state2)
                else:
                    actions2 = np.zeros((0, 2))

                actions = np.concatenate((actions1, actions2), axis=0)

                agent_infos={}
                for kkk in agent_infos1:
                    agent_infos[kkk] = np.concatenate((agent_infos1[kkk], agent_infos2[kkk]), axis=0)

            # ttt3 = time.time()
            # if j==0:
            #     print("actions", actions.T, "uref", uref.T)
            actions = actions + uref
            agent_infos['ref_actions'] = uref
            # print("after merging", actions)

            # TODO refinement (only used for testing)
            if refinement:
                if j>0:
                    env_out_of_lane = np.stack([env_infos[jj]["out_of_lane"] for jj in range(args.n_envs1+args.n_envs2)], axis=0)
                    cbf_intervals = get_ext_indices(args.cbf_intervals)
                    safe_mask, medium_mask, dang_mask = get_masks(args, state, cbf_intervals, env_out_of_lane)

                    u_init1, u_res_new1, u_init2, u_res_new2, pl_dict = refinement_cache
                    u_init_dict={}
                    u_res_new_dict={}
                    if u_init1 is not None:
                        u_init_dict["u_init1"] = u_init1
                        u_res_new_dict["u_res_new1"] = u_res_new1
                    if u_init2 is not None:
                        u_init_dict["u_init2"] = u_init2
                        u_res_new_dict["u_res_new2"] = u_res_new2
                    _ = tf.get_default_session().run(u_init_dict, feed_dict={})

                    u_res1 = 0.0 * actions1
                    u_res2 = 0.0 * actions2

                    split = args.n_envs1
                    to_feed_dict={}
                    if u_init1 is not None:
                        to_feed_dict[pl_dict["state1"]] = state[:split]
                        to_feed_dict[pl_dict["uref1"]] = uref[:split]
                        to_feed_dict[pl_dict["safe_mask1"]] = safe_mask[:split]
                        to_feed_dict[pl_dict["dang_mask1"]] = dang_mask[:split]
                        to_feed_dict[pl_dict["medium_mask1"]] = medium_mask[:split]

                    if u_init2 is not None:
                        to_feed_dict[pl_dict["state2"]] = state[split:]
                        to_feed_dict[pl_dict["uref2"]] = uref[split:]
                        to_feed_dict[pl_dict["safe_mask2"]] = safe_mask[split:]
                        to_feed_dict[pl_dict["dang_mask2"]] = dang_mask[split:]
                        to_feed_dict[pl_dict["medium_mask2"]] = medium_mask[split:]

                    # print("n_envs1 %d, n_envs2 %d env_H %d"%(args.n_envs1, args.n_envs2, args.env_H))
                    # for key in pl_dict:
                    #     print(key, to_feed_dict[pl_dict[key]].shape)

                    for ti in range(args.refine_n_iter):
                        u_res_np_dict = tf.get_default_session().run(u_res_new_dict,
                              feed_dict = to_feed_dict)

                    if u_init1 is not None:
                        u_res1 = u_res_np_dict["u_res_new1"]
                    if u_init2 is not None:
                        u_res2 = u_res_np_dict["u_res_new2"]

                    u_res = np.concatenate((u_res1, u_res2), axis=0)

                    # print(u_res)
                    # u_res = np.clip(u_res, -args.u_res_norm, args.u_res_norm)
                    if args.pre_action_clip is not None:
                        actions = np.clip(actions, -args.pre_action_clip, args.pre_action_clip)

                    actions = actions + u_res

                    # print(actions)
                    actions = np.clip(actions, -args.u_res_norm, args.u_res_norm)
                else:
                    if args.pre_action_clip is not None:
                        actions = np.clip(actions, -args.pre_action_clip, args.pre_action_clip)



            # ped_scene = copy.deepcopy(self.traj_data["Pedestrian"]["snapshots"][t])
            # cyc_scene = copy.deepcopy(self.traj_data["Biker"]["snapshots"][t])
            if args.save_traj_data:   # TODO(video)
                the_t = env._wrapped_env.t - 1
                saved_data["agents1"][the_t] = {}
                saved_data["agents2"][the_t] = {}

                # all pedestrians (but egoids) from scene
                # all egoids from scene
                # all vehs from scene
                for agi in env._wrapped_env.traj_data["Pedestrian"]["snapshots"][the_t]:
                    if agi not in env._wrapped_env.egoids1:
                        agent0 = env._wrapped_env.traj_data["Pedestrian"]["snapshots"][the_t][agi]
                    else:
                        agent0 = env._wrapped_env.ego_peds1[env._wrapped_env.backtrace[agi]]

                    saved_data["agents1"][the_t][agi] = [agi, agent0[0], agent0[1], agent0[2], agent0[3]]

                for agi in env._wrapped_env.traj_data["Biker"]["snapshots"][the_t]:
                    if agi not in env._wrapped_env.egoids2:
                        agent0 = env._wrapped_env.traj_data["Biker"]["snapshots"][the_t][agi]
                    else:
                        agent0 = env._wrapped_env.ego_peds2[env._wrapped_env.backtrace[agi]-env._wrapped_env.n_veh1]

                    saved_data["agents2"][the_t][agi] = [agi, agent0[0], agent0[1], agent0[2], agent0[3]]


            next_obs, _, dones, env_infos = env.step(actions)

            # ttt4 = time.time()

            # print("out after step", env._wrapped_env.curr_state2[:, :2])

            if args.debug_render and itr % args.debug_render_freq == 0 and j != args.env_H-1:
                env.render()

            # print("out after render", env._wrapped_env.curr_state2[:, :2])

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(args.n_envs1 + args.n_envs2)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(args.n_envs1 + args.n_envs2)]

            for idx, observation, action, env_info, agent_info, done in zip(
                itertools.count(), obses, actions, env_infos, agent_infos, dones
            ):
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

            obses = next_obs

            # ttt5 = time.time()
            _coll=[env_infos[kk]["is_colliding"] for kk in range(len(env_infos))]
            if np.sum(_coll)>0:
                print(j, _coll)
            # sum_dt1+=ttt2-ttt1
            # sum_dt2 += ttt3 - ttt2
            # sum_dt3 += ttt4 - ttt3
            # sum_dt4 += ttt5 - ttt4


        # print("sum_dt", sum_dt1, sum_dt2, sum_dt3, sum_dt4)
        #
        # tt3=time.time()

        for idx in range(args.n_envs1 + args.n_envs2):
            merged_observations=np.stack(running_paths[idx]["observations"], axis=0)
            merged_actions=np.stack(running_paths[idx]["actions"], axis=0)
            merged_env_infos={}
            merged_agent_infos = {}
            for k in running_paths[idx]["env_infos"][0]:
                merged_env_infos[k] = np.stack([tmpd[k] for tmpd in running_paths[idx]["env_infos"]], axis=0)
            for k in running_paths[idx]["agent_infos"][0]:
                merged_agent_infos[k] = np.stack([tmpd[k] for tmpd in running_paths[idx]["agent_infos"]], axis=0)

            paths.append(dict(
                observations=merged_observations,
                actions=merged_actions,
                env_infos=merged_env_infos,
                agent_infos=merged_agent_infos,
            ))
        # tt4=time.time()
        # print("inner collect:",tt2-tt1, tt3-tt2,tt4-tt3)

    return paths


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


def get_masks(args, true_obs, cbf_intervals, out_lane):
    cbf_obs = true_obs.reshape((-1, true_obs.shape[-1]))[:, cbf_intervals]
    safe_d1 = args.safe_dist_threshold
    safe_d2 = args.safe_dist_threshold_side
    dang_d1 = args.dang_dist_threshold
    dang_d2 = args.dang_dist_threshold_side

    safe_mask = get_safe_mask_agent(args, cbf_obs, out_lane, safe_d1, safe_d2, check_safe=True)
    dang_mask = get_safe_mask_agent(args, cbf_obs, out_lane, dang_d1, dang_d2, check_safe=False)
    medium_mask = np.logical_and(~safe_mask, ~dang_mask)

    safe_mask = safe_mask.astype(dtype=np.float32)
    medium_mask = medium_mask.astype(dtype=np.float32)
    dang_mask = dang_mask.astype(dtype=np.float32)

    return safe_mask, medium_mask, dang_mask

def get_safe_mask_agent(args, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
    # get the safe mask from the states
    # input: (batchsize, feat_dim)
    # return: (batchsize, 1)
    n_nei_feat = 4

    nei_s = s[:, :-2].reshape((s.shape[0], args.num_neighbors, n_nei_feat))
    nei_d = np.linalg.norm(nei_s[:, :, :2], axis=-1)
    collide = (nei_d < dist_threshold + 2 * args.ped_radius)
    dang_mask = collide >= 0.5

    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s"%(
    #     check_safe, arr_print(nei_s[0,0]), dist_threshold, args.ped_radius, dang_mask[0,0],
    # ))
    #
    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s" % (
    #     check_safe, arr_print(nei_s[0, 1]), dist_threshold, args.ped_radius, dang_mask[0, 1],
    # ))
    #
    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s" % (
    #     check_safe, arr_print(nei_s[0, 2]), dist_threshold, args.ped_radius, dang_mask[0, 2],
    # ))
    #
    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s" % (
    #     check_safe, arr_print(nei_s[0, 3]), dist_threshold, args.ped_radius, dang_mask[0, 3],
    # ))
    #
    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s" % (
    #     check_safe, arr_print(nei_s[0, 4]), dist_threshold, args.ped_radius, dang_mask[0, 4],
    # ))
    #
    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s" % (
    #     check_safe, arr_print(nei_s[0, 5]), dist_threshold, args.ped_radius, dang_mask[0, 5],
    # ))
    #
    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s" % (
    #     check_safe, arr_print(nei_s[0, 6]), dist_threshold, args.ped_radius, dang_mask[0, 6],
    # ))
    #
    # print("check-safe:%s s:%s <%.2f+2*%.2f mask:%s" % (
    #     check_safe, arr_print(nei_s[0, 7]), dist_threshold, args.ped_radius, dang_mask[0, 7],
    # ))
    #
    # print()

    if check_safe:  # no collision for each row, means that ego vehicle is safe
        return np.logical_not(dang_mask)
    else:  # one collision in rows means that ego vehicle is dang
        return dang_mask


def optimize_policy(itr, pl_dict, tf_dict, train_op1, train_op2, summary_op, paths, summary_writer, args):
    obs = np.concatenate([path["observations"] for path in paths], axis=0)
    agent_infos={}
    env_infos={}

    for k in paths[0]["agent_infos"]:
        agent_infos[k] = np.concatenate([path["agent_infos"][k] for path in paths], axis=0)

    # for path in paths:
    #     print(path["env_infos"].keys())

    for k in paths[0]["env_infos"]:
        env_infos[k] = np.concatenate([path["env_infos"][k] for path in paths], axis=0)
    if args.zero_policy:
        np_dict={}
    else:
        env_out_of_lane = env_infos["out_of_lane"]

        cbf_intervals = get_ext_indices(args.cbf_intervals)

        safe_mask, medium_mask, dang_mask = get_masks(args, obs, cbf_intervals, env_out_of_lane)

        state = obs[:, cbf_intervals]
        uref = agent_infos["ref_actions"]

        train_op_list=[]
        to_feed_dict={}

        split = args.n_envs1 * args.env_H
        if train_op1 is not None:
            train_op_list.append(train_op1)
            to_feed_dict[pl_dict["state1"]] = state[:split]
            to_feed_dict[pl_dict["uref1"]] = uref[:split]
            to_feed_dict[pl_dict["safe_mask1"]] = safe_mask[:split]
            to_feed_dict[pl_dict["dang_mask1"]] = dang_mask[:split]
            to_feed_dict[pl_dict["medium_mask1"]] = medium_mask[:split]

        if train_op2 is not None:
            train_op_list.append(train_op2)
            to_feed_dict[pl_dict["state2"]] = state[split:]
            to_feed_dict[pl_dict["uref2"]] = uref[split:]
            to_feed_dict[pl_dict["safe_mask2"]] = safe_mask[split:]
            to_feed_dict[pl_dict["dang_mask2"]] = dang_mask[split:]
            to_feed_dict[pl_dict["medium_mask2"]] = medium_mask[split:]


        np_dict, _, _, sum_op = tf.get_default_session().run(
            [tf_dict] + train_op_list +[summary_op],
            feed_dict=to_feed_dict)

        # TODO DEBUG
        # if args.debug_easy:
        #     for debug_i in range(args.env_H):
        #         logger.log("%d (%.4f %.4f %.4f %.4f %.4f)  [%s] [%s] %d %.4f %.4f | %d %.4f %.4f  (%.4f %.4f"%(
        #             debug_i, state[debug_i, 6], state[debug_i, 13], state[debug_i, 15],
        #             state[debug_i, 40], state[debug_i, 42],
        #             arr_print(np_dict["debug_state_first"][debug_i, 0]),
        #             arr_print(np_dict["debug_state_first"][debug_i, 1]),
        #             safe_mask[debug_i, 0] + medium_mask[debug_i, 0] * 1 + dang_mask[debug_i, 0] * 2,
        #             np_dict["debug_score"][debug_i, 0],
        #             np_dict["debug_score_next"][debug_i, 0],
        #             safe_mask[debug_i, 1] + medium_mask[debug_i, 1] * 1 + dang_mask[debug_i, 1] * 2,
        #             np_dict["debug_score"][debug_i, 1],
        #             np_dict["debug_score_next"][debug_i, 1],
        #             np_dict["debug_u"][debug_i, 0],
        #             np_dict["debug_u"][debug_i, 1],
        #         ), with_prefix=False, with_timestamp=False)

        if args.debug_easy:
            for debug_i in range(args.env_H):
                # logger.log("%d (%.4f %.4f |%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f) "
                #            "pol(%s) [%s] [%s] %d %.4f %.4f | %d %.4f %.4f  (%.4f %.4f"%(
                #     debug_i, state[debug_i, 32], state[debug_i, 33], state[debug_i, 0],
                #     state[debug_i, 1], state[debug_i, 2], state[debug_i, 3],
                #     state[debug_i, 4], state[debug_i, 5], state[debug_i, 6], state[debug_i, 7],
                #     arr_print(np_dict["debug_policy_fusion"][debug_i, -6:]),
                #     arr_print(np_dict["debug_state_first"][debug_i, 0]),
                #     arr_print(np_dict["debug_state_first"][debug_i, 1]),
                #     safe_mask[debug_i, 0]*0 + medium_mask[debug_i, 0] * 1 + dang_mask[debug_i, 0] * 2,
                #     np_dict["debug_score"][debug_i, 0],
                #     np_dict["debug_score_next"][debug_i, 0],
                #     safe_mask[debug_i, 1]*0 + medium_mask[debug_i, 1] * 1 + dang_mask[debug_i, 1] * 2,
                #     np_dict["debug_score"][debug_i, 1],
                #     np_dict["debug_score_next"][debug_i, 1],
                #     np_dict["debug_u"][debug_i, 0],
                #     np_dict["debug_u"][debug_i, 1],
                # ), with_prefix=False, with_timestamp=False)
                # logger.log("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s"%(
                #     arr_print(np_dict["debug_state_policy"][debug_i, 0]),
                #     arr_print(np_dict["debug_state_policy"][debug_i, 1]),
                #     arr_print(np_dict["debug_state_policy"][debug_i, 2]),
                #     arr_print(np_dict["debug_state_policy"][debug_i, 3]),
                #     arr_print(np_dict["debug_state_policy"][debug_i, 4]),
                #     arr_print(np_dict["debug_state_policy"][debug_i, 5]),
                #     arr_print(np_dict["debug_state_policy"][debug_i, 6]),
                #     arr_print(np_dict["debug_state_policy"][debug_i, 7]),
                # ), with_prefix=False, with_timestamp=False)
                _=1

            #todo where does most cbf detect wrong
            score=np_dict["debug_score"][:, 0]
            mask0=safe_mask[:, 0]
            mask1=dang_mask[:, 0]

            gap = -score * mask0 + score * mask1
            gap = gap * (1-medium_mask[:, 0])

            indices = (-gap).argsort()[:50]
            for ii, ind in enumerate(indices):
                logger.log("i=%d gap:%.4f state:%s mask:%d score:%.4f"%(
                    ii, gap[ind], arr_print(np_dict["debug_state_first"][ind]), mask0[ind]*0 + mask1[ind],
                    np_dict["debug_score"][ind]
                ))
        # TODO END of DEBUG

    out_of_lane = np.mean(env_infos["out_of_lane"])
    is_colliding = np.mean(env_infos["is_colliding"])

    new_coll = np.mean(np.sum(env_infos["is_colliding"].reshape(-1, args.env_H), axis=1) > 0.5)
    rmse = np.mean(env_infos["rmse_pos"])

    np_dict["ool"] = out_of_lane
    np_dict["coll"] = is_colliding
    np_dict["coll1"] = new_coll
    np_dict["rmse"] = rmse

    split=args.n_envs1
    np_dict["coll_ped"] = np.mean(env_infos["is_colliding"].reshape(-1, args.env_H)[:split])
    np_dict["coll1_ped"] = np.mean(np.sum(env_infos["is_colliding"].reshape(-1, args.env_H)[:split], axis=1) > 0.5)
    np_dict["coll_cyc"] =np.mean(env_infos["is_colliding"].reshape(-1, args.env_H)[split:])
    np_dict["coll1_cyc"] = np.mean(np.sum(env_infos["is_colliding"].reshape(-1, args.env_H)[split:], axis=1) > 0.5)

    # new ped/cyc RMSE
    np_dict["rmse_ped"] = np.mean(env_infos["rmse_pos"].reshape(-1, args.env_H)[:split])
    np_dict["rmse_cyc"] = np.mean(env_infos["rmse_pos"].reshape(-1, args.env_H)[split:])

    if not args.zero_policy:
        summary = tf.Summary()
        summary.value.add(tag='0_out_lane', simple_value=out_of_lane)
        summary_writer.add_summary(summary, itr)

        summary = tf.Summary()
        summary.value.add(tag='1_collision', simple_value=is_colliding)
        summary_writer.add_summary(summary, itr)

        summary = tf.Summary()
        summary.value.add(tag='2_traj_coll', simple_value=new_coll)
        summary_writer.add_summary(summary, itr)

        summary = tf.Summary()
        summary.value.add(tag='3_rmse_pos', simple_value=rmse)
        summary_writer.add_summary(summary, itr)


        summary_writer.add_summary(sum_op, itr)
        summary_writer.flush()

    return np_dict

def feature_encoding(x, args, signed=True, uref=None):
    # ego features
    assert args.record_vxvy # we need vx,vy features, probably for policy net
    ego_vx = x[:, 4*args.num_neighbors + 0]
    ego_vy = x[:, 4*args.num_neighbors + 1]

    nei_x = x[:, :-2]
    bs = tf.shape(x)[0]
    nei_x = tf.reshape(nei_x, [bs, args.num_neighbors, 4])

    nei_dx = nei_x[:, :, 0]
    nei_dy = nei_x[:, :, 1]
    nei_vx = nei_x[:, :, 2]
    nei_vy = nei_x[:, :, 3]
    nei_feat=[]
    for i in range(args.num_neighbors):
        nei_feat_each = tf.stack([ego_vx, ego_vy, nei_dx[:, i], nei_dy[:, i], nei_vx[:, i], nei_vy[:, i]], axis=-1)
        nei_feat.append(nei_feat_each)
    nei_feat = tf.stack(nei_feat, axis=1)
    return nei_feat


def get_obs_mask(x, args):
    # c,lld,rld,lrd,rrd,th,v,a,w,L,W,
    # ind, dx, dy, dth, v, a,w,L,W
    x_nei = tf.reshape(x[:, :-2], [-1, args.num_neighbors, 4])

    valid = tf.less_equal(tf.norm(x_nei[:, :, :2], axis=-1), args.obs_radius)

    x_mask = tf.cast(valid, tf.float32)
    return x_mask  # shape (N, 6)

def network_cbf(x, args, tf_dict, dbg_name):

    if args.enable_radius:
        x_mask = get_obs_mask(x, args)

    cat_x = feature_encoding(x, args)
    if args.dist_only:
        cat_x = tf.norm(cat_x[:, :, 2:4], axis=-1, keepdims=True)
    elif args.dist2_only:
        cat_x = cat_x[:, :, 2:4]
    elif args.dist3_only:
        cat_x = tf.concat([
            tf.norm(cat_x[:, :, 2:4], axis=-1, keepdims=True) - 2 * args.ped_radius - args.safe_dist_threshold,
            tf.norm(cat_x[:, :, 2:4], axis=-1, keepdims=True) - 2 * args.ped_radius - args.dang_dist_threshold
        ], axis=-1)
    if args.debug_easy:
        tf_dict["debug_state_%s"%dbg_name] = cat_x

    for i, hidden_num in enumerate(args.jcbf_hidden_layer_dims):
        cat_x = tf.contrib.layers.conv1d(inputs=cat_x,
                                         num_outputs=hidden_num,
                                         kernel_size=1,
                                         reuse=tf.AUTO_REUSE,
                                         scope='cbf/conv%d' % i,
                                         activation_fn=tf.nn.relu)

    cat_x = tf.contrib.layers.conv1d(inputs=cat_x,
                                     num_outputs=1,
                                     kernel_size=1,
                                     reuse=tf.AUTO_REUSE,
                                     scope='cbf/conv%d' % len(args.jcbf_hidden_layer_dims),
                                     activation_fn=None)
    cat_x = tf.squeeze(cat_x, axis=-1)  # TODO(modified)

    if args.enable_radius:
        cat_x = cat_x * x_mask

    return cat_x  # TODO (bs, num_nei)


class MyPolicy:
    def __init__(self, env, args, name):
        self.args = args
        self.recurrent = True  # used in many places - just set to True
        self.vectorized = True  # used in batch_polopt.py
        self.name = name

    def get_action_tf_flat(self, tf_dict, state_input):  # TODO overwrite
        self.gt_obs = state_input
        self.action = network_policy(state_input, tf_dict, self.args, self.name)
        return self.action

    def get_action_tf_flat_with_uref(self, tf_dict, state_input, uref):  # TODO overwrite
        self.gt_obs = state_input
        self.uref = uref
        self.action = network_policy(state_input, tf_dict, self.args, self.name, uref)
        return self.action

    def get_actions_flat(self, gt_obs):
        session = tf.get_default_session()
        action, = session.run([self.action], feed_dict={
            self.gt_obs: gt_obs,
        })

        agent_infos = {"mean": action}
        return action, agent_infos

    def get_actions_flat_with_uref(self, gt_obs, uref):
        session = tf.get_default_session()
        action, = session.run([self.action], feed_dict={
            self.gt_obs: gt_obs,
            self.uref: uref,
        })

        agent_infos = {"mean": action}
        return action, agent_infos

    def get_params(self, trainable):
        return [v for v in tf.trainable_variables() if "%s/"%(self.name) in v.name]

    def set_param_values(self, params):
        # utils_cbf.set_policy_param_values(params, self.args)
        var_list = self.get_params(trainable=True)
        assign = tf.group(*[tf.assign(var, val) for (var, val) in zip(var_list, params)])
        session = tf.get_default_session()
        session.run(assign)

    def get_policy_param_values(self, args):
        var_list = self.get_params(trainable=True)
        session = tf.get_default_session()
        return [session.run(v) for v in var_list]


def network_policy(x, tf_dict, args, net_name, uref=None):
    if args.enable_radius:
        x_mask = get_obs_mask(x, args)
    nei_feat = feature_encoding(x, args, uref=uref)
    if args.debug_easy:
        tf_dict["debug_state_policy"] = nei_feat
        if args.enable_radius:
            tf_dict["debug_x_mask"] = x_mask

    for i, hidden_num in enumerate(args.policy_hidden_layer_dims):
        nei_feat = tf.contrib.layers.conv1d(inputs=nei_feat,
                                         num_outputs=hidden_num,
                                         kernel_size=1,
                                         reuse=tf.AUTO_REUSE,
                                         scope='%s/conv%d' % (net_name,i),
                                         activation_fn=tf.nn.relu)
    if args.enable_radius:
        nei_feat = tf.reduce_max(nei_feat*tf.expand_dims(x_mask, axis=-1), reduction_indices=[1])
    else:
        nei_feat = tf.reduce_max(nei_feat, reduction_indices=[1])

    if uref is not None and args.include_u_ref_feat:
        if args.include_speed_feat:
            nei_feat = tf.concat([nei_feat, uref, x[:, -2:]], axis=-1)
        else:
            nei_feat = tf.concat([nei_feat, uref], axis=-1)
    else:
        if args.include_speed_feat:
            nei_feat = tf.concat([nei_feat, x[:, -2:]], axis=-1)
    tf_dict["debug_policy_fusion"] = nei_feat

    x = nei_feat
    for i, hidden_num in enumerate(args.policy_hidden_fusion_dims):
        x = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=hidden_num,
            reuse=tf.AUTO_REUSE,
            scope='%s/convII%d' % (net_name,i),
            activation_fn=tf.nn.relu)
    act = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=2,
        reuse=tf.AUTO_REUSE,
        scope='%s/dense%d' % (net_name,len(args.policy_hidden_layer_dims)+len(args.policy_hidden_fusion_dims)),
        activation_fn=None)

    assert not args.scaling_action_output
    assert not args.clip_action_output
    assert not  args.debug_accel_only

    if args.zero_policy:
        act = act * 0.0

    return act


class MockEnv:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

def arr_print(arr, format="%.4f"):
    s = " ".join([format%item for item in arr])
    return s

def smart_add(a, b):
    if a>=0 and b>=0:
        return (a+b)/2
    elif a<0 and b<0:
        return -1
    elif a<0:
        return b
    else:
        return a

def main():
    t0=time.time()
    args = configs.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.exp_name = utils.get_exp_name(args.exp_name)
    exp_dir = ospj(utils.get_exp_home(), args.exp_name)
    log_dir = ospj(exp_dir, 'imitate', 'log')
    os.makedirs(log_dir, exist_ok=True)
    logger.add_text_output(ospj(exp_dir, 'log.txt'))
    logger.add_text_output(ospj(log_dir, 'log.txt'))
    args.full_log_dir = log_dir

    np.savez(ospj(log_dir, 'args'), args=args)
    summary_writer = tf.summary.FileWriter(exp_dir)

    utils.write_cmd_to_file(log_dir, sys.argv)

    debug_args = utils.match_for_args_dict(args)

    env, act_low, act_high = utils.build_pedcyc_env(args, debug_args)

    ps_indices = get_ext_indices(args.ps_intervals)
    ref_obs_space = rllab.spaces.Box(low=env.spec.observation_space.low[ps_indices], high=env.spec.observation_space.high[ps_indices])
    ref_action_space = env.spec.action_space
    ref_env = MockEnv(ref_obs_space, ref_action_space)
    if args.zero_reference and args.dest_controller_type=="none":
        policy_ref = GaussianGRUPolicy(
                    name="policy_ref",
                    env_spec=ref_env,
                    hidden_dim=args.recurrent_hidden_dim,
                    output_nonlinearity=None,
                    learn_std=True,
                    args=args
                )
    else:
        policy_ref=None

    if args.control_mode == "ped_cyc" or args.control_mode == "ped_only":
        policy1 = MyPolicy(env, args, name="myp1")
    else:
        policy1 = None
    if args.control_mode == "ped_cyc" or args.control_mode == "cyc_only":
        policy2 = MyPolicy(env, args, name="myp2")
    else:
        policy2 = None

    pl_dict, tf_dict, train_op1, train_op2, summary_op = build_computation_graph(policy1, policy2, policy_ref, args)

    ool_list = []
    coll_list = []
    coll1_list = []

    coll_ped_list=[]
    coll_cyc_list=[]
    coll1_ped_list = []
    coll1_cyc_list = []

    rmse_list = []

    rmse_ped_list=[]
    rmse_cyc_list=[]



    k_table = np.load("K_lqr_v_t0.1000.npz")["table"].item()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        obs_mean, obs_var = load_pretrained_weights(args, policy1, policy2, policy_ref)
        summary_writer.add_graph(sess.graph)

        params = dict()
        params['jcbfer'] = utils_cbf.get_cbf_param_values(args)
        if policy1 is not None:
            params['policy1'] = policy1.get_policy_param_values(args)
        if policy2 is not None:
            params['policy2'] = policy2.get_policy_param_values(args)
        h_utils.save_params(log_dir, params, 0, max_to_keep=10000)

        # for iterations
        for ep_i in range(args.n_itr):
            t1=time.time()
            paths = collect_samples(ep_i, env, policy1, policy2, policy_ref, args, debug_args, obs_mean, obs_var, k_table=k_table)
            # tt1=time.time()
            np_dict = optimize_policy(ep_i, pl_dict, tf_dict, train_op1, train_op2, summary_op, paths, summary_writer, args)
            # tt2=time.time()
            ool_list.append(np_dict["ool"])
            coll_list.append(np_dict["coll"])
            coll1_list.append(np_dict["coll1"])
            rmse_list.append(np_dict["rmse"])

            coll_ped_list.append(np_dict["coll_ped"])
            coll1_ped_list.append(np_dict["coll1_ped"])
            coll_cyc_list.append(np_dict["coll_cyc"])
            coll1_cyc_list.append(np_dict["coll1_cyc"])

            rmse_ped_list.append(np_dict["rmse_ped"])
            rmse_cyc_list.append(np_dict["rmse_cyc"])

            # print("for loop time",tt1-t1, tt2-tt1)
            if args.zero_policy:
                logger.log("%03d %02d:%02d:%02d col:%.4f(%.4f) %.2f(%.2f) e:%.2f(%.2f)  %.4f(%.4f) %.4f(%.4f) %.4f(%.4f)| %.4f(%.4f) %.4f(%.4f) %.4f(%.4f)"%(
                    ep_i,
                    (t1 - t0) // 3600,
                    ((t1 - t0) % 3600) // 60,
                    (t1 - t0) % 60, coll_list[-1], np.mean(coll_list),
                    coll1_list[-1], np.mean(coll1_list), rmse_list[-1], np.mean(rmse_list),
                    coll_ped_list[-1], np.mean(coll_ped_list), coll1_ped_list[-1], np.mean(coll1_ped_list),
                    rmse_ped_list[-1], np.mean(rmse_ped_list),
                    coll_cyc_list[-1], np.mean(coll_cyc_list), coll1_cyc_list[-1], np.mean(coll1_cyc_list),
                    rmse_cyc_list[-1], np.mean(rmse_cyc_list),
                ), with_prefix=False, with_timestamp=False)
            else:

                if args.control_mode == "ped_cyc":
                    np_total_loss = np_dict["total_loss1"] + np_dict["total_loss2"]
                    np_loss_crit =  np_dict["loss_crit1"] + np_dict["loss_crit2"]
                    np_loss_grad = np_dict["loss_grad1"] + np_dict["loss_grad2"]
                    np_loss_reg = np_dict["loss_reg1"] + np_dict["loss_reg2"]
                    np_num_safe = np_dict["num_safe1"] + np_dict["num_safe2"]
                    np_num_medium = np_dict["num_medium1"] + np_dict["num_medium2"]
                    np_num_dang = np_dict["num_dang1"] + np_dict["num_dang2"]
                    np_acc_safe = smart_add(np_dict["acc_safe1"], np_dict["acc_safe2"])
                    np_acc_dang = smart_add(np_dict["acc_dang1"], np_dict["acc_dang2"])
                    np_acc_d_safe = smart_add(np_dict["acc_d_safe1"], np_dict["acc_d_safe2"])
                    np_acc_d_medium = smart_add(np_dict["acc_d_medium1"], np_dict["acc_d_medium2"])
                    np_acc_d_dang = smart_add(np_dict["acc_d_dang1"], np_dict["acc_d_dang2"])

                elif args.control_mode == "ped_only":
                    np_total_loss = np_dict["total_loss1"]
                    np_loss_crit = np_dict["loss_crit1"]
                    np_loss_grad = np_dict["loss_grad1"]
                    np_loss_reg = np_dict["loss_reg1"]
                    np_num_safe = np_dict["num_safe1"]
                    np_num_medium = np_dict["num_medium1"]
                    np_num_dang = np_dict["num_dang1"]
                    np_acc_safe = np_dict["acc_safe1"]
                    np_acc_dang = np_dict["acc_dang1"]
                    np_acc_d_safe = np_dict["acc_d_safe1"]
                    np_acc_d_medium = np_dict["acc_d_medium1"]
                    np_acc_d_dang = np_dict["acc_d_dang1"]
                else:
                    np_total_loss = np_dict["total_loss2"]
                    np_loss_crit = np_dict["loss_crit2"]
                    np_loss_grad = np_dict["loss_grad2"]
                    np_loss_reg = np_dict["loss_reg2"]
                    np_num_safe = np_dict["num_safe2"]
                    np_num_medium = np_dict["num_medium2"]
                    np_num_dang = np_dict["num_dang2"]
                    np_acc_safe = np_dict["acc_safe2"]
                    np_acc_dang = np_dict["acc_dang2"]
                    np_acc_d_safe = np_dict["acc_d_safe2"]
                    np_acc_d_medium = np_dict["acc_d_medium2"]
                    np_acc_d_dang = np_dict["acc_d_dang2"]

                # monitoring (training-related)
                logger.log('%03d %02d:%02d:%02d L %.4f c %.4f g %.4f r %.4f [%4d %4d %4d] acc %.3f %.3f d %.3f %.3f %.3f '
                           'col:%.4f(%.4f) %.2f(%.2f) e:%.2f(%.2f) %.4f(%.4f) %.4f(%.4f) %.4f(%.4f)| %.4f(%.4f) %.4f(%.4f) %.4f(%.4f)' % (
                    ep_i,
                    (t1 - t0) // 3600,
                    ((t1 - t0) % 3600)//60,
                    (t1 - t0) % 60,
                    # np_dict["total_loss1"] + np_dict["total_loss2"],
                    # np_dict["loss_crit1"] + np_dict["loss_crit2"],
                    # np_dict["loss_grad1"] + np_dict["loss_grad2"],
                    # np_dict["loss_reg1"] + np_dict["loss_reg2"],
                    # np_dict["num_safe1"] + np_dict["num_safe2"],
                    # np_dict["num_medium1"] + np_dict["num_medium2"],
                    # np_dict["num_dang1"] + np_dict["num_dang2"],
                    # smart_add(np_dict["acc_safe1"], np_dict["acc_safe2"]),
                    # smart_add(np_dict["acc_dang1"], np_dict["acc_dang2"]),
                    # smart_add(np_dict["acc_d_safe1"], np_dict["acc_d_safe2"]),
                    # smart_add(np_dict["acc_d_medium1"], np_dict["acc_d_medium2"]),
                    # smart_add(np_dict["acc_d_dang1"], np_dict["acc_d_dang2"]),
                    np_total_loss, np_loss_crit, np_loss_grad, np_loss_reg,
                    np_num_safe, np_num_medium, np_num_dang,
                    np_acc_safe, np_acc_dang, np_acc_d_safe, np_acc_d_medium, np_acc_d_dang,
                    coll_list[-1], np.mean(coll_list), coll1_list[-1], np.mean(coll1_list), rmse_list[-1], np.mean(rmse_list),
                    coll_ped_list[-1], np.mean(coll_ped_list), coll1_ped_list[-1], np.mean(coll1_ped_list),
                    rmse_ped_list[-1], np.mean(rmse_ped_list),
                    coll_cyc_list[-1], np.mean(coll_cyc_list), coll1_cyc_list[-1], np.mean(coll1_cyc_list),
                    rmse_cyc_list[-1], np.mean(rmse_cyc_list),
                ), with_prefix=False, with_timestamp=False)

                # save models
                if ep_i % args.save_model_freq == 0:
                    params = dict()
                    params['jcbfer'] = utils_cbf.get_cbf_param_values(args)
                    if policy1 is not None:
                        params['policy1'] = policy1.get_policy_param_values(args)
                    if policy2 is not None:
                        params['policy2'] = policy2.get_policy_param_values(args)
                    h_utils.save_params(log_dir, params, ep_i + 1, max_to_keep=10000)


if __name__ == "__main__":
    main()

