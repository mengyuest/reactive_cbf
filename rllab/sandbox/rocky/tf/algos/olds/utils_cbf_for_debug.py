import tensorflow as tf
import numpy as np
import hgail

import sandbox.rocky.tf.algos.utils_ngsim as utils_ngsim
import sandbox.rocky.tf.algos.utils_mono as utils_mono
import sandbox.rocky.tf.algos.utils_ped as utils_ped

def network_cbf(x, args):
    if args.use_point_net:
        if args.use_ped:
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs * args.num_neighbors, 4])  # TODO 4 is hardcoded for now
        elif args.use_mono==False:
            return network_cbf_ngsim_pointnet(x, args)
        else:
            raise NotImplementedError

    # weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
    # weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
    weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
    # weights_initializer = tf.contrib.layers.xavier_initializer()

    for i, hidden_num in enumerate(args.jcbf_hidden_layer_dims):

        x = tf.contrib.layers.fully_connected(
            x,
            int(args.jcbf_hidden_layer_dims[i]),
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense%d' % i,
            activation_fn=tf.nn.relu)
        # x = tf.Print(x, [x[:bs*8//500]], "x1", summarize=-1)

    if args.fc_then_max:
        x = tf.contrib.layers.fully_connected(
                x,
                1,
                weights_initializer=weights_initializer,
                biases_initializer=tf.zeros_initializer(),
                reuse=tf.AUTO_REUSE,
                scope='cbf/dense%d' % (len(args.jcbf_hidden_layer_dims)),
                activation_fn=None)
        x = tf.squeeze(x)
        # x = tf.Print(x, [x[:bs*8 // 500]], "x2", summarize=-1)

        if args.use_point_net:
            x = tf.reshape(x, [bs, -1])
            x = tf.reduce_max(x, reduction_indices=[1])
        # x = tf.Print(x, [x[:bs // 500]], "x3", summarize=-1)
    else:

        # x = tf.Print(x, [x[:bs*8 // 500]], "x2", summarize=-1)

        if args.use_point_net:
            x = tf.reshape(x, [bs, args.num_neighbors, args.jcbf_hidden_layer_dims[-1]])
            x = tf.reduce_max(x, reduction_indices=[1])

        x = tf.contrib.layers.fully_connected(
            x,
            1,
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense%d' % (len(args.jcbf_hidden_layer_dims)),
            activation_fn=None)
        x = tf.squeeze(x)

        # x = tf.Print(x, [x[:bs // 500]], "x3", summarize=-1)
    return x

def network_cbf_ngsim_pointnet(x, args):
    ego_x = x[:, :11]
    nei_x = x[:, 11:]

    bs = tf.shape(x)[0]
    nei_x = tf.reshape(nei_x, [bs * 6, 9])

    dbg_dict={}
    ind=0
    dbg_dict["dbgpt_%d_x"%ind] = x
    ind+=1
    dbg_dict["dbgpt_%d_ego_x"%ind] = ego_x
    ind+=1
    dbg_dict["dbgpt_%d_nei_x"%ind] = nei_x
    ind+=1

    # weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
    # weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
    weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
    # weights_initializer = tf.contrib.layers.xavier_initializer()

    for i, hidden_num in enumerate(args.jcbf_hidden_layer_dims):

        nei_x = tf.contrib.layers.fully_connected(
            nei_x,
            int(args.jcbf_hidden_layer_dims[i]),
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense%d' % i,
            activation_fn=tf.nn.relu)

        dbg_dict["dbgpt_%d_nei_x_%d" % (ind, i)] = nei_x
        ind += 1

        ego_x = tf.contrib.layers.fully_connected(
            ego_x,
            int(args.jcbf_hidden_layer_dims[i]),
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense_ego%d' % i,
            activation_fn=tf.nn.relu)

        dbg_dict["dbgpt_%d_ego_x_%d" % (ind, i)] = ego_x
        ind += 1

        # x = tf.Print(x, [x[:bs*8//500]], "x1", summarize=-1)

    if args.fc_then_max:
        raise NotImplementedError
    else:
        # x = tf.Print(x, [x[:bs*8 // 500]], "x2", summarize=-1)

        nei_x = tf.reshape(nei_x, [bs, 6, args.jcbf_hidden_layer_dims[-1]])
        nei_x = tf.reduce_max(nei_x, reduction_indices=[1])
        dbg_dict["dbgpt_%d_nei_x_max" % (ind)] = nei_x
        ind += 1

        # # missing from pytorch
        # ego_x = tf.contrib.layers.fully_connected(
        #     ego_x,
        #     int(args.jcbf_hidden_layer_dims[i]),
        #     weights_initializer=weights_initializer,
        #     biases_initializer=tf.zeros_initializer(),
        #     reuse=tf.AUTO_REUSE,
        #     scope='cbf/dense_ego%d' % (len(args.jcbf_hidden_layer_dims)),
        #     activation_fn=tf.nn.relu)

        x = nei_x + ego_x
        dbg_dict["dbgpt_%d_x_merge" % (ind)] = x
        ind += 1

        x = tf.contrib.layers.fully_connected(
            x,
            1,
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense%d' % (len(args.jcbf_hidden_layer_dims)),
            activation_fn=None)
        x = tf.squeeze(x)
        dbg_dict["dbgpt_%d_x_final" % (ind)] = x

    return x, dbg_dict


def get_var_list():
    return [v for v in tf.trainable_variables() if "cbf" in v.name]

def get_var_name_list():
    return [v.name for v in tf.trainable_variables() if "cbf" in v.name]

def set_cbf_param_values(values, args):
    var_list = get_var_list()
    # print([x.name for x in var_list])
    # print([v.shape for v in values])
    assign = tf.group(*[tf.assign(var, val) for (var, val) in zip(var_list, values)])
    session = tf.get_default_session()
    session.run(assign)

def get_cbf_param_values(args):
    var_list = get_var_list()
    session = tf.get_default_session()
    return [session.run(v) for v in var_list]

def dbg_print_cbf_params_values(args):
    var_list = get_var_list()
    var_names = get_var_name_list()
    session = tf.get_default_session()
    var_values = [session.run(v) for v in var_list]
    for vi, varn in enumerate(var_names):
        print(varn, var_values[vi].flatten())#[:3])


def dbg(x):
    return tf.strings.format('{}', x, summarize=-1)

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

def post_process_affordance(aff):
    _n, _t, _k = aff.shape
    # curve too large, clip to value
    aff[:, :, 0] = np.clip(aff[:, :, 0], -1, 1)

    # lld,rld,lrd,rrd too large, clip
    aff[:, :, 1:3] = np.clip(aff[:, :, 1:3], -5, 5)
    aff[:, :, 3:5] = np.clip(aff[:, :, 3:5], -30, 30)

    # as long as neighbor invalid, zero them
    aff_nei_flat = aff[:, :, 11:].reshape((_n * _t * 6, 9))
    aff_nei_flat = (aff_nei_flat[:, 0:1] > 0.5).astype(np.float32) * aff_nei_flat

    aff[:, :, 11:] = aff_nei_flat.reshape((_n, _t, 6 * 9))

    return aff


def normalize_affordance(state):
    info_norm = [1.0, 5.0, 5.0, 30.0, 30.0]
    ego_norm = [1.0, 30.0, 4.0, 0.15, 10.0, 3.0]
    nei_norm = [1.0, 15.0, 30.0, 1.0, 30.0, 4.0, 0.15, 10.0, 3.0] * 6
    normalizer = tf.constant([
        info_norm + ego_norm + nei_norm
    ])
    return state / normalizer

def get_indices(args):
    cbf_intervals = get_ext_indices(args.cbf_intervals)
    if any([args.lane_control, args.multilane_control,
            args.naive_control]) and args.cbf_ctrl_intervals != "":
        print("Using Primal Control and PolicyNet as Rectification!")
        cbf_ctrl_intervals = get_ext_indices(args.cbf_ctrl_intervals)
    else:
        cbf_ctrl_intervals = ""
    return cbf_intervals, cbf_ctrl_intervals

def setup_cbf(args, network, obs, _action, safe_mask, dang_mask, medium_mask, dist_info_vars=None,
              ref_action_var=None, ref_mean_var=None, ref_log_std_var=None, doing_refine=False):
    tf_dict = {}
    joint_dict = {}

    cbf_intervals, cbf_ctrl_intervals = get_indices(args)

    n_dim = tf.cast(tf.shape(obs)[0], dtype=tf.float32)
    t_dim = tf.cast(tf.shape(obs)[1], dtype=tf.float32)
    obs = tf.reshape(obs, [-1, obs.shape[2]])

    if args.ref_policy and args.refine_policy == False:
        if args.deterministic_policy_for_cbf:
            if args.residual_u:
                action = dist_info_vars["mean"] + ref_mean_var
            else:
                action = dist_info_vars["mean"]
        else:
            rnd_sym = tf.random.normal(shape=tf.shape(dist_info_vars["mean"]))
            action = rnd_sym * dist_info_vars["log_std"] + dist_info_vars["mean"]
        action = tf.reshape(action, [-1, tf.shape(action)[2]])

        if args.use_mono:
            action = action * tf.constant([[4.0]])
        elif args.use_ped:
            action = action * tf.constant([[4.0, 4.0]])
        else:
            action = action * tf.constant([[4.0, 0.15]])
    else:
        action = _action
        action = tf.reshape(action, [-1, action.shape[2]])

    # h_func_input = tf.gather(obs, cbf_intervals, axis=1)  # needs un-normalization

    state_input = tf.gather(obs, cbf_intervals, axis=1)

    ### 2. FIND THE CONTROLLER OUTPUT
    if cbf_ctrl_intervals == "":
        primal_control = action * 0.0
    else:
        primal_control = tf.gather(obs, cbf_ctrl_intervals, axis=1)

    ### 3. DEFINE MONITORING VALUES


    ### 4. DEFINE COMPUTATION GRAPH
    if args.normalize_affordance:
        if args.use_mono or args.use_ped:
            raise NotImplementedError
        if args.use_my_cbf:
            h_scores = network_cbf(normalize_affordance(state_input), args)
        else:
            h_scores = tf.squeeze(network(normalize_affordance(state_input)))
    else:
        if args.use_my_cbf:
            h_scores, dbg_dict = network_cbf(state_input, args)
            for dbg_k in dbg_dict:
                joint_dict[dbg_k] = dbg_dict[dbg_k]
        else:
            h_scores = tf.squeeze(network(state_input))
    tf_dict["state_input"] = state_input
    joint_dict["state_input"] = state_input
    tf_dict["action"] = action
    tf_dict["h_scores"] = h_scores
    joint_dict["h_scores"] = h_scores

    num_safe = tf.reduce_sum(safe_mask)
    num_dang = tf.reduce_sum(dang_mask)
    num_medium = tf.reduce_sum(medium_mask)

    loss_safe_full = tf.math.maximum(-h_scores + args.h_safe_thres, 0) * safe_mask / (1e-5 + num_safe)
    loss_safe_full = tf.reshape(loss_safe_full, [n_dim, t_dim])

    loss_dang_full = tf.math.maximum(h_scores + args.h_dang_thres, 0) * dang_mask / (1e-5 + num_dang)
    loss_dang_full = tf.reshape(loss_dang_full, [n_dim, t_dim])

    loss_safe = tf.reduce_sum(loss_safe_full)
    loss_dang = tf.reduce_sum(loss_dang_full)

    acc_dang = tf.reduce_sum(tf.cast(tf.less_equal(h_scores, 0), tf.float32) * dang_mask) / (
            1e-12 + num_dang)
    acc_safe = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores, 0), tf.float32) * safe_mask) / (
            1e-12 + num_safe)

    acc_dang = tf.cond(tf.greater(num_dang, 0), lambda: acc_dang, lambda: -tf.constant(1.0))
    acc_safe = tf.cond(tf.greater(num_safe, 0), lambda: acc_safe, lambda: -tf.constant(1.0))

    if args.use_mono:
        state_tplus1 = utils_mono.dynamics_mono(args, state_input, action, primal_control)
    elif args.use_ped:
        state_tplus1 = utils_ped.dynamics_ped(args, state_input, action, primal_control)
    else:
        state_tplus1, debug_dict = utils_ngsim.dynamics_attr(args, state_input, action, primal_control)
        tf_dict.update(debug_dict)
    tf_dict["state_tplus1"] = state_tplus1

    if args.normalize_affordance:
        if args.use_mono or args.use_ped:
            raise NotImplementedError
        if args.use_my_cbf:
            h_scores_future = network_cbf(normalize_affordance(state_tplus1), args)
        else:
            h_scores_future = tf.squeeze(network(normalize_affordance(state_tplus1)))
    else:
        if args.use_my_cbf:
            h_scores_future, _ = network_cbf(state_tplus1, args)
        else:
            h_scores_future = tf.squeeze(network(state_tplus1))
    tf_dict["h_scores_future"] = h_scores_future
    if doing_refine:
        loss_safe_deriv_full = tf.math.maximum(-h_scores_future + 0.99 * h_scores, 0) * safe_mask / (1e-12 + num_safe)
        loss_safe_deriv_full = tf.reshape(loss_safe_deriv_full, [n_dim, t_dim])
        loss_safe_deriv = tf.reduce_sum(loss_safe_deriv_full)

        loss_dang_deriv_full = tf.math.maximum(-h_scores_future + 0.99 * h_scores, 0) * dang_mask / (1e-12 + num_dang)
        loss_dang_deriv_full = tf.reshape(loss_dang_deriv_full, [n_dim, t_dim])
        loss_dang_deriv = tf.reduce_sum(loss_dang_deriv_full)

        loss_medium_deriv_full = tf.math.maximum(-h_scores_future + 0.99 * h_scores, 0) * medium_mask / (1e-12 + num_medium)
        loss_medium_deriv_full = tf.reshape(loss_medium_deriv_full, [n_dim, t_dim])
        loss_medium_deriv = tf.reduce_sum(loss_medium_deriv_full)
    else:
        loss_safe_deriv_full = tf.math.maximum(args.grad_safe_thres - h_scores_future + 0.99 * h_scores,
                                               0) * safe_mask / (1e-12 + num_safe)
        loss_safe_deriv_full = tf.reshape(loss_safe_deriv_full, [n_dim, t_dim])
        loss_safe_deriv = tf.reduce_sum(loss_safe_deriv_full)

        loss_dang_deriv_full = tf.math.maximum(args.grad_dang_thres - h_scores_future + 0.99 * h_scores,
                                               0) * dang_mask / (1e-12 + num_dang)
        loss_dang_deriv_full = tf.reshape(loss_dang_deriv_full, [n_dim, t_dim])
        loss_dang_deriv = tf.reduce_sum(loss_dang_deriv_full)

        loss_medium_deriv_full = tf.math.maximum(args.grad_medium_thres - h_scores_future + 0.99 * h_scores,
                                                 0) * medium_mask / (1e-12 + num_medium)
        loss_medium_deriv_full = tf.reshape(loss_medium_deriv_full, [n_dim, t_dim])
        loss_medium_deriv = tf.reduce_sum(loss_medium_deriv_full)

    tf_dict["save_h_deriv_data"] = h_scores_future - 0.99 * h_scores
    tf_dict["h_deriv_acc_safe"] = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                                             tf.float32) * safe_mask) / (1e-12 + num_safe)
    joint_dict["h_deriv_acc_safe"] = tf_dict["h_deriv_acc_safe"]
    tf_dict["h_deriv_acc_dang"] = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                                             tf.float32) * dang_mask) / (1e-12 + num_dang)
    joint_dict["h_deriv_acc_dang"] = tf_dict["h_deriv_acc_dang"]
    tf_dict["h_deriv_acc_medium"] = tf.reduce_sum(
        tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                tf.float32) * medium_mask) / (1e-12 + num_medium)
    joint_dict["h_deriv_acc_medium"] = tf_dict["h_deriv_acc_medium"]

    tf_dict["h_deriv_acc_safe"] = tf.cond(tf.greater(num_safe, 0), lambda: tf_dict["h_deriv_acc_safe"],
                                               lambda: -tf.constant(1.0))
    joint_dict["h_deriv_acc_safe"] = tf_dict["h_deriv_acc_safe"]
    tf_dict["h_deriv_acc_dang"] = tf.cond(tf.greater(num_dang, 0), lambda: tf_dict["h_deriv_acc_dang"],
                                               lambda: -tf.constant(1.0))
    joint_dict["h_deriv_acc_dang"] = tf_dict["h_deriv_acc_dang"]
    tf_dict["h_deriv_acc_medium"] = tf.cond(tf.greater(num_medium, 0),
                                                 lambda: tf_dict["h_deriv_acc_medium"],
                                                 lambda: -tf.constant(1.0))
    joint_dict["h_deriv_acc_medium"] = tf_dict["h_deriv_acc_medium"]
    if args.use_policy_reference and args.refine_policy == False:  # TODO since they maintain the shape (N/T, T, 2)
        if args.residual_u:
            loss_reg_policy_full = tf.reduce_sum(
                tf.math.square(dist_info_vars["mean"]) / (n_dim * t_dim), axis=[2]
            )
            # if not args.use_ped:
            #     exit("args.residual_u only supports use_ped")
            # loss_reg_policy_full = tf.reduce_sum(
            #     tf.math.square(action / tf.constant([[4.0,4.0]]) - ref_mean_var) / (n_dim * t_dim), axis=[2]
            # )
        else:
            loss_reg_policy_full = tf.reduce_sum(
                tf.math.square(dist_info_vars["mean"] - ref_mean_var) / (n_dim * t_dim), axis=[2]
            ) + tf.reduce_sum(
                tf.math.square(dist_info_vars["log_std"] - ref_log_std_var) / (n_dim * t_dim), axis=[2]
            )
    elif args.use_nominal_controller and args.refine_policy == False:
        # TODO(yue) this currently only supports PedSim
        loss_reg_policy_full = tf.sqrt(tf.reduce_sum(
            tf.math.square(tf.reshape(dist_info_vars["mean"], [n_dim, t_dim, 2]) - ref_action_var/4.0), axis=[2])) / (
                                           n_dim * t_dim)
    else:
        if args.reg_for_all_control:
            loss_reg_policy_full = tf.reduce_sum(tf.math.square(action + primal_control) / (n_dim * t_dim),
                                                 axis=[1])
        else:
            loss_reg_policy_full = tf.reduce_sum(tf.math.square(action) / (n_dim * t_dim), axis=[1])

        loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim, t_dim])
    loss_reg_policy = tf.reduce_sum(loss_reg_policy_full)

    tf_dict["deriv_total"] = loss_safe_deriv * args.safe_deriv_loss_weight \
                                  + loss_dang_deriv * args.dang_deriv_loss_weight \
                                  + loss_medium_deriv * args.medium_deriv_loss_weight

    # TODO this is only for cbf loss!
    joint_dict["debug_loss_full"] = total_loss_full = loss_safe_full * args.safe_loss_weight \
                      + loss_dang_full * args.dang_loss_weight \
                      + loss_safe_deriv_full * args.safe_deriv_loss_weight \
                      + loss_dang_deriv_full * args.dang_deriv_loss_weight \
                      + loss_medium_deriv_full * args.medium_deriv_loss_weight

    total_loss_full = loss_safe_full * args.safe_loss_weight \
                      + loss_dang_full * args.dang_loss_weight \
                      + loss_safe_deriv_full * args.safe_deriv_loss_weight \
                      + loss_dang_deriv_full * args.dang_deriv_loss_weight \
                      + loss_medium_deriv_full * args.medium_deriv_loss_weight \
                      + loss_reg_policy_full * args.reg_policy_loss_weight

    total_loss = loss_safe * args.safe_loss_weight \
                 + loss_dang * args.dang_loss_weight \
                 + loss_safe_deriv * args.safe_deriv_loss_weight \
                 + loss_dang_deriv * args.dang_deriv_loss_weight \
                 + loss_medium_deriv * args.medium_deriv_loss_weight \
                 + loss_reg_policy * args.reg_policy_loss_weight

    tf_dict["total_loss"] = total_loss
    joint_dict["total_loss"] = tf_dict["total_loss"]
    tf_dict["loss_safe"] = loss_safe * args.safe_loss_weight
    joint_dict["loss_safe"] = tf_dict["loss_safe"]
    tf_dict["loss_dang"] = loss_dang * args.dang_loss_weight
    joint_dict["loss_dang"] = tf_dict["loss_dang"]
    tf_dict["loss_crit"] = loss_safe * args.safe_loss_weight + loss_dang * args.dang_loss_weight
    joint_dict["loss_crit"] = tf_dict["loss_crit"]
    tf_dict["loss_grad"] = loss_safe_deriv * args.safe_deriv_loss_weight \
                                + loss_dang_deriv * args.dang_deriv_loss_weight \
                                + loss_medium_deriv * args.medium_deriv_loss_weight
    joint_dict["loss_grad"] = tf_dict["loss_grad"]
    # tf_dict["loss_grad"] = tf.Print(tf_dict["loss_grad"], gradients, "d_u_res", summarize=-1)

    tf_dict["loss_safe_deriv"] = loss_safe_deriv * args.safe_deriv_loss_weight
    joint_dict["loss_safe_deriv"] = tf_dict["loss_safe_deriv"]
    tf_dict["loss_dang_deriv"] = loss_dang_deriv * args.dang_deriv_loss_weight
    joint_dict["loss_dang_deriv"] = tf_dict["loss_dang_deriv"]
    tf_dict["loss_medium_deriv"] = loss_medium_deriv * args.medium_deriv_loss_weight
    joint_dict["loss_medium_deriv"] = tf_dict["loss_medium_deriv"]
    tf_dict["loss_reg_policy"] = loss_reg_policy * args.reg_policy_loss_weight
    joint_dict["loss_reg_policy"] = tf_dict["loss_reg_policy"]
    tf_dict["num_dang"] = num_dang
    joint_dict["num_dang"] = tf_dict["num_dang"]
    tf_dict["num_safe"] = num_safe
    joint_dict["num_safe"] = tf_dict["num_safe"]
    tf_dict["num_medium"] = num_medium
    joint_dict["num_medium"] = tf_dict["num_medium"]
    tf_dict["acc_dang"] = acc_dang
    joint_dict["acc_dang"] = tf_dict["acc_dang"]
    tf_dict["acc_safe"] = acc_safe
    joint_dict["acc_safe"] = tf_dict["acc_safe"]

    if args.use_my_cbf:
        net_var_list = get_var_list()
    else:
        net_var_list = network.var_list
    return cbf_intervals, cbf_ctrl_intervals, total_loss, total_loss_full, net_var_list, tf_dict, joint_dict



#TODO(yue)
def init_opt_refine(args, is_recurrent, env, network, grad_norm_rescale, grad_norm_clip):
    is_recurrent = int(is_recurrent)
    true_obs_var = env.observation_space.new_tensor_variable('true_obs', extra_dims=1 + is_recurrent)
    true_action_var = env.action_space.new_tensor_variable('true_action', extra_dims=1 + is_recurrent)
    safe_mask = tf.placeholder(tf.float32, [None], name="safe_mask")  # safe labels
    dang_mask = tf.placeholder(tf.float32, [None], name="dang_mask")  # dangerous labels
    medium_mask = tf.placeholder(tf.float32, [None], name="medium_mask")  # medium labels

    refine_input_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]

    if args.use_mono:
        u_res = tf.Variable(tf.zeros([args.n_envs, 1, 1]), name='u_res')
    elif args.use_ped:
        u_res = tf.Variable(tf.zeros([args.n_envs, 1, 2]), name='u_res')
    else:
        u_res = tf.Variable(tf.zeros([args.n_envs, 1, 2]), name='u_res')
    u_init = tf.assign(u_res, tf.zeros_like(u_res))
    # u_init = None

    cbf_intervals, cbf_ctrl_intervals, cbf_loss, cbf_loss_full, _, refine_dict, _ = \
        setup_cbf(args, network, true_obs_var, true_action_var + u_res, safe_mask, dang_mask, medium_mask, doing_refine=True)

    gradients = tf.gradients(refine_dict["loss_grad"], u_res)

    # TODO(debug) monitoring some variables
    refine_dict["gradients0"] = gradients[0]
    refine_dict["safe_mask"] = safe_mask
    refine_dict["dang_mask"] = dang_mask
    refine_dict["medium_mask"] = medium_mask
    refine_dict["true_obs_var"] = true_obs_var
    # refine_dict["h_scores"]
    # refine_dict["h_scores_future"]
    refine_dict["u_res_prev"] = u_res * 1.0

    if args.use_mono:
        dynamics_ops = utils_mono.dynamics_mono
        old_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res, [-1, 1]),
                                        tf.reshape(true_action_var * 0.0, [-1, 1]))
        old_h_scores_future = network_cbf(old_state_tplus1, args)
    elif args.use_ped:
        dynamics_ops = utils_ped.dynamics_ped
        old_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))
        old_h_scores_future = network_cbf(old_state_tplus1, args)
    else:
        dynamics_ops = utils_ngsim.dynamics_attr
        raise NotImplementedError



    # clipped_gradients = hgail.misc.tf_utils.clip_gradients(gradients, grad_norm_rescale, grad_norm_clip)
    # refine_optimizer = tf.train.AdamOptimizer(args.refine_learning_rate)
    # # refine_optimizer = tf.train.RMSPropOptimizer(args.refine_learning_rate)
    # refine_global_step = tf.Variable(0, name='cbf/refine_global_step', trainable=False)
    # refine_op = refine_optimizer.apply_gradients([(clipped_gradients[0], u_res)], global_step=refine_global_step)
    u_res_new = tf.assign(u_res, u_res - gradients[0] * args.refine_learning_rate)

    # compute future states and future scores again

    if args.use_mono:
        mod_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                              tf.reshape(true_action_var + u_res_new, [-1, 1]), tf.reshape(true_action_var * 0.0, [-1, 1]))
    elif args.use_ped:
        mod_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res_new, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))
    else:
        dynamics_ops = utils_ngsim.dynamics_attr
        raise NotImplementedError
    mod_h_scores_future = network_cbf(mod_state_tplus1, args)
    refine_dict["old_state_tplus1"] = old_state_tplus1
    refine_dict["old_h_scores_future"] = old_h_scores_future
    refine_dict["mod_state_tplus1"] = mod_state_tplus1
    refine_dict["mod_h_scores_future"] = mod_h_scores_future

    refine_op = None


    return cbf_intervals, cbf_ctrl_intervals, refine_input_list, u_res_new, u_init, refine_op, refine_dict



#TODO(yue)
def init_opt_refine_qp(args, is_recurrent, env, network):
    is_recurrent = int(is_recurrent)
    true_obs_var = env.observation_space.new_tensor_variable('true_obs', extra_dims=1 + is_recurrent)
    # true_action_var = env.action_space.new_tensor_variable('true_action', extra_dims=1 + is_recurrent)
    # safe_mask = tf.placeholder(tf.float32, [None], name="safe_mask")  # safe labels
    # dang_mask = tf.placeholder(tf.float32, [None], name="dang_mask")  # dangerous labels
    # medium_mask = tf.placeholder(tf.float32, [None], name="medium_mask")  # medium labels

    # refine_input_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]
    refine_input_list = [true_obs_var]
    cbf_intervals, cbf_ctrl_intervals = get_indices(args)

    obs = tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]])

    state_input = tf.gather(obs, cbf_intervals, axis=1)

    if args.normalize_affordance:
        if args.use_mono or args.use_ped:
            raise NotImplementedError
        if args.use_my_cbf:
            h_scores = network_cbf(normalize_affordance(state_input), args)
        else:
            h_scores = tf.squeeze(network(normalize_affordance(state_input)))
    else:
        if args.use_my_cbf:
            h_scores = network_cbf(state_input, args)
        else:
            h_scores = tf.squeeze(network(state_input))


    dhdx = tf.gradients(h_scores, state_input)

    # dhdx = [tf.gradients(h_scores[i], state_input[i]) for i in range(tf.shape(h_scores))]

    refine_dict={"dhdx": dhdx, "hx": h_scores}


    return cbf_intervals, cbf_ctrl_intervals, refine_input_list, refine_dict