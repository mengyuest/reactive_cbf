import tensorflow as tf
import numpy as np
import hgail

import sandbox.rocky.tf.algos.utils_ngsim as utils_ngsim
import sandbox.rocky.tf.algos.utils_mono as utils_mono
import sandbox.rocky.tf.algos.utils_ped as utils_ped
import sandbox.rocky.tf.algos.utils_easy as utils_easy

import sandbox.rocky.tf.algos.utils_debug as utils_debug

def network_cbf(x, args):
    if args.agent_cbf:  # TODO only for NGSIM for now
        return network_cbf_agent(x, args)

    if args.use_point_net:
        if args.use_ped:
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs * args.num_neighbors, 4])  # TODO 4 is hardcoded for now
        elif args.use_mono==False:
            return network_cbf_ngsim_pointnet(x, args)
        else:
            raise NotImplementedError

    weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)

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
        x = tf.squeeze(x, axis=-1)  # TODO(modified)
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
        x = tf.squeeze(x, axis=-1)  # TODO(modified)

    return x


def network_cbf_ngsim_pointnet(x, args):
    ego_x = x[:, :11]
    nei_x = x[:, 11:]

    bs = tf.shape(x)[0]
    nei_x = tf.reshape(nei_x, [bs * 6, 9])

    weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)

    for i, hidden_num in enumerate(args.jcbf_hidden_layer_dims):

        nei_x = tf.contrib.layers.fully_connected(
            nei_x,
            int(args.jcbf_hidden_layer_dims[i]),
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense%d' % i,
            activation_fn=tf.nn.relu)

        ego_x = tf.contrib.layers.fully_connected(
            ego_x,
            int(args.jcbf_hidden_layer_dims[i]),
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense_ego%d' % i,
            activation_fn=tf.nn.relu)

        # x = tf.Print(x, [x[:bs*8//500]], "x1", summarize=-1)

    if args.fc_then_max:
        raise NotImplementedError
    else:
        nei_x = tf.reshape(nei_x, [bs, 6, args.jcbf_hidden_layer_dims[-1]])
        nei_x = tf.reduce_max(nei_x, reduction_indices=[1])

        x = nei_x + ego_x

        x = tf.contrib.layers.fully_connected(
            x,
            1,
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='cbf/dense%d' % (len(args.jcbf_hidden_layer_dims)),
            activation_fn=None)
        x = tf.squeeze(x, axis=-1)  # TODO(modified)

    return x

def network_cbf_agent(x, args):
    ego_x = x[:, :11]
    nei_x = x[:, 11:]

    bs = tf.shape(x)[0]
    nei_x = tf.reshape(nei_x, [bs, 6, 9])

    if args.new_cbf_pol:
        nei_x = tf.stack(
            [nei_x[:, :, 2] - 0.5 * (nei_x[:, :, -2] + tf.expand_dims(ego_x[:, -2], axis=1)), nei_x[:, :, 4]], axis=-1)
        nei_x = tf.stack([nei_x[:, 0], nei_x[:, 3]], axis=1)
        cat_x = []
        for i in range(2):
            cat_x.append(tf.concat([ego_x[:, 3:4], ego_x[:, 4:5], ego_x[:, 6:7], nei_x[:, i, :]], axis=-1))
        cat_x = tf.stack(cat_x, axis=1)  # TODO (bs, 2, 5)  lrd, rrd, v_ego, d, v_nei
    elif args.debug_simple_cbf:
        nei_x = tf.norm(nei_x[:, :, 1:3], axis=-1)
        nei_x = tf.reshape(nei_x, [bs, 6, 1])
        nei_x = tf.stack([nei_x[:, 0], nei_x[:, 1]*0+30, nei_x[:, 2]*0+30, nei_x[:, 3], nei_x[:, 4]*0+30, nei_x[:, 5]*0+30], axis=1)
        cat_x = []
        for i in range(6):
            cat_x.append(tf.concat([ego_x[:, 3:4], ego_x[:, 4:5], nei_x[:, i, :]], axis=-1))
        cat_x = tf.stack(cat_x, axis=1)  # TODO (bs, 6, 3)
    else:
        # TODO remove the ind part
        nei_x = nei_x[:, :, 1:]

        cat_x = []
        for i in range(6):
            cat_x.append(tf.concat([ego_x, nei_x[:, i, :]], axis=-1))
        cat_x = tf.stack(cat_x, axis=1)  # TODO (bs, 6, 11+8)

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

    return cat_x  # TODO (bs, 6)

def get_var_list():
    return [v for v in tf.trainable_variables() if "cbf" in v.name]

def get_var_name_list():
    return [v.name for v in tf.trainable_variables() if "cbf" in v.name]

def set_cbf_param_values(values, args):
    var_list = get_var_list()
    assign = tf.group(*[tf.assign(var, val) for (var, val) in zip(var_list, values)])
    session = tf.get_default_session()
    session.run(assign)

def get_cbf_param_values(args):
    var_list = get_var_list()
    session = tf.get_default_session()
    return [session.run(v) for v in var_list]

class MyPolicy:
    def __init__(self, env, args):
        self.args = args
        self.recurrent = True  # used in many places - just set to True
        self.vectorized = True  # used in batch_polopt.py
        self.name = "myp"
        self.gt_obs = env.observation_space.new_tensor_variable('myp/obs', extra_dims=1 + int(self.recurrent))
        # self.u_ref = tf.placeholder(tf.float32, [None, 2], name="myp/uref")
        self.action = self.get_action_tf(self.gt_obs)

    def get_action_tf(self, gt_obs):
        n_dim = tf.cast(tf.shape(gt_obs)[0], dtype=tf.float32)
        t_dim = tf.cast(tf.shape(gt_obs)[1], dtype=tf.float32)
        gt_obs = tf.reshape(gt_obs, [-1, gt_obs.shape[2]])
        cbf_intervals, _ = get_indices(self.args)
        state_input = tf.gather(gt_obs, cbf_intervals, axis=1)
        act = network_policy(state_input, self.args)
        return tf.reshape(act, [n_dim, t_dim, -1])

    def get_action_tf_flat(self, state_input):  # TODO overwrite
        self.gt_obs = state_input
        self.action = network_policy(state_input, self.args)
        return self.action

    def get_actions_flat(self, gt_obs):
        session = tf.get_default_session()
        if self.args.second_fusion:
            action, = session.run([self.action], feed_dict={
                self.gt_obs: gt_obs,
                # self.u_ref: u_ref,
            })
        else:
            action, = session.run([self.action], feed_dict={
                self.gt_obs: gt_obs,
            })
        agent_infos = {"mean": action}
        return action, agent_infos

    def get_actions(self, gt_obs):
        session = tf.get_default_session()
        if self.args.second_fusion:
            action, = session.run([self.action], feed_dict={
                self.gt_obs: np.expand_dims(gt_obs, axis=1),
                # self.u_ref: u_ref,
            })
        else:
            action, = session.run([self.action], feed_dict={
                self.gt_obs: np.expand_dims(gt_obs, axis=1),
            })
        action = np.squeeze(action, axis=1)  # TODO(modified))
        agent_infos = {"mean": action}
        return action, agent_infos

    def get_params(self, trainable):
        return get_policy_var_list()

    def log_diagnostics(self, path):  # TODO (yue): used by batch_polopt.py for saving iters; and GAIL.save_itr_params
        return

    def set_param_values(self, params):
        set_policy_param_values(params, self.args)

    def reset(self, dones):
        return

    @property
    def state_info_specs(self):
        return []

    @property
    def state_info_keys(self):
        return []

    def dist_info_sym(self, gt_obs):
        return {"mean": self.get_action_tf(gt_obs)}

def network_policy(x, args):
    if args.policy_use_point_net:
        if args.use_ped:
            act = network_policy_ped_pointnet(x, args)
        elif args.use_mono:
            raise NotImplementedError
        else:
            if args.new_cbf_pol:
                act = network_policy_ngsim_pointnet_easy(x, args)
            else:
                act = network_policy_ngsim_pointnet(x, args)

    if args.debug_accel_only:
        act = tf.stack([act[:, 0], 0 * act[:, 1]], axis=-1)
    if args.zero_policy:
        act = act * 0.0
    return act

def network_policy_ped_pointnet(x, args):
    bs = tf.shape(x)[0]
    nei_x = tf.reshape(x, [bs * args.num_neighbors, 4])

    weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)

    for i, hidden_num in enumerate(args.policy_hidden_layer_dims):
        nei_x = tf.contrib.layers.fully_connected(
            nei_x,
            int(args.policy_hidden_layer_dims[i]),
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='myp/dense%d' % i,
            activation_fn=tf.nn.relu)

    nei_x = tf.reshape(nei_x, [bs, args.num_neighbors, args.policy_hidden_layer_dims[-1]])
    nei_x = tf.reduce_max(nei_x, reduction_indices=[1])

    act = tf.contrib.layers.fully_connected(
        nei_x,
        2,
        weights_initializer=weights_initializer,
        biases_initializer=tf.zeros_initializer(),
        reuse=tf.AUTO_REUSE,
        scope='myp/dense%d' % (len(args.policy_hidden_layer_dims)),
        activation_fn=None)
    act = tf.squeeze(act)
    return act


def network_policy_ngsim_pointnet(x, args):
    ego_x = x[:, :11]
    nei_x = x[:, 11:]

    bs = tf.shape(x)[0]
    nei_x = tf.reshape(nei_x, [bs * 6, 9])

    ego_x = tf.reshape(ego_x, [bs, 1, 11])  # make sure conv1d can handle
    nei_x = tf.reshape(nei_x[:, 1:], [bs, 6, 8])  # remove ind
    ego_x_cp = tf.tile(ego_x, [1, 6, 1])
    nei_feat = tf.concat([ego_x_cp, nei_x], axis=-1)
    for i, hidden_num in enumerate(args.policy_hidden_layer_dims):
        nei_feat = tf.contrib.layers.conv1d(inputs=nei_feat,
                                         num_outputs=hidden_num,
                                         kernel_size=1,
                                         reuse=tf.AUTO_REUSE,
                                         scope='myp/conv%d' % i,
                                         activation_fn=tf.nn.relu)

    nei_feat = tf.reduce_max(nei_feat, reduction_indices=[1])

    # x.shape, u_ref?
    # TODO any heuristic we can follow to shapify the trajectory reference?
    x = tf.concat([ego_x[:, 0], nei_feat, tf.cos(ego_x[:, 0, 5:6]), tf.sin(ego_x[:, 0, 5:6])], axis=-1)
    for i, hidden_num in enumerate(args.policy_hidden_fusion_dims):
        x = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=hidden_num,
            reuse=tf.AUTO_REUSE,
            scope='myp/convII%d' % i,
            activation_fn=tf.nn.relu)
    act = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=2,
        reuse=tf.AUTO_REUSE,
        scope='myp/dense%d' % (len(args.policy_hidden_layer_dims)+len(args.policy_hidden_fusion_dims)),
        activation_fn=None)

    return act


def network_policy_ngsim_pointnet_easy(x, args):
    ego_x = x[:, :11]
    nei_x = x[:, 11:]

    bs = tf.shape(x)[0]
    nei_x = tf.reshape(nei_x, [bs, 6, 9])

    nei_feat = tf.stack([ego_x[:, 6],
                         nei_x[:, 0, 2] - 0.5 * (nei_x[:, 0, -2] + ego_x[:, -2]), nei_x[:, 0, 4],
                         # (ego_x[:, 6]*ego_x[:, 6]-nei_x[:, 0, 4]*nei_x[:, 0, 4])/2/4,
                         nei_x[:, 3, 2] - 0.5 * (nei_x[:, 3, -2] + ego_x[:, -2]), nei_x[:, 3, 4],
                         # (ego_x[:, 6]*ego_x[:, 6]-nei_x[:, 3, 4]*nei_x[:, 3, 4])/2/4,
                         ], axis=-1)

    nei_feat = tf.reshape(nei_feat, [bs, 1, 5])

    for i, hidden_num in enumerate(args.policy_hidden_layer_dims):
        nei_feat = tf.contrib.layers.conv1d(inputs=nei_feat,
                                            num_outputs=hidden_num,
                                            kernel_size=1,
                                            reuse=tf.AUTO_REUSE,
                                            scope='myp/conv%d' % i,
                                            activation_fn=tf.nn.relu)

    nei_feat = tf.reduce_max(nei_feat, reduction_indices=[1])

    # x.shape, u_ref?
    # TODO any heuristic we can follow to shapify the trajectory reference?
    # x = tf.concat([ego_x[:, 0], nei_feat, tf.cos(ego_x[:, 0, 5:6]), tf.sin(ego_x[:, 0, 5:6])], axis=-1)
    x = nei_feat
    for i, hidden_num in enumerate(args.policy_hidden_fusion_dims):
        x = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=hidden_num,
            reuse=tf.AUTO_REUSE,
            scope='myp/convII%d' % i,
            activation_fn=tf.nn.relu)
    act = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=2,
        reuse=tf.AUTO_REUSE,
        scope='myp/dense%d' % (len(args.policy_hidden_layer_dims) + len(args.policy_hidden_fusion_dims)),
        activation_fn=None)

    return act


class MyPolicyHighLevel:
    def __init__(self, env, args):
        self.args = args
        self.recurrent = True
        self.vectorized = True
        self.name = "hlp"

    def get_choice(self, gt_obs):
        session = tf.get_default_session()
        tau, logits, choice = session.run([self.tau, self.logits, self.choice],
                                             feed_dict={self.gt_obs: np.expand_dims(gt_obs, axis=1)})
        logits = np.squeeze(logits, axis=1)  # TODO(modified)
        choice = np.squeeze(choice, axis=1)  # TODO(modified)
        agent_infos = {"tau": tau, "logits": logits, "choice": choice}  # TODO here we have the difference
        return choice, agent_infos

    def get_choice_flat(self, gt_obs):
        session = tf.get_default_session()
        tau, logits, choice = session.run([self.tau, self.logits, self.choice],
                                             feed_dict={self.gt_obs: gt_obs})
        agent_infos = {"tau": tau, "logits": logits, "choice": choice}  # TODO here we have the difference
        return choice, agent_infos

    def build_graph(self, gt_obs_tf):
        n_dim = tf.cast(tf.shape(gt_obs_tf)[0], dtype=tf.float32)
        t_dim = tf.cast(tf.shape(gt_obs_tf)[1], dtype=tf.float32)
        gt_obs_tf = tf.reshape(gt_obs_tf, [-1, gt_obs_tf.shape[2]])
        cbf_intervals, _ = get_indices(self.args)
        state_input = tf.gather(gt_obs_tf, cbf_intervals, axis=1)

        tau = tf.Variable(self.args.temperature, name="temperature")
        logits = network_high_level_policy(state_input, self.args)  # this will be NT*2, two classes

        if self.args.always_keeping:
            choice = tf.stack([tf.ones_like(logits[:, 0]), tf.zeros_like(logits[:, 1])], axis=-1)
        else:
            choice = gumbel_softmax(logits, tau, hard=True)  # one-hot vector for 0,1

        self.tau = tau
        self.logits = tf.reshape(logits, [n_dim, t_dim, 2])
        self.choice = tf.reshape(choice, [n_dim, t_dim, 2])
        return self.tau, self.logits, self.choice

    def build_graph_flat(self, gt_obs_tf):
        self.gt_obs = gt_obs_tf
        self.tau = tf.Variable(self.args.temperature, name="temperature")
        self.logits = network_high_level_policy(gt_obs_tf, self.args)  # this will be NT*2, two classes

        if self.args.always_keeping:
            self.choice = tf.stack([tf.ones_like(self.logits[:, 0]),
                                    tf.zeros_like(self.logits[:, 1])], axis=-1)
        else:
            self.choice = gumbel_softmax(self.logits, self.tau, hard=True)  # one-hot vector for 0,1

        return self.tau, self.logits, self.choice

    def get_params(self, trainable):
        return get_high_level_policy_var_list()

    def log_diagnostics(self, path):  # TODO (yue): used by batch_polopt.py for saving iters; and GAIL.save_itr_params
        return

    def set_param_values(self, params):
        set_high_level_policy_param_values(params, self.args)

    def reset(self, dones):
        return

    @property
    def state_info_specs(self):
        return []

    @property
    def state_info_keys(self):
        return []

    def dist_info_sym(self, gt_obs_tf):  # TODO this is not a good try; too many class vars
        self.gt_obs = gt_obs_tf
        tau_tf, logits_tf, choice_tf = self.build_graph(gt_obs_tf)
        agent_infos_tf = {"tau": tau_tf, "logits": logits_tf, "choice": choice_tf}  # TODO here we have the difference
        return agent_infos_tf

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape, minval=0, maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax(y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def network_high_level_policy(x_pl, args):
    weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
    for i in range(len(args.high_level_policy_hiddens)):
        x_pl = tf.contrib.layers.fully_connected(
            x_pl,
            int(args.high_level_policy_hiddens[i]),
            weights_initializer=weights_initializer,
            biases_initializer=tf.zeros_initializer(),
            reuse=tf.AUTO_REUSE,
            scope='hlp/dense%d' % i,
            activation_fn=tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(
        x_pl,
        2,
        weights_initializer=weights_initializer,
        biases_initializer=tf.zeros_initializer(),
        reuse=tf.AUTO_REUSE,
        scope='hlp/dense%d' % len(args.high_level_policy_hiddens))
    return logits  # TODO(modified)) tf.squeeze(logits)


def get_policy_var_list():
    return [v for v in tf.trainable_variables() if "myp/" in v.name]


def get_policy_var_name_list():
    return [v.name for v in tf.trainable_variables() if "myp/" in v.name]


def set_policy_param_values(values, args):
    var_list = get_policy_var_list()
    assign = tf.group(*[tf.assign(var, val) for (var, val) in zip(var_list, values)])
    session = tf.get_default_session()
    session.run(assign)


def get_policy_param_values(args):
    var_list = get_policy_var_list()
    session = tf.get_default_session()
    return [session.run(v) for v in var_list]


def get_high_level_policy_var_list():
    return [v for v in tf.trainable_variables() if "hlp/" in v.name]


def get_high_level_policy_var_name_list():
    return [v.name for v in tf.trainable_variables() if "hlp/" in v.name]


def set_high_level_policy_param_values(values, args):
    var_list = get_high_level_policy_var_list()
    assign = tf.group(*[tf.assign(var, val) for (var, val) in zip(var_list, values)])
    session = tf.get_default_session()
    session.run(assign)


def get_high_level_policy_param_values(args):
    var_list = get_high_level_policy_var_list()
    session = tf.get_default_session()
    return [session.run(v) for v in var_list]


def lane_keeping_controller(gt_obs, args):
    cbf_intervals, _ = get_indices(args)
    state = gt_obs[:, cbf_intervals]

    lld = state[:, 1]
    rld = state[:, 2]
    theta = state[:, 5]

    dt=0.1
    tol_dist=0.5
    theta_max=0.5
    omega_clip=0.15

    tol = np.abs(lld - rld) < tol_dist
    theta_desired = -1 * theta_max * rld / (lld + rld) + theta_max * lld / (lld + rld)
    theta_desired = (1-tol) * theta_desired
    omega = (theta_desired - theta)/dt
    omega = np.clip(omega, omega_clip * -1, omega_clip * 1)
    accel = np.zeros_like(omega)
    control = np.stack((accel, omega), axis=-1)
    return control


def dbg(x):
    return tf.strings.format('{}', x, summarize=-1)


def get_true_actions(actions, args):
    if args.no_action_scaling == False:
        if args.use_mono:
            true_actions = actions * np.array([[4, ]])
        elif args.use_ped:
            true_actions = actions * np.array([[4, 4]])
        elif args.use_easy:
            true_actions = actions * np.array([[4, 4]])
        else:
            true_actions = actions * np.array([[4, 0.15]])
    else:
        true_actions = actions
    if len(true_actions.shape) == 2:
        true_actions = np.expand_dims(true_actions, axis=1)
    return true_actions

def get_true_observations(args, observations, obs_mean, obs_var):
    if args.no_obs_normalize:
        true_obs = observations
    else:
        true_obs = observations * np.sqrt(obs_var) + obs_mean
    if len(true_obs.shape) == 2:
        true_obs = np.expand_dims(true_obs, axis=1)
    return true_obs

def get_masks(args, true_obs, cbf_intervals, out_lane):

    cbf_obs = true_obs.reshape((-1, true_obs.shape[-1]))[:, cbf_intervals]
    safe_d1 = args.safe_dist_threshold
    safe_d2 = args.safe_dist_threshold_side
    dang_d1 = args.dang_dist_threshold
    dang_d2 = args.dang_dist_threshold_side

    if args.use_mono:
        get_mask_fn = utils_mono.get_safe_mask_mono
    elif args.use_ped:
        get_mask_fn = utils_ped.get_safe_mask_ped
    elif args.use_easy:
        get_mask_fn = utils_easy.get_safe_mask_easy
    else:
        if args.agent_cbf:
            get_mask_fn = utils_ngsim.get_safe_mask_agent
        else:
            get_mask_fn = utils_ngsim.get_safe_mask
    safe_mask = get_mask_fn(args, cbf_obs, out_lane, safe_d1, safe_d2, check_safe=True)
    dang_mask = get_mask_fn(args, cbf_obs, out_lane, dang_d1, dang_d2, check_safe=False)
    medium_mask = np.logical_and(~safe_mask, ~dang_mask)

    safe_mask = safe_mask.astype(dtype=np.float32)
    medium_mask = medium_mask.astype(dtype=np.float32)
    dang_mask = dang_mask.astype(dtype=np.float32)

    return safe_mask, medium_mask, dang_mask

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


def get_actions_tf(args, _action, dist_info_vars, ref_action_var, ref_mean_var):
    if args.ref_policy and args.refine_policy == False:
        if args.deterministic_policy_for_cbf:
            if args.residual_u:
                if args.use_nominal_controller: # only for PedSim and EasySim
                    assert args.use_ped or args.use_easy
                    action = dist_info_vars["mean"] + ref_action_var
                elif args.high_level:
                    action = dist_info_vars["mean"] + ref_action_var
                else:
                    action = dist_info_vars["mean"] + ref_mean_var
            else:
                action = dist_info_vars["mean"]
        else:
            rnd_sym = tf.random.normal(shape=tf.shape(dist_info_vars["mean"]))
            action = rnd_sym * dist_info_vars["log_std"] + dist_info_vars["mean"]
        action = tf.reshape(action, [-1, tf.shape(action)[2]])

        if args.no_action_scaling==False:
            if args.use_mono:
                action = action * tf.constant([[4.0]])
            elif args.use_ped:
                action = action * tf.constant([[4.0, 4.0]])
            elif args.use_easy:
                action = action * tf.constant([[4.0, 4.0]])
            else:
                action = action * tf.constant([[4.0, 0.15]])
    else:
        action = _action
        action = tf.reshape(action, [-1, action.shape[2]])
    return action


def get_cbf_value_tf(args, state_input, network):
    if args.normalize_affordance:
        if any([args.use_mono, args.use_ped, args.use_easy]):
            raise NotImplementedError
        state_input = normalize_affordance(state_input)

    return network_cbf(state_input, args)


def get_next_state(args, state, action, primal_control):
    # return dynamics(args,state,action,primal_control)  # TODO(debug)
    if args.use_mono:
        return utils_mono.dynamics_mono(args, state, action, primal_control)
    elif args.use_ped:
        return utils_ped.dynamics_ped(args, state, action, primal_control)
    elif args.use_easy:
        return utils_easy.dynamics_easy(args, state, action, primal_control)
    else:
        return utils_ngsim.dynamics_attr(args, state, action, primal_control)

def dynamics(args, state, control, primal_control):
    n_ego_feat = 11
    n_nei_feat = 9

    feat_x_def = tf.constant([0.0, -15.0, 15.0, 0.0, -15.0, 15.0])
    feat_y_def = tf.constant([30.0, 30.0, 30.0, -30.0, -30.0, -30.0])

    next_state = state * 1.0
    # perform precised updates (numerically)
    dT = 0.1
    discrete_num = args.cbf_discrete_num
    dt = dT / discrete_num

    for tt in range(discrete_num):
        # symbol table
        curve = next_state[:, 0]
        lld = next_state[:, 1]
        rld = next_state[:, 2]
        lrd = next_state[:, 3]
        rrd = next_state[:, 4]
        theta = next_state[:, 2 + 3]
        v = next_state[:, 3 + 3]
        length = next_state[:, 6 + 3]
        width = next_state[:, 7 + 3]
        accel = control[:, 0] + primal_control[:, 0]
        omega = control[:, 1] + primal_control[:, 1]
        if args.debug_accel_only:
            omega = control[:, 1] * 0.0 + primal_control[:, 1]

        # 1-order dynamic
        dx = v * tf.sin(theta) * dt
        ds = v * dt

        # 2-order dynamic
        dv = accel * dt
        dtheta = omega * dt

        # updates
        new_curve = curve
        new_lld = lld - dx
        new_rld = rld + dx
        new_lrd = lrd - dx
        new_rrd = rrd + dx
        new_theta = theta + dtheta
        new_v = v + dv

        new_accel = accel
        new_omega = omega
        new_length = length
        new_width = width

        # neighbors
        nei_feat = tf.reshape(next_state[:, n_ego_feat:], [-1, n_nei_feat])
        nei_ind = nei_feat[:, 0]
        nei_x = nei_feat[:, 1]
        nei_y = nei_feat[:, 2]
        nei_theta = nei_feat[:, 3]
        nei_v = nei_feat[:, 4]

        nei_accel = nei_feat[:, 5]
        nei_omega = nei_feat[:, 6]

        nei_length = nei_feat[:, 7]
        nei_width = nei_feat[:, 8]

        # [1,2,3]->[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
        ds_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ds, [6]), [6, -1])), [-1])
        dtheta_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(dtheta, [6]), [6, -1])), [-1])
        new_v_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(new_v, [6]), [6, -1])), [-1])

        # 1-order dynamic
        nei_dx = - nei_v * tf.sin(nei_theta) * dt
        nei_dy = nei_v * tf.cos(nei_theta) * dt

        # 2-order dynamic
        nei_dtheta = nei_omega * dt
        nei_dv = nei_accel * dt

        # updates
        new_nei_x = nei_x + nei_dx - 0
        new_nei_y = nei_y + nei_dy - ds_cp
        # (TODO) consider transformation caused by ego rotation
        new_nei_x, new_nei_y = \
            new_nei_x * tf.cos(dtheta_cp) + new_nei_y * tf.sin(dtheta_cp), \
            - new_nei_x * tf.sin(dtheta_cp) + new_nei_y * tf.cos(dtheta_cp)
        new_nei_theta = nei_theta + nei_dtheta - dtheta_cp
        new_nei_v = nei_v + nei_dv
        new_nei_length = nei_length
        new_nei_width = nei_width

        # indicator and bounding values
        # using current calculated value, or the default value, depending on the indicator
        # TODO clip
        new_nei_x = tf.clip_by_value(new_nei_x, -15.0, 15.0)
        new_nei_y = tf.clip_by_value(new_nei_y, -30.0, 30.0)

        new_nei_x = nei_ind * new_nei_x + (1 - nei_ind) * (tf.tile(feat_x_def, [tf.shape(nei_ind)[0] // 6]))  # TODO
        new_nei_y = nei_ind * new_nei_y + (1 - nei_ind) * (tf.tile(feat_y_def, [tf.shape(nei_ind)[0] // 6]))  # TODO
        new_nei_theta = nei_ind * new_nei_theta
        new_nei_v = nei_ind * new_nei_v + (1 - nei_ind) * new_v_cp
        new_nei_accel = nei_accel
        new_nei_omega = nei_omega
        new_nei_length = nei_ind * new_nei_length
        new_nei_width = nei_ind * new_nei_width

        # merge them to a single tensor
        new_ego = tf.stack(
            [new_curve, new_lld, new_rld, new_lrd, new_rrd, new_theta, new_v, new_accel, new_omega, new_length,
             new_width], axis=-1)
        new_nei = tf.stack(
            [nei_ind, new_nei_x, new_nei_y, new_nei_theta, new_nei_v, new_nei_accel, new_nei_omega, new_nei_length,
             new_nei_width], axis=-1)

        new_nei = tf.reshape(new_nei, [-1, 6 * n_nei_feat])
        next_state = tf.concat([new_ego, new_nei], axis=-1)

    return next_state


def get_loss_shape(args, n_dim, t_dim):
    if args.agent_cbf:
        if args.new_cbf_pol:
            loss_shape = [n_dim, t_dim, 2]
        else:
            loss_shape = [n_dim, t_dim, 6]
    else:
        loss_shape = [n_dim, t_dim]
    return loss_shape


def loss_barrier(args, h_scores, safe_mask, dang_mask, num_safe, num_dang, n_dim, t_dim):
    loss_safe_full = tf.math.maximum(-h_scores + args.h_safe_thres, 0) * safe_mask / (1e-5 + num_safe)
    loss_dang_full = tf.math.maximum(h_scores + args.h_dang_thres, 0) * dang_mask / (1e-5 + num_dang)

    loss_shape = get_loss_shape(args, n_dim, t_dim)

    loss_safe_full = tf.reshape(loss_safe_full, loss_shape)
    loss_dang_full = tf.reshape(loss_dang_full, loss_shape)

    loss_safe = tf.reduce_sum(loss_safe_full)
    loss_dang = tf.reduce_sum(loss_dang_full)

    acc_safe = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores, 0), tf.float32) * safe_mask) / (1e-12 + num_safe)
    acc_dang = tf.reduce_sum(tf.cast(tf.less_equal(h_scores, 0), tf.float32) * dang_mask) / (1e-12 + num_dang)

    acc_safe = tf.cond(tf.greater(num_safe, 0), lambda: acc_safe, lambda: -tf.constant(1.0))
    acc_dang = tf.cond(tf.greater(num_dang, 0), lambda: acc_dang, lambda: -tf.constant(1.0))
    return loss_safe_full, loss_dang_full, loss_safe, loss_dang, acc_safe, acc_dang


def loss_derivative(args, doing_refine, h_scores, h_scores_future, safe_mask, medium_mask, dang_mask,
                             num_safe, num_medium, num_dang, n_dim, t_dim):
    if doing_refine:
        deriv_safe_thres = 0.0
        deriv_medium_thres = 0.0
        deriv_dang_thres = 0.0

    else:
        deriv_safe_thres = args.grad_safe_thres
        deriv_medium_thres = args.grad_medium_thres
        deriv_dang_thres = args.grad_dang_thres


    loss_safe_deriv_full = tf.math.maximum(deriv_safe_thres - h_scores_future + 0.99 * h_scores,
                                           0) * safe_mask / (1e-12 + num_safe)
    loss_medium_deriv_full = tf.math.maximum(deriv_medium_thres - h_scores_future + 0.99 * h_scores,
                                             0) * medium_mask / (1e-12 + num_medium)
    loss_dang_deriv_full = tf.math.maximum(deriv_dang_thres - h_scores_future + 0.99 * h_scores,
                                           0) * dang_mask / (1e-12 + num_dang)

    loss_shape = get_loss_shape(args, n_dim, t_dim)

    loss_safe_deriv_full = tf.reshape(loss_safe_deriv_full, loss_shape)
    loss_medium_deriv_full = tf.reshape(loss_medium_deriv_full, loss_shape)
    loss_dang_deriv_full = tf.reshape(loss_dang_deriv_full, loss_shape)

    loss_safe_deriv = tf.reduce_sum(loss_safe_deriv_full)
    loss_medium_deriv = tf.reduce_sum(loss_medium_deriv_full)
    loss_dang_deriv = tf.reduce_sum(loss_dang_deriv_full)

    h_deriv_acc_safe = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                             tf.float32) * safe_mask) / (1e-12 + num_safe)
    h_deriv_acc_medium = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                               tf.float32) * medium_mask) / (1e-12 + num_medium)
    h_deriv_acc_dang = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                             tf.float32) * dang_mask) / (1e-12 + num_dang)

    h_deriv_acc_safe = tf.cond(tf.greater(num_safe, 0), lambda: h_deriv_acc_safe, lambda: -tf.constant(1.0))
    h_deriv_acc_medium = tf.cond(tf.greater(num_medium, 0), lambda: h_deriv_acc_medium, lambda: -tf.constant(1.0))
    h_deriv_acc_dang = tf.cond(tf.greater(num_dang, 0), lambda: h_deriv_acc_dang, lambda: -tf.constant(1.0))

    return loss_safe_deriv_full, loss_medium_deriv_full, loss_dang_deriv_full, \
           loss_safe_deriv, loss_medium_deriv, loss_dang_deriv, \
           h_deriv_acc_safe, h_deriv_acc_medium, h_deriv_acc_dang


def loss_regularization(args, action, primal_control, dist_info_vars,
                             ref_action_var, ref_mean_var, ref_log_std_var, n_dim, t_dim, safe_mask, num_safe):
    if args.use_policy_reference and args.refine_policy == False:  # TODO since they maintain the shape (N/T, T, 2)
        if args.residual_u:
            loss_reg_policy_full = tf.reduce_sum(tf.math.square(dist_info_vars["mean"]) / (n_dim * t_dim), axis=[2])
        else:
            if args.use_my_policy:
                loss_reg_policy_full = tf.reduce_sum(
                    tf.math.square(dist_info_vars["mean"] - ref_mean_var) / (n_dim * t_dim), axis=[2]
                )
            else:
                loss_reg_policy_full = tf.reduce_sum(
                    tf.math.square(dist_info_vars["mean"] - ref_mean_var) / (n_dim * t_dim), axis=[2]
                ) + tf.reduce_sum(
                    tf.math.square(dist_info_vars["log_std"] - ref_log_std_var) / (n_dim * t_dim), axis=[2]
                )
    elif args.high_level and args.refine_policy == False:
        if args.residual_u:
            if args.no_action_scaling:
                loss_reg_policy_full = tf.reduce_sum(tf.math.square(dist_info_vars["mean"]/tf.constant([[4.0, 0.15]])) / (n_dim * t_dim), axis=[2])
            else:
                loss_reg_policy_full = tf.reduce_sum(tf.math.square(dist_info_vars["mean"]) / (n_dim * t_dim), axis=[2])
        else:
            raise NotImplementedError
    elif args.use_nominal_controller and args.refine_policy == False:
        if args.residual_u:
            loss_reg_policy_full = tf.reduce_sum(tf.math.square(dist_info_vars["mean"]) / (n_dim * t_dim), axis=[2])
        else:
            # TODO(yue) this currently only supports PedSim & EasySim
            loss_reg_policy_full = tf.sqrt(tf.reduce_sum(
                tf.math.square(tf.reshape(dist_info_vars["mean"], [n_dim, t_dim, 2]) - ref_action_var), axis=[2])) / (
                                               n_dim * t_dim)
    else:
        if args.reg_for_all_control:
            loss_reg_policy_full = tf.reduce_sum(tf.math.square(action + primal_control) / (n_dim * t_dim), axis=[1])
        else:
            loss_reg_policy_full = tf.reduce_sum(tf.math.square(action) / (n_dim * t_dim), axis=[1])

        loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim, t_dim])

    if args.reg_with_safe_mask:
        if args.agent_cbf:
            loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim * t_dim, 1])
            loss_reg_policy_full = (loss_reg_policy_full * n_dim * t_dim * safe_mask) / num_safe
            if args.new_cbf_pol:
                loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim, t_dim, 2])
            else:
                loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim, t_dim, 6])
        else:
            loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim * t_dim])
            loss_reg_policy_full = (loss_reg_policy_full * n_dim * t_dim * safe_mask) / num_safe
            loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim, t_dim])
        loss_reg_policy = tf.reduce_sum(loss_reg_policy_full)
    else:
        if args.agent_cbf:
            loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim, t_dim, 1])  # TODO because of broadcast (1,1,1,1,1,1) + (1) = (2,2,2,2,2,2)
            if args.new_cbf_pol:
                loss_reg_policy = tf.reduce_sum(loss_reg_policy_full) * 2
            else:
                loss_reg_policy = tf.reduce_sum(loss_reg_policy_full) * 6
        else:
            loss_reg_policy = tf.reduce_sum(loss_reg_policy_full)

    return loss_reg_policy_full, loss_reg_policy


def setup_cbf(args, network, obs, _action, safe_mask, dang_mask, medium_mask, dist_info_vars=None,
              ref_action_var=None, ref_mean_var=None, ref_log_std_var=None, doing_refine=False):
    if args.agent_cbf:
        assert not any([args.use_mono, args.use_easy, args.use_ped])  # TODO only for NGSIM data for now

    joint_dict = {}

    cbf_intervals, cbf_ctrl_intervals = get_indices(args)

    n_dim = tf.cast(tf.shape(obs)[0], dtype=tf.float32)
    t_dim = tf.cast(tf.shape(obs)[1], dtype=tf.float32)

    obs = tf.reshape(obs, [-1, obs.shape[2]])
    action = get_actions_tf(args, _action, dist_info_vars, ref_action_var, ref_mean_var)
    state_input = tf.gather(obs, cbf_intervals, axis=1)
    if cbf_ctrl_intervals == "":
        primal_control = action * 0.0
    else:
        primal_control = tf.gather(obs, cbf_ctrl_intervals, axis=1)


    ### DEFINE COMPUTATION GRAPH
    h_scores = get_cbf_value_tf(args, state_input, network)  # TODO the network might not be used
    joint_dict["debug_score"] = h_scores
    state_tplus1 = get_next_state(args, state_input, action, primal_control)
    h_scores_future = get_cbf_value_tf(args, state_tplus1, network)
    joint_dict["debug_score_next"] = h_scores_future
    num_safe = tf.reduce_sum(safe_mask)
    num_dang = tf.reduce_sum(dang_mask)
    num_medium = tf.reduce_sum(medium_mask)

    loss_safe_full, loss_dang_full, loss_safe, loss_dang, acc_safe, acc_dang = \
        loss_barrier(args, h_scores, safe_mask, dang_mask, num_safe, num_dang, n_dim, t_dim)

    loss_safe_deriv_full, loss_medium_deriv_full, loss_dang_deriv_full, \
    loss_safe_deriv, loss_medium_deriv, loss_dang_deriv, h_deriv_acc_safe, h_deriv_acc_medium, h_deriv_acc_dang = \
        loss_derivative(args, doing_refine, h_scores, h_scores_future, safe_mask, medium_mask, dang_mask,
                             num_safe, num_medium, num_dang, n_dim, t_dim)

    loss_reg_policy_full, loss_reg_policy = loss_regularization(args, action, primal_control, dist_info_vars,
                             ref_action_var, ref_mean_var, ref_log_std_var, n_dim, t_dim, safe_mask, num_safe)

    total_loss_full = loss_safe_full * args.safe_loss_weight \
                      + loss_dang_full * args.dang_loss_weight \
                      + loss_safe_deriv_full * args.safe_deriv_loss_weight \
                      + loss_dang_deriv_full * args.dang_deriv_loss_weight \
                      + loss_medium_deriv_full * args.medium_deriv_loss_weight \
                      + loss_reg_policy_full * args.reg_policy_loss_weight

    # total_loss = tf.reduce_sum(total_loss_full)

    loss_crit = loss_safe * args.safe_loss_weight + loss_dang * args.dang_loss_weight
    loss_grad = loss_safe_deriv * args.safe_deriv_loss_weight + \
                loss_medium_deriv * args.medium_deriv_loss_weight + \
                loss_dang_deriv * args.dang_deriv_loss_weight
    loss_reg = loss_reg_policy * args.reg_policy_loss_weight

    total_loss = loss_crit + loss_grad + loss_reg

    # TODO(debug)
    # joint_dict["A"] = loss_crit
    # joint_dict["B"] = loss_grad
    # joint_dict["C"] = loss_reg
    # joint_dict["AB"]=loss_crit+loss_grad
    # joint_dict["AC"]=loss_crit+loss_reg
    # joint_dict["BC"]=loss_grad+loss_reg
    # joint_dict["AB+C"]=joint_dict["AB"]+loss_reg
    # joint_dict["ABC"]=loss_crit+loss_grad+loss_reg

    # joint_dict["sum(s)+sum(d)"] = loss_safe * args.safe_loss_weight + loss_dang * args.dang_loss_weight
    # joint_dict["sum(s+d)"] = tf.reduce_sum(loss_safe_full * args.safe_loss_weight + loss_dang_full * args.dang_loss_weight)
    #
    # joint_dict["sum(ds)+sum(dm)+sum(dd)"] = loss_safe_deriv * args.safe_deriv_loss_weight + \
    #                                         loss_medium_deriv * args.medium_deriv_loss_weight + \
    #                                         loss_dang_deriv * args.dang_deriv_loss_weight
    #
    # joint_dict["sum(ds+dm+dd)"] = tf.reduce_sum(loss_safe_deriv_full * args.safe_deriv_loss_weight + \
    #                                         loss_medium_deriv_full * args.medium_deriv_loss_weight + \
    #                                         loss_dang_deriv_full * args.dang_deriv_loss_weight)
    #
    # joint_dict["sum(s)+sum(d)+sum(ds)+sum(dm)+sum(dd)"] = loss_safe * args.safe_loss_weight + \
    #                                                       loss_dang * args.dang_loss_weight + \
    #                                                       loss_safe_deriv * args.safe_deriv_loss_weight + \
    #                                                       loss_medium_deriv * args.medium_deriv_loss_weight + \
    #                                                       loss_dang_deriv * args.dang_deriv_loss_weight
    # joint_dict["sum(s+d+ds+dm+dd)"] = tf.reduce_sum(loss_safe_full * args.safe_loss_weight +
    #                                                 loss_dang_full * args.dang_loss_weight +
    #                                                 loss_safe_deriv_full * args.safe_deriv_loss_weight +
    #                                                 loss_medium_deriv_full * args.medium_deriv_loss_weight +
    #                                                 loss_dang_deriv_full * args.dang_deriv_loss_weight )



    if args.use_my_policy ==False and args.refine_policy==False:
        total_loss += 0.0 * tf.reduce_sum(dist_info_vars["log_std"])

    # joint_dict["total_loss"] = total_loss
    # joint_dict["loss_safe"] = loss_safe * args.safe_loss_weight
    # joint_dict["loss_dang"] = loss_dang * args.dang_loss_weight
    # joint_dict["loss_crit"] = loss_safe * args.safe_loss_weight + loss_dang * args.dang_loss_weight
    # joint_dict["loss_safe_deriv"] = loss_safe_deriv * args.safe_deriv_loss_weight
    # joint_dict["loss_medium_deriv"] = loss_medium_deriv * args.medium_deriv_loss_weight
    # joint_dict["loss_dang_deriv"] = loss_dang_deriv * args.dang_deriv_loss_weight
    # joint_dict["loss_grad"] = loss_safe_deriv * args.safe_deriv_loss_weight \
    #                             + loss_dang_deriv * args.dang_deriv_loss_weight \
    #                             + loss_medium_deriv * args.medium_deriv_loss_weight
    # joint_dict["loss_reg_policy"] = loss_reg_policy * args.reg_policy_loss_weight



    joint_dict["loss_safe"] = loss_safe * args.safe_loss_weight
    joint_dict["loss_dang"] = loss_dang * args.dang_loss_weight
    joint_dict["loss_d_safe"] = loss_safe_deriv * args.safe_deriv_loss_weight
    joint_dict["loss_d_medium"] = loss_medium_deriv * args.medium_deriv_loss_weight
    joint_dict["loss_d_dang"] = loss_dang_deriv * args.dang_deriv_loss_weight
    joint_dict["loss_crit"] = loss_crit
    joint_dict["loss_grad"] = loss_grad
    joint_dict["loss_reg_policy"] = loss_reg
    joint_dict["total_loss"] = total_loss


    joint_dict["num_safe"] = num_safe
    joint_dict["num_medium"] = num_medium
    joint_dict["num_dang"] = num_dang
    joint_dict["acc_safe"] = acc_safe
    joint_dict["acc_dang"] = acc_dang
    joint_dict["h_deriv_acc_safe"] = h_deriv_acc_safe
    joint_dict["h_deriv_acc_medium"] = h_deriv_acc_medium
    joint_dict["h_deriv_acc_dang"] = h_deriv_acc_dang


    # # TODO(debug)
    # gradients_policy = tf.gradients(total_loss, get_policy_var_list())
    # for gk in gradients_policy:
    #     joint_dict["debug_pol_"+gk.name] = gk
    #
    # gradients_policy_reg = tf.gradients(loss_reg, get_policy_var_list())
    # for gk in gradients_policy_reg:
    #     joint_dict["debug_reg_pol"+gk.name] = gk
    #
    # gradients_policy_der = tf.gradients(loss_grad, get_policy_var_list())
    # for gk in gradients_policy_der:
    #     joint_dict["debug_der_pol" + gk.name] = gk

    if args.print_debug:
        # TODO DEBUG
        joint_dict["debug_state"] = state_input
        joint_dict["debug_score"] = h_scores
        joint_dict["debug_next_state"] = state_tplus1
        joint_dict["debug_next_score"] = h_scores_future
        joint_dict["debug_action"] = action
        joint_dict["debug_dist_mean"] = dist_info_vars["mean"]
        joint_dict["debug_ref_action"] = ref_action_var
        joint_dict["debug_condition"] = h_scores_future - 0.99 * h_scores
        joint_dict["debug_safe"] = safe_mask
        joint_dict["debug_medium"] = medium_mask
        joint_dict["debug_dang"] = dang_mask
        # TODO DEBUG(end)

        if args.debug_gradually:
            test_as=[0.5, 0.3, 0.1, 0.05, 0.03, 0.01, -0.01, -0.03, -0.05, -0.1, -0.3, -0.5]
            for ti, ta in enumerate(test_as):
                s_dbg = get_next_state(args, state_input, action+tf.constant([[ta, 0.0]]), primal_control)
                h_dbg = get_cbf_value_tf(args, s_dbg, network)
                joint_dict["debug_ha%d"%(ti)]=h_dbg

        if args.debug_gradually_omega:
            test_as=[0.5, 0.3, 0.1, 0.05, 0.03, 0.01, -0.01, -0.03, -0.05, -0.1, -0.3, -0.5]
            for ti, ta in enumerate(test_as):
                s_dbg = get_next_state(args, state_input, action+tf.constant([[0.0, ta]]), primal_control)
                h_dbg = get_cbf_value_tf(args, s_dbg, network)
                joint_dict["debug_hw%d"%(ti)]=h_dbg

    for key, value in joint_dict.items():
        if "debug" not in key:
            tf.summary.scalar(key, value)

    net_var_list = get_var_list()
    return cbf_intervals, cbf_ctrl_intervals, total_loss, total_loss_full, net_var_list, joint_dict


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
    elif args.use_easy:
        u_res = tf.Variable(tf.zeros([args.n_envs, 1, 2]), name='u_res')
    else:
        u_res = tf.Variable(tf.zeros([args.n_envs, 1, 2]), name='u_res')
    u_init = tf.assign(u_res, tf.zeros_like(u_res))
    # u_init = None

    cbf_intervals, cbf_ctrl_intervals, cbf_loss, cbf_loss_full, _, refine_dict = \
        setup_cbf(args, network, true_obs_var, true_action_var + u_res, safe_mask, dang_mask, medium_mask, doing_refine=True)

    gradients = tf.gradients(refine_dict["loss_grad"], u_res)

    # TODO(debug) monitoring some variables
    refine_dict["gradients0"] = gradients[0]
    refine_dict["safe_mask"] = safe_mask
    refine_dict["dang_mask"] = dang_mask
    refine_dict["medium_mask"] = medium_mask
    refine_dict["true_obs_var"] = true_obs_var
    refine_dict["u_res_prev"] = u_res * 1.0

    if args.use_mono:
        dynamics_ops = utils_mono.dynamics_mono
        old_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res, [-1, 1]),
                                        tf.reshape(true_action_var * 0.0, [-1, 1]))
    elif args.use_ped:
        dynamics_ops = utils_ped.dynamics_ped
        old_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))
    elif args.use_easy:
        dynamics_ops = utils_easy.dynamics_easy
        old_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))
    else:
        dynamics_ops = utils_ngsim.dynamics_attr
        old_state_tplus1 = dynamics_ops(args, tf.gather(tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]), cbf_intervals, axis=1),
                                        tf.reshape(true_action_var + u_res, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))

    old_h_scores_future = get_cbf_value_tf(args, old_state_tplus1, network)

    u_res_new = tf.assign(u_res, u_res - gradients[0] * args.refine_learning_rate)

    # compute future states and future scores again
    if args.use_mono:
        mod_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                              tf.reshape(true_action_var + u_res_new, [-1, 1]), tf.reshape(true_action_var * 0.0, [-1, 1]))
    elif args.use_ped:
        mod_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res_new, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))
    elif args.use_easy:
        mod_state_tplus1 = dynamics_ops(args, tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]),
                                        tf.reshape(true_action_var + u_res_new, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))
    else:
        mod_state_tplus1 = dynamics_ops(args, tf.gather(tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]]), cbf_intervals, axis=1),
                                        tf.reshape(true_action_var + u_res_new, [-1, 2]),
                                        tf.reshape(true_action_var * 0.0, [-1, 2]))

    mod_h_scores_future = get_cbf_value_tf(args, mod_state_tplus1, network)
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
    refine_input_list = [true_obs_var]

    cbf_intervals, cbf_ctrl_intervals = get_indices(args)

    obs = tf.reshape(true_obs_var, [-1, true_obs_var.shape[2]])
    state_input = tf.gather(obs, cbf_intervals, axis=1)

    if args.normalize_affordance:
        if args.use_mono or args.use_ped:
            raise NotImplementedError
        h_scores = network_cbf(normalize_affordance(state_input), args)
    else:
        h_scores = network_cbf(state_input, args)


    dhdx = tf.gradients(h_scores, state_input)

    refine_dict = {"dhdx": dhdx, "hx": h_scores}


    return cbf_intervals, cbf_ctrl_intervals, refine_input_list, refine_dict