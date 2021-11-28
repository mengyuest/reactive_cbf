import numpy as np
import tensorflow as tf
import hgail
import pickle
import rllab.misc.logger as logger

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


class LearnableCBF(object):
    def __init__(self, network, policy, env,
                 gradient_penalty, n_train_epochs, grad_norm_rescale, debug_nan,
                 train_batch_size, cbf_intervals, args):
        ### 0. CONFIGURATIONS
        self.network = network
        self.policy = policy
        self.env = env
        self.gradient_penalty = gradient_penalty

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.n_train_epochs = n_train_epochs
        self.grad_norm_rescale = grad_norm_rescale
        self.debug_nan = debug_nan

        self.train_batch_size = train_batch_size

        self.args = args

        self.cbf_intervals = get_ext_indices(cbf_intervals)
        if self.args.multilane_control and self.args.ctrl_intervals != "":
            print("Using Primal Control and PolicyNet as Rectification!")
            self.ctrl_intervals = get_ext_indices(self.args.ctrl_intervals)

        self.safe_dist_threshold = 1
        self.dang_dist_threshold = 0

        # this if for gamma
        self.safe_thres = 0.1
        self.dang_thres = 0.1
        self.grad_thres = 0.1

        self.safe_loss_weight = args.safe_loss_weight  # 1.0
        self.dang_loss_weight = args.dang_loss_weight  # 1.0
        self.safe_deriv_loss_weight = args.safe_deriv_loss_weight  # 1.0 #0.000001 #1.0
        self.dang_deriv_loss_weight = args.dang_deriv_loss_weight  # 1.0 #0.000001 #1.0
        self.medium_deriv_loss_weight = args.medium_deriv_loss_weight  # 3.0 #0.000003 #3.0

        self.grad_norm_rescale = 40.
        self.grad_norm_clip = 10000.

        self.reduced=False
        if args.attention:
            logger.log("len(cbf_intervals)=%d"%len(self.cbf_intervals))
            if len(self.cbf_intervals)==48:
                self.n_ego_feat = 6
                self.n_nei_feat = 7
            elif len(self.cbf_intervals)==46:
                self.n_ego_feat = 4
                self.n_nei_feat = 7
                self.reduced=True
            else:
                raise  NotImplementedError
        elif args.attractive:
            self.n_ego_feat = 8
            self.n_nei_feat = 9
        else:
            raise NotImplementedError
        self.n_feat = self.n_ego_feat + 6 * self.n_nei_feat

        self.feat_x_def = tf.constant([0.0, -15.0, 15.0, 0.0, -15.0, 15.0])
        self.feat_y_def = tf.constant([30.0, 30.0, 30.0, -30.0, -30.0, -30.0])

        ### 1. PLACEHOLDERS
        self.h_func_input = tf.placeholder(tf.float32,
                                           [None, self.n_feat])  # needs un-normalization when doing transition model
        self.state_input = tf.placeholder(tf.float32, [None, self.n_feat])
        self.safe_mask = tf.placeholder(tf.float32, [None])  # safe labels
        self.dang_mask = tf.placeholder(tf.float32, [None])  # dangerous labels
        self.medium_mask = tf.placeholder(tf.float32, [None])  # medium labels
        self.obs_mean_pl = tf.placeholder(tf.float32, [self.n_feat])
        self.obs_var_pl = tf.placeholder(tf.float32, [self.n_feat])
        self.primal_control = tf.placeholder(tf.float32, [None, 2])  # lane-keeping accel, omega

        ### 2. FIND THE POLICYNET OUTPUT
        is_recurrent = 1
        pol_obs_var = self.env.observation_space.new_tensor_variable(
            'pol_obs',
            extra_dims=1 + is_recurrent,
        )
        pol_state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name="pol_" + k)
            for k, shape in self.policy.state_info_specs
        }
        pol_state_info_vars_list = [pol_state_info_vars[k] for k in self.policy.state_info_keys]
        pol_dist_info_vars = self.policy.dist_info_sym(pol_obs_var,
                                                       pol_state_info_vars)  # dict(mean=means, log_std=log_stds)
        policy_action = pol_dist_info_vars["mean"]
        self.policy_output = tf.reshape(policy_action, [-1, 2]) * [[4.0, 0.15]]  # TODO switch to real control values
        self.pol_obs_var = pol_obs_var
        self.pol_state_info_vars_list = pol_state_info_vars_list

        ### 3. DEFINE MONITORING VALUES
        self.tf_dict = {}
        self.loss_safe_list = []
        self.loss_dang_list = []
        self.acc_safe_list = []
        self.acc_dang_list = []
        self.num_safe_list = []
        self.num_dang_list = []
        self.input_list = []
        self.mean_list = []
        self.var_list = []

        self.is_colliding_list=[]
        self.rmse_list=[]

        ### 4. DEFINE COMPUTATION GRAPH
        self.h_scores = tf.squeeze(self.network(self.h_func_input))

        # define losses
        # 1. safe set loss
        # 2. dangerous set loss
        # 3. derivative loss (this one related to policy)
        #    this needs a forward simulation
        num_safe = tf.reduce_sum(self.safe_mask)
        num_dang = tf.reduce_sum(self.dang_mask)
        num_medium = tf.reduce_sum(self.medium_mask)
        loss_safe = tf.reduce_sum(tf.math.maximum(-self.h_scores + self.safe_thres, 0) * self.safe_mask) / (
                    1e-5 + num_safe)
        loss_dang = tf.reduce_sum(tf.math.maximum(self.h_scores + self.dang_thres, 0) * self.dang_mask) / (
                    1e-5 + num_dang)

        acc_dang = tf.reduce_sum(tf.cast(tf.less_equal(self.h_scores, 0), tf.float32) * self.dang_mask) / (
                    1e-12 + num_dang)
        acc_safe = tf.reduce_sum(tf.cast(tf.greater_equal(self.h_scores, 0), tf.float32) * self.safe_mask) / (
                1e-12 + num_safe)

        acc_dang = tf.cond(tf.greater(num_dang, 0), lambda: acc_dang, lambda: -tf.constant(1.0))
        acc_safe = tf.cond(tf.greater(num_safe, 0), lambda: acc_safe, lambda: -tf.constant(1.0))

        if self.args.attractive:
            self.state_tplus1 = self.dynamics_attr(self.state_input, self.policy_output, self.primal_control)
        elif self.args.attention:
            self.state_tplus1 = self.dynamics(self.state_input, self.policy_output, self.primal_control)


        if self.args.normalized_cbf_input:
            self.normalized_state_tplus1 = (self.state_tplus1 - self.obs_mean_pl) / (tf.clip_by_value(tf.sqrt(self.obs_var_pl), 1e-6, 1e6))
            h_scores_future = tf.squeeze(self.network(self.normalized_state_tplus1))
        else:
            h_scores_future = tf.squeeze(self.network(self.state_tplus1))

        loss_safe_deriv = tf.reduce_sum(tf.math.maximum(
            self.grad_thres - h_scores_future + 0.99 * self.h_scores, 0) * self.safe_mask) / (
                                  1e-12 + num_safe)
        loss_dang_deriv = tf.reduce_sum(tf.math.maximum(
            self.grad_thres - h_scores_future + 0.99 * self.h_scores, 0) * self.dang_mask) / (
                                  1e-12 + num_dang)
        loss_medium_deriv = tf.reduce_sum(tf.math.maximum(
            self.grad_thres - h_scores_future + 0.99 * self.h_scores, 0) * self.medium_mask) / (
                                    1e-12 + num_medium)

        loss_reg_policy = tf.reduce_mean(tf.math.square(policy_action))

        total_loss = loss_safe * self.safe_loss_weight \
                     + loss_dang * self.dang_loss_weight \
                     + loss_safe_deriv * self.safe_deriv_loss_weight \
                     + loss_dang_deriv * self.dang_deriv_loss_weight \
                     + loss_medium_deriv * self.medium_deriv_loss_weight \
                     + loss_reg_policy * self.args.reg_policy_loss_weight

        self.tf_dict["learning_rate"] = self.learning_rate

        self.tf_dict["total_loss"] = total_loss
        self.tf_dict["loss_safe"] = loss_safe * self.safe_loss_weight
        self.tf_dict["loss_dang"] = loss_dang * self.dang_loss_weight

        self.tf_dict["loss_crit"] = loss_safe * self.safe_loss_weight + loss_dang * self.dang_loss_weight
        self.tf_dict["loss_grad"] = loss_safe_deriv * self.safe_deriv_loss_weight \
                                    + loss_dang_deriv * self.dang_deriv_loss_weight \
                                    + loss_medium_deriv * self.medium_deriv_loss_weight

        self.tf_dict["loss_safe_deriv"] = loss_safe_deriv * self.safe_deriv_loss_weight
        self.tf_dict["loss_dang_deriv"] = loss_dang_deriv * self.dang_deriv_loss_weight
        self.tf_dict["loss_medium_deriv"] = loss_medium_deriv * self.medium_deriv_loss_weight
        self.tf_dict["loss_reg_policy"] = loss_reg_policy * self.args.reg_policy_loss_weight

        self.tf_dict["num_dang"] = num_dang
        self.tf_dict["num_safe"] = num_safe
        self.tf_dict["acc_dang"] = acc_dang
        self.tf_dict["acc_safe"] = acc_safe

        ### 5. VARIABLES, GRADIENTS AND TRAINING OPERATIONS
        self.total_var_list = self.network.var_list \
                              + [x for x in self.policy.get_params(trainable=True)
                                 if "log_std" not in x.name]  # TODO(impl) since we are not using std
        self.gradients = tf.gradients(total_loss, self.total_var_list)
        clipped_gradients = hgail.misc.tf_utils.clip_gradients(
            self.gradients, self.grad_norm_rescale, self.grad_norm_clip)

        self.global_step = tf.Variable(0, name='cbf/global_step', trainable=False)
        self.train_op = self.optimizer.apply_gradients([(g, v)
                                                        for (g, v) in zip(clipped_gradients, self.total_var_list)],
                                                       global_step=self.global_step)

    # check feature definition at julia_pkgs/AutoRisk/src/extraction/feature_extractors.jl
    # function AutomotiveDrivingModels.pull_features!(
    #         ext::AttentionExtractor,...
    def dynamics(self, state, control, primal_control):
        offset = 2 if self.reduced else 0
        # state = tf.Print(state, [state[0, :3-offset], state[0, 3-offset:6-offset]], "")

        next_state = state * 1.0

        # perform precised updates (numerically)
        dT = 0.1
        discrete_num = 4
        dt = dT / discrete_num

        for _ in range(discrete_num):
            # symbol table
            if not self.reduced:
                x = next_state[:, 0]
                y = next_state[:, 1]
            theta = next_state[:, 2-offset]
            v = next_state[:, 3-offset]
            length=next_state[:, 4-offset]
            width=next_state[:, 5-offset]
            accel = control[:, 0] + primal_control[:, 0]
            omega = control[:, 1] + primal_control[:, 1]

            # 1-order dynamic
            dx = v * tf.cos(theta) * dt
            dy = v * tf.sin(theta) * dt
            ds = v * dt

            # 2-order dynamic
            dv = accel * dt
            dtheta = omega * dt

            # updates
            if not self.reduced:
                new_x = x + dx
                new_y = y + dy
            new_theta = theta + dtheta
            new_v = v + dv
            new_length = length
            new_width = width

            # neighbors
            nei_feat = tf.reshape(next_state[:, self.n_ego_feat:], [-1, self.n_nei_feat])
            nei_ind = nei_feat[:, 0]
            nei_x = nei_feat[:, 1]
            nei_y = nei_feat[:, 2]
            nei_theta=nei_feat[:, 3]
            nei_v=nei_feat[:, 4]
            nei_length=nei_feat[:, 5]
            nei_width=nei_feat[:, 6]

            # [1,2,3]->[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
            ds_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ds, [6]), [6, -1])), [-1])
            dtheta_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(dtheta, [6]), [6, -1])), [-1])
            new_v_cp =tf.reshape(tf.transpose(tf.reshape(tf.tile(new_v, [6]), [6, -1])), [-1])

            # 1-order dynamic #TODO (very important, think carefully)
            nei_dx = -nei_v * tf.sin(nei_theta) * dt
            nei_dy = nei_v * tf.cos(nei_theta) * dt

            # 2-order dynamic
            # (TODO) assuming not knowing neighbor's high-order dynamics
            nei_dtheta = 0.0 * dt
            nei_dv = 0.0 * dt

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
            new_nei_x = nei_ind * new_nei_x + (1-nei_ind)*(tf.tile(self.feat_x_def, [tf.shape(nei_ind)[0]//6])) #TODO
            new_nei_y = nei_ind * new_nei_y + (1-nei_ind)*(tf.tile(self.feat_y_def, [tf.shape(nei_ind)[0]//6])) #TODO
            new_nei_theta = nei_ind * new_nei_theta
            new_nei_v = nei_ind * new_nei_v + (1-nei_ind) * new_v_cp
            new_nei_length = nei_ind * new_nei_length
            new_nei_width = nei_ind * new_nei_width

            # merge them to a single tensor
            if self.reduced:
                new_ego = tf.stack([new_theta, new_v, new_length, new_width], axis=-1)
            else:
                new_ego = tf.stack([new_x, new_y, new_theta, new_v, new_length, new_width], axis=-1)
            new_nei = tf.stack(
                [nei_ind, new_nei_x, new_nei_y, new_nei_theta, new_nei_v, new_nei_length, new_nei_width], axis=-1)

            # new_nei = new_nei * nei_feat[:, 0:1] + default_val * (1-nei_feat[:, 0:1])
            new_nei = tf.reshape(new_nei, [-1, 6 * self.n_nei_feat])
            next_state = tf.concat([new_ego, new_nei], axis=-1)

        return next_state


    # check feature definition at julia_pkgs/AutoRisk/src/extraction/feature_extractors.jl
    # function AutomotiveDrivingModels.pull_features!(
    #         ext::AttractiveExtractor,...
    def dynamics_attr(self, state, control, primal_control):
        offset = 2 if self.reduced else 0

        next_state = state * 1.0

        # perform precised updates (numerically)
        dT = 0.1
        discrete_num = 4
        dt = dT / discrete_num

        for _ in range(discrete_num):
            # symbol table
            if not self.reduced:
                x1 = next_state[:, 0]
                x2 = next_state[:, 1]
            theta = next_state[:, 2-offset]
            v = next_state[:, 3-offset]
            length=next_state[:, 6-offset]
            width=next_state[:, 7-offset]
            accel = control[:, 0] + primal_control[:, 0]
            omega = control[:, 1] + primal_control[:, 1]

            # 1-order dynamic
            dx = v * tf.sin(theta) * dt
            ds = v * dt

            # 2-order dynamic
            dv = accel * dt
            dtheta = omega * dt

            # updates
            if not self.reduced:
                new_x1 = x1 + dx
                new_x2 = x2 + dx
            new_theta = theta + dtheta
            new_v = v + dv
            new_accel = accel
            new_omega = omega
            new_length = length
            new_width = width

            # neighbors
            nei_feat = tf.reshape(next_state[:, self.n_ego_feat:], [-1, self.n_nei_feat])
            nei_ind = nei_feat[:, 0]
            nei_x = nei_feat[:, 1]
            nei_y = nei_feat[:, 2]
            nei_theta=nei_feat[:, 3]
            nei_v=nei_feat[:, 4]
            nei_accel=nei_feat[:, 5]
            nei_omega=nei_feat[:, 6]
            nei_length=nei_feat[:, 7]
            nei_width=nei_feat[:, 8]

            # [1,2,3]->[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
            ds_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ds, [6]), [6, -1])), [-1])
            dtheta_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(dtheta, [6]), [6, -1])), [-1])
            new_v_cp =tf.reshape(tf.transpose(tf.reshape(tf.tile(new_v, [6]), [6, -1])), [-1])

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
            new_nei_x = nei_ind * new_nei_x + (1-nei_ind)*(tf.tile(self.feat_x_def, [tf.shape(nei_ind)[0]//6])) #TODO
            new_nei_y = nei_ind * new_nei_y + (1-nei_ind)*(tf.tile(self.feat_y_def, [tf.shape(nei_ind)[0]//6])) #TODO
            new_nei_theta = nei_ind * new_nei_theta
            new_nei_v = nei_ind * new_nei_v + (1-nei_ind) * new_v_cp
            new_nei_accel = nei_accel
            new_nei_omega = nei_omega
            new_nei_length = nei_ind * new_nei_length
            new_nei_width = nei_ind * new_nei_width

            # merge them to a single tensor
            if self.reduced:
                new_ego = tf.stack([new_theta, new_v, new_accel, new_omega, new_length, new_width], axis=-1)
            else:
                new_ego = tf.stack([new_x1, new_x2, new_theta, new_v, new_accel, new_omega, new_length, new_width], axis=-1)
            new_nei = tf.stack(
                [nei_ind, new_nei_x, new_nei_y, new_nei_theta, new_nei_v, new_nei_accel, new_nei_omega, new_nei_length, new_nei_width], axis=-1)

            new_nei = tf.reshape(new_nei, [-1, 6 * self.n_nei_feat])
            next_state = tf.concat([new_ego, new_nei], axis=-1)

        return next_state

    def get_safe_mask(self, s, safe_dist_threshold):
        # get the safe mask from the states
        # input: (batchsize, 48)
        # return: (batchsize, 1)

        # TODO out-of-road check
        bs = s.shape[0]
        neighbor_s = s[:, self.n_ego_feat:].reshape((bs, 6, self.n_nei_feat))
        nei_ind = neighbor_s[:, :, 0]

        offset=2 if self.reduced else 0

        # TODO a better collision model (maybe Minkovski Convex Collision Checking)
        # right now we just use (dx**2+dy**2)<ego_L/2+nei_L/2
        # dy=y-ego_length/2-nei_length/2*|cos(ego_theta)|-nei_width/2*|sin(nei_theta)|
        # dx=x-ego_width/2-nei_width/2*|cos(ego_theta)|-nei_width/2*|sin(nei_theta)|
        if self.args.attractive:
            ego_l = s[:, 6]
            ego_w = s[:, 7]
        else:
            ego_l = s[:, 4-offset]
            ego_w = s[:, 5-offset]
        ego_r=(((ego_l ** 2 + ego_w ** 2) ** 0.5) / 2)
        if self.args.attractive:
            nei_l=neighbor_s[:, :, 7]
            nei_w=neighbor_s[:, :, 8]
        else:
            nei_l=neighbor_s[:, :, 5]
            nei_w=neighbor_s[:, :, 6]
        nei_r=((nei_l ** 2 + nei_w ** 2) ** 0.5) / 2
        dx=neighbor_s[:, :, 1]
        dy=neighbor_s[:, :, 2]

        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + safe_dist_threshold)
        safe_mask = (np.sum(nei_collide, axis=1) < 0.5)  # no collision for each row, means that ego vehicle is safe

        return safe_mask

    def get_dang_mask(self, s, dang_dist_threshold):
        bs = s.shape[0]
        neighbor_s = s[:, self.n_ego_feat:].reshape((bs, 6, self.n_nei_feat))
        nei_ind = neighbor_s[:, :, 0]

        offset = 2 if self.reduced else 0
        if self.args.attractive:
            ego_l = s[:, 6]
            ego_w = s[:, 7]
        else:
            ego_l = s[:, 4 - offset]
            ego_w = s[:, 5 - offset]
        ego_r = (((ego_l ** 2 + ego_w ** 2) ** 0.5) / 2)
        if self.args.attractive:
            nei_l=neighbor_s[:, :, 7]
            nei_w=neighbor_s[:, :, 8]
        else:
            nei_l = neighbor_s[:, :, 5]
            nei_w = neighbor_s[:, :, 6]
        nei_r = ((nei_l ** 2 + nei_w ** 2) ** 0.5) / 2
        dx = neighbor_s[:, :, 1]
        dy = neighbor_s[:, :, 2]

        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dang_dist_threshold)
        dang_mask = (np.sum(nei_collide, axis=1) > 0.5)  # no collision for each row means that ego vehicle is safe

        return dang_mask

    def recover_true_obs(self, obs):
        return obs * np.sqrt(self.obs_var) + self.obs_mean

    def train(self, itr, samples_data, local_mode=False):
        # shape in samples_data
        # observations: (N/T, T, feat_dim)   T is the trajectory length, most case=200
        # and N/T = n_sim * n_veh  n_sim = N/T/n_veh is the num of simulations, n_veh is num of controlled vehicles

        if local_mode:
            self.obs_mean = samples_data["mean"][itr]
            self.obs_var = samples_data["var"][itr]
            observations = samples_data["input"][itr]
            batchsize = observations.shape[0]

        else:
            normalized_env = hgail.misc.utils.extract_normalizing_env(self.env)
            self.obs_mean = normalized_env._obs_mean
            self.obs_var = normalized_env._obs_var
            batchsize = samples_data["rewards"].shape[0]
            observations = samples_data["observations"]
            agent_infos = samples_data["agent_infos"]
            state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]

        self.loss_safe_list.append([])
        self.loss_dang_list.append([])
        self.acc_safe_list.append([])
        self.acc_dang_list.append([])
        self.num_safe_list.append([])
        self.num_dang_list.append([])

        for train_ep in range(self.n_train_epochs):

            self.loss_safe_list[-1].append([])
            self.loss_dang_list[-1].append([])
            self.acc_safe_list[-1].append([])
            self.acc_dang_list[-1].append([])
            self.num_safe_list[-1].append([])
            self.num_dang_list[-1].append([])

            indices = [xxx for xxx in range(observations.shape[0])]
            np.random.shuffle(indices)

            num_iters = ((batchsize - 1) // self.train_batch_size) + 1
            for train_itr in range(num_iters):
                batch = {
                    "observations": observations[
                        indices[train_itr * self.train_batch_size:(train_itr + 1) * self.train_batch_size]],
                    "state_info_list": [
                        xxx[indices[train_itr * self.train_batch_size:(train_itr + 1) * self.train_batch_size]]
                        for xxx in state_info_list],
                    "env_infos": samples_data["env_infos"],
                }
                self._train_batch(itr, train_ep, train_itr, num_iters, batch)

                # session = tf.get_default_session()
                # vars = [v for v in tf.trainable_variables() if "policy" in v.name]
                # vars_vals = session.run(vars)
                # for var, val in zip(vars, vars_vals):
                #     print("var: {}\tvalue: {}".format(var.name, val.flatten()[:4]))  # ...or sort it in a list....
                # print()

        # np.savez("snippets/tmp.npz", {key:np.array(xxx) for key,xxx in
        #                               [ ("loss_safe", self.loss_safe_list),
        #                                 ("loss_dang", self.loss_dang_list),
        #                                 ("acc_safe", self.acc_safe_list),
        #                                 ("acc_dang", self.acc_dang_list),
        #                                 ("num_safe", self.num_safe_list),
        #                                 ("num_dang", self.num_dang_list)]})

        self.input_list.append(observations)
        self.mean_list.append(self.obs_mean)
        self.var_list.append(self.obs_var)

        if not local_mode:
            self._save_data()


    def _save_data(self):
        with open("snippets/data/tmp_%s.pkl" % (self.args.exp_name), "wb") as pkl_f:
            pickle.dump({key: np.array(xxx) for key, xxx in
                         [("loss_safe", self.loss_safe_list),
                          ("loss_dang", self.loss_dang_list),
                          ("acc_safe", self.acc_safe_list),
                          ("acc_dang", self.acc_dang_list),
                          ("num_safe", self.num_safe_list),
                          ("num_dang", self.num_dang_list)]}, pkl_f)

        # with open("snippets/data/%s.pkl"%(self.args.exp_name), "wb") as pkl_f:
        #     pickle.dump({key:np.array(xxx) for key,xxx in
        #                  [("input", self.input_list),
        #                   ("mean", self.mean_list),
        #                   ("var", self.var_list)]
        #                  }, pkl_f)

    def _train_batch(self, itr, train_ep, train_itr, num_iters, batch):
        #     gather inputs from trajectories/paths
        #     input: states
        #     get environment mean and var to recover data
        #     derive labels: safe/dangerous
        #     compute dangerous/safe loss
        #     compute derivative
        #     compute action-imitation loss
        #     optimize for both policy and h function

        policy_needs_obs = batch["observations"]
        policy_needs_state = batch["state_info_list"]

        full_obs = batch["observations"].reshape((-1, batch["observations"].shape[-1]))

        true_full_obs = self.recover_true_obs(full_obs)

        true_obs = true_full_obs[:, self.cbf_intervals]

        # rectify to zero/one, as the original normalized wrapper is not reliable
        # (in hgail/hgail/envs/vectorized_normalized_env.py)
        for i in range(6):
            true_obs[:, self.n_ego_feat+self.n_nei_feat*i] = \
                (true_obs[:,self.n_ego_feat+self.n_nei_feat*i]>0.5).astype(np.float32)

        if self.args.normalized_cbf_input:
            h_func_input = full_obs[:, self.cbf_intervals]
        else:
            h_func_input = true_obs

        safe_mask = self.get_safe_mask(true_obs, self.safe_dist_threshold)
        dang_mask = self.get_dang_mask(true_obs, self.dang_dist_threshold)
        medium_mask = np.logical_and(~safe_mask, ~dang_mask)
        safe_mask = safe_mask.astype(dtype=np.float32)
        dang_mask = dang_mask.astype(dtype=np.float32)
        medium_mask = medium_mask.astype(dtype=np.float32)

        primal_control = np.zeros((full_obs.shape[0], 2))

        if self.args.multilane_control and self.args.ctrl_intervals != "":
            primal_control = true_full_obs[:, self.ctrl_intervals]

        # step learning rate (if needed)
        # if itr > self.args.n_itr//2:
        #     learning_rate = self.args.cbf_learning_rate / 10.
        # else:
        #     learning_rate = self.args.cbf_learning_rate

        learning_rate = self.args.cbf_learning_rate

        session = tf.get_default_session()

        feed_dict = {
            # self.policy_input: full_obs,
            self.h_func_input: h_func_input,  # normalized_obs,
            self.state_input: true_obs,
            self.safe_mask: safe_mask,
            self.dang_mask: dang_mask,
            self.medium_mask: medium_mask,
            self.obs_mean_pl: self.obs_mean[self.cbf_intervals],
            self.obs_var_pl: self.obs_var[self.cbf_intervals],
            self.pol_obs_var: policy_needs_obs,
            self.learning_rate: learning_rate,
            self.primal_control: primal_control,

        }

        feed_dict.update({tf_pl: np_val for tf_pl, np_val in zip(self.pol_state_info_vars_list, policy_needs_state)})

        _, np_dict = session.run([self.train_op, self.tf_dict], feed_dict=feed_dict)
        if train_itr == 0 and train_ep % (max(1, self.n_train_epochs // 4)) == 0:
            logger.log("cbf%1d[%02d/%02d] lr:%.4f loss:%.4f lcrit:%.4f lgrad:%.4f lreg:%.4f\t"
                  "lsafe:%.4f ldang:%.4f #s:%4d #d:%4d acc-s:%.4f acc-d:%.4f" % (
                      train_ep, train_itr, num_iters, np_dict["learning_rate"], np_dict["total_loss"],
                      np_dict["loss_crit"],
                      np_dict["loss_grad"], np_dict["loss_reg_policy"],
                      np_dict["loss_safe"], np_dict["loss_dang"], np_dict["num_safe"],
                      np_dict["num_dang"], np_dict["acc_safe"], np_dict["acc_dang"]))

        self.loss_safe_list[-1][-1].append(np_dict["loss_safe"])
        self.loss_dang_list[-1][-1].append(np_dict["loss_dang"])
        self.acc_safe_list[-1][-1].append(np_dict["acc_safe"])
        self.acc_dang_list[-1][-1].append(np_dict["acc_dang"])
        self.num_safe_list[-1][-1].append(np_dict["num_safe"])
        self.num_dang_list[-1][-1].append(np_dict["num_dang"])

        # for key in ["debug_h_scores","dang_mask","debug_dang_cast_value","debug_dang_mask_sum","debug_num_dang"]:
        #     print(key, np_dict[key])
        # next_obs = self.dynamics(true_obs, u)
        # safety_ratio = 1-np.mean(self.get_safe_mask(next_obs))
        # safety_ratio = np.mean(safety_ratio == 1)
        # # safety_ratios_epoch.append(safety_ratio)
        # loss_list_np, acc_list_np = out[-2], out[-1]
        # # loss_lists_np.append(loss_list_np)
        # # acc_lists_np.append(acc_list_np)
        # return safety_ratio, loss_list_np, acc_list_np

        # # an example to check for NaN errors
        # rewards = self.network.forward(obs, acts, deterministic=True)
        #
        # if np.any(np.isnan(rewards)) and self.debug_nan:
        #     import ipdb
        #     ipdb.set_trace()

        # return 0
