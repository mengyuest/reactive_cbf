from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf

import numpy as np #TODO(yue)
import hgail

import sandbox.rocky.tf.algos.utils_cbf as utils_cbf
import sandbox.rocky.tf.algos.utils_ngsim as utils_ngsim
import sandbox.rocky.tf.algos.utils_mono as utils_mono
import sandbox.rocky.tf.algos.utils_ped as utils_ped
from os.path import join as ospj
import time

import os  # TODO(yue)
from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False
import scipy
import numpy.matlib

class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.args = kwargs["args"]  # TODO(yue)
        self.kwargs = kwargs  # TODO(yue)
        if self.args.use_policy_reference: # TODO(yue)
            self.policy_as_ref = kwargs["policy_as_ref"]
        else:
            self.policy_as_ref = None
        # TODO(yue)
        if os.path.exists(ospj(self.args.full_log_dir, "train_log_curr.txt")):
            self.log_curr = None
            self.log_avg = None
        else:
            self.log_curr = open(ospj(self.args.full_log_dir, "train_log_curr.txt"), "a+")
            self.log_avg = open(ospj(self.args.full_log_dir, "train_log_avg.txt"), "a+")
        self.log_t = time.time()
        self.log_prev_t = time.time()


        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        #TODO(yue)
        self.grad_norm_rescale = 40.
        self.grad_norm_clip = 10000.
        self.tf_dict={}

        if self.args.refine_policy:
            if self.args.qp_solve:
                self.cbf_intervals, self.cbf_ctrl_intervals, self.refine_input_list, self.refine_dict = \
                        utils_cbf.init_opt_refine_qp(self.args, self.policy.recurrent, self.env, self.kwargs["network"])
            else:
                self.cbf_intervals, self.cbf_ctrl_intervals, self.refine_input_list, \
                    self.u_res, self.u_init, self.refine_op, self.refine_dict \
                        = utils_cbf.init_opt_refine(self.args, self.policy.recurrent, self.env,
                                                self.kwargs["network"], self.grad_norm_rescale, self.grad_norm_clip)
        else:
            if self.args.ref_policy:
                self.init_opt_ref()
            else:
                self.init_opt_base()


    def init_opt_base(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable('obs', extra_dims=1 + is_recurrent)
        action_var = self.env.action_space.new_tensor_variable( 'action', extra_dims=1 + is_recurrent)
        advantage_var = tensor_utils.new_tensor('advantage', ndim=1 + is_recurrent, dtype=tf.float32)
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * advantage_var)

        input_list = [obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list

        if is_recurrent:
            input_list.append(valid_var)

        # TODO(yue)
        if self.args.joint_cbf:
            true_obs_var = self.env.observation_space.new_tensor_variable('true_obs', extra_dims=1 + is_recurrent)
            true_action_var = self.env.action_space.new_tensor_variable('true_action', extra_dims=1 + is_recurrent)

            safe_mask = tf.placeholder(tf.float32, [None], name="safe_mask")  # safe labels
            dang_mask = tf.placeholder(tf.float32, [None], name="dang_mask")  # dangerous labels
            medium_mask = tf.placeholder(tf.float32, [None], name="medium_mask")  # medium labels

            self.cbf_intervals, self.cbf_ctrl_intervals, cbf_loss, cbf_loss_full, cbf_network_params, self.cbf_dict, self.joint_dict = \
                utils_cbf.setup_cbf(self.args, self.kwargs["network"], true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask)
            self.tf_dict["lr_cbf_loss"] = tf.reduce_sum(lr * cbf_loss_full * valid_var)
            surr_loss = surr_loss + self.tf_dict["lr_cbf_loss"]
            #TODO(yue) moved to extra list, because some of it is not sample-dividable
            extra_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]
            # self.cbf_optimizer = tf.train.RMSPropOptimizer(self.args.cbf_learning_rate)
            self.cbf_optimizer = tf.train.AdamOptimizer(self.args.cbf_learning_rate)
            self.gradients = tf.gradients(surr_loss, cbf_network_params)
            clipped_gradients = hgail.misc.tf_utils.clip_gradients(
                self.gradients, self.grad_norm_rescale, self.grad_norm_clip)
            self.global_step = tf.Variable(0, name='cbf/global_step', trainable=False)
            self.cbf_train_op = self.cbf_optimizer.apply_gradients([(g, v)
                              for (g, v) in zip(clipped_gradients, cbf_network_params)], global_step=self.global_step)
            self.input_list = input_list
            self.cbf_input_list = extra_list
            self.optimizer.update_opt(
                loss=surr_loss,
                target=self.policy,
                leq_constraint=(mean_kl, self.step_size),
                inputs=input_list,
                extra_inputs=extra_list,
                constraint_name="mean_kl"
            )

            #TODO(yue) files to save
            self.save_h_deriv_data=[]

        else:
            self.optimizer.update_opt(
                loss=surr_loss,
                target=self.policy,
                leq_constraint=(mean_kl, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl"
            )
        return dict()

    def init_opt_ref(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable('obs', extra_dims=1 + is_recurrent)
        action_var = self.env.action_space.new_tensor_variable( 'action', extra_dims=1 + is_recurrent)

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

        input_list = [obs_var, action_var]+ state_info_vars_list

        #TODO(yue)
        if self.args.use_policy_reference:
            ref_action_var = self.env.action_space.new_tensor_variable('ref_action', extra_dims=1 + is_recurrent)
            ref_mean_var = self.env.action_space.new_tensor_variable('ref_mean', extra_dims=1 + is_recurrent)
            ref_log_std_var = self.env.action_space.new_tensor_variable('ref_log_std', extra_dims=1 + is_recurrent)
            input_list += [ref_action_var, ref_mean_var, ref_log_std_var]
        elif self.args.use_nominal_controller:
            ref_action_var = self.env.action_space.new_tensor_variable('ref_action', extra_dims=1 + is_recurrent)
            input_list += [ref_action_var]
            ref_mean_var = None
            ref_log_std_var = None
        else:
            ref_action_var = None
            ref_mean_var = None
            ref_log_std_var = None

        if is_recurrent:
            input_list.append(valid_var)

        # TODO(yue)
        true_obs_var = self.env.observation_space.new_tensor_variable('true_obs', extra_dims=1 + is_recurrent)
        true_action_var = self.env.action_space.new_tensor_variable('true_action', extra_dims=1 + is_recurrent)

        safe_mask = tf.placeholder(tf.float32, [None], name="safe_mask")  # safe labels
        dang_mask = tf.placeholder(tf.float32, [None], name="dang_mask")  # dangerous labels
        medium_mask = tf.placeholder(tf.float32, [None], name="medium_mask")  # medium labels

        self.cbf_intervals, self.cbf_ctrl_intervals, cbf_loss, cbf_loss_full, cbf_network_params, self.cbf_dict, self.joint_dict = \
            utils_cbf.setup_cbf(self.args, self.kwargs["network"], true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask,
                           dist_info_vars, ref_action_var, ref_mean_var, ref_log_std_var)
        self.tf_dict["lr_cbf_loss"] = tf.reduce_sum(cbf_loss_full * valid_var)
        # self.tf_dict["lr_cbf_loss"] = self.cbf_dict["h_scores"][0]

        #TODO(yue) moved to extra list, because some of it is not sample-dividable
        extra_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]
        # self.cbf_optimizer = tf.train.RMSPropOptimizer(self.args.cbf_learning_rate)
        self.cbf_optimizer = tf.train.AdamOptimizer(self.args.cbf_learning_rate)
        cbf_gradients = tf.gradients(self.tf_dict["lr_cbf_loss"], cbf_network_params)
        cbf_clipped_gradients = hgail.misc.tf_utils.clip_gradients(
            cbf_gradients, self.grad_norm_rescale, self.grad_norm_clip)
        self.cbf_global_step = tf.Variable(0, name='cbf/global_step', trainable=False)
        self.cbf_train_op = self.cbf_optimizer.apply_gradients([(g, v)
                          for (g, v) in zip(cbf_clipped_gradients, cbf_network_params)], global_step=self.cbf_global_step)
        self.input_list = input_list
        self.cbf_input_list = extra_list

        # TODO(debug)
        self.tf_dict["lr_cbf_loss"] = self.tf_dict["lr_cbf_loss"] + 0.0 * tf.reduce_sum(dist_info_vars["log_std"])

        if self.args.alternative_update:
            assert self.args.joint_for_policy_only==False and self.args.joint_for_cbf_only==False

            params_h=cbf_network_params
            grad_h = tf.gradients(self.tf_dict["lr_cbf_loss"], params_h)
            grad_h_clip = hgail.misc.tf_utils.clip_gradients(grad_h, self.grad_norm_rescale, self.grad_norm_clip)
            global_step_h = tf.Variable(0, name='joint/global_step_h', trainable=False)
            optimizer_h = tf.train.AdamOptimizer(self.args.joint_learning_rate)
            self.train_op_h = optimizer_h.apply_gradients([(g, v) for (g, v) in zip(grad_h_clip, params_h)],
                                                                       global_step=global_step_h)

            params_a = self.policy.get_params(trainable=True)
            grad_a = tf.gradients(self.tf_dict["lr_cbf_loss"], params_a)
            grad_a_clip = hgail.misc.tf_utils.clip_gradients(grad_a, self.grad_norm_rescale, self.grad_norm_clip)
            global_step_a = tf.Variable(0, name='joint/global_step_a', trainable=False)
            optimizer_a = tf.train.AdamOptimizer(self.args.joint_learning_rate)
            self.train_op_a = optimizer_a.apply_gradients([(g, v) for (g, v) in zip(grad_a_clip, params_a)],
                                                                       global_step=global_step_a)

        else:
            # self.joint_optimizer = tf.train.RMSPropOptimizer(self.args.joint_learning_rate)
            self.joint_optimizer = tf.train.AdamOptimizer(self.args.joint_learning_rate)
            if self.args.joint_for_policy_only:
                joint_network_params = self.policy.get_params(trainable=True)
            elif self.args.joint_for_cbf_only:
                joint_network_params = cbf_network_params
            else:
                joint_network_params = cbf_network_params + self.policy.get_params(trainable=True)
            joint_gradients = tf.gradients(self.tf_dict["lr_cbf_loss"], joint_network_params)
            self.joint_dict["lr_cbf_loss"] = self.tf_dict["lr_cbf_loss"]
            for gi in range(len(joint_gradients)):
                self.joint_dict["gradients_%d"%gi] = joint_gradients[gi]

            self.joint_global_step = tf.Variable(0, name='joint/global_step', trainable=False)
            if self.args.disable_joint_clip_norm:
                self.joint_train_op = self.joint_optimizer.apply_gradients([(g, v)
                                                                            for (g, v) in
                                                                            zip(joint_gradients, joint_network_params)],
                                                                           global_step=self.joint_global_step)
            else:
                joint_clipped_gradients = hgail.misc.tf_utils.clip_gradients(
                    joint_gradients, self.grad_norm_rescale, self.grad_norm_clip)
                for gi in range(len(joint_clipped_gradients)):
                    self.joint_dict["clipped_gradients_%d" % gi] = joint_clipped_gradients[gi]
                self.joint_train_op = self.joint_optimizer.apply_gradients([(g, v)
                                                                        for (g, v) in
                                                                        zip(joint_clipped_gradients, joint_network_params)],
                                                                       global_step=self.joint_global_step)



        #TODO(yue) files to save
        self.save_h_deriv_data=[]

        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        def style(a, b=None, c=None):
            if c is None:
                if b is None:
                    return "%.4f -> %.4f (%.4f)" % (a[0], a[-1], np.mean(a))
                else:
                    return "%.4f %.4f -> %.4f %.4f (%.4f %.4f)" % (
                        a[0], b[0], a[-1], b[-1], np.mean(a), np.mean(b)
                    )
            else:
                return "%.4f %.4f %.4f -> %.4f %.4f %.4f (%.4f %.4f %.4f)" % (
                    a[0], b[0],c[0], a[-1], b[-1], c[-1], np.mean(a), np.mean(b), np.mean(c)
                )

        def print_aff(aff, name):
            print(name)
            print("c:%.4f  ll:%.4f rl:%.4f lr:%.4f rr:%.4f"%(aff[0], aff[1], aff[2], aff[3], aff[4]))
            print("th:%.4f v:%.4f  a:%.4f  w:%.4f  L:%.4f  W:%.4f"%(aff[5], aff[6], aff[7], aff[8], aff[9], aff[10]))
            # offset=11
            for i in range(6):
                offset = 11 + 9*i
                print("nei%d-%d x:%.4f y:%.4f th:%.4f v:%.4f a:%.4f w:%.4f L:%.4f W:%.4f" % (
                    i, aff[offset], aff[offset+1], aff[offset+2], aff[offset+3], aff[offset+4], aff[offset+5], aff[offset+6], aff[offset+7], aff[offset+8]))
            print("accel [%.7f, %.7f, %.7f, %.7f, %.7f, %.7f]"%tuple([aff[11+9*i+5] for i in range(6)]))
            print("omega [%.7f, %.7f, %.7f, %.7f, %.7f, %.7f]"%tuple([aff[11+9*i+6] for i in range(6)]))
            print()

        if self.args.save_data:
            buffer={}

        if self.args.ref_policy:
            all_input_values = tuple(ext.extract(
                samples_data,
                "observations", "actions",
            ))
        else:
            all_input_values = tuple(ext.extract(
                samples_data,
                "observations", "actions", "advantages"
            ))
        agent_infos = samples_data["agent_infos"]
        if self.args.ref_policy:
            state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
            all_input_values += tuple(state_info_list)


            # get possible policy_as_ref outputs
            if self.args.use_policy_reference:
                if self.args.residual_u:
                    ref_action = agent_infos["ref_actions"]
                    ref_mean = agent_infos["ref_mean"]
                    ref_log_std = agent_infos["ref_log_std"]
                else:
                    self.policy_as_ref.reset([True]*all_input_values[0].shape[0])
                    ref_mean=[]
                    ref_log_std=[]
                    ref_action=[]
                    for ti in range(all_input_values[0].shape[1]):
                        actions, agent_infos = self.policy_as_ref.get_actions(all_input_values[0][:, ti])
                        ref_action.append(actions)
                        ref_mean.append(agent_infos["mean"])
                        ref_log_std.append(agent_infos["log_std"])

                    ref_action = np.stack(ref_action, axis=1)
                    ref_mean = np.stack(ref_mean, axis=1)
                    ref_log_std = np.stack(ref_log_std, axis=1)
                all_input_values += tuple([ref_action, ref_mean, ref_log_std])
            elif self.args.use_nominal_controller:
                ref_action = samples_data["env_infos"]["controls"]
                all_input_values += tuple([ref_action,])
                if self.args.save_data:
                    buffer["ref_action"] = ref_action

        else:
            state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
            dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        # logger.log("Computing loss before")
        if self.args.joint_cbf:  #TODO(yue)
            # TODO (compute normalized features)
            obs_mean = samples_data["env_infos"]["obs_mean"]
            obs_var = samples_data["env_infos"]["obs_var"]

            env_is_colliding = samples_data["env_infos"]["is_colliding"]
            env_out_of_lane  = samples_data["env_infos"]["out_of_lane"]

            norm_obs = all_input_values[0]
            norm_actions = all_input_values[1]
            true_obs = norm_obs * np.sqrt(obs_var) + obs_mean
            if self.args.save_data:
                buffer["true_obs"] = true_obs
            if self.args.local_data_path is not None:
                buffer = np.load(ospj(self.args.local_data_path, "buffer_%d.npz"%(itr+1)))["buffer"].item()
                true_obs = buffer["true_obs"]

            if self.args.clip_affordance:
                if self.args.use_mono or self.args.use_ped:
                    raise NotImplementedError
                else:
                    true_obs[:, :, self.cbf_intervals] = utils_cbf.post_process_affordance(true_obs[:, :, self.cbf_intervals])

            # TODO (compute save/dang masks)
            if self.args.use_mono:
                true_actions = norm_actions * np.array([[4,]])
                safe_mask = utils_mono.get_safe_mask_mono(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane,
                                               self.args.safe_dist_threshold, self.args.safe_dist_threshold_side,
                                               check_safe=True)
                dang_mask = utils_mono.get_safe_mask_mono(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane,
                                               self.args.dang_dist_threshold, self.args.dang_dist_threshold_side,
                                               check_safe=False)
            elif self.args.use_ped:
                true_actions = norm_actions * np.array([[4, 4]])
                safe_mask = utils_ped.get_safe_mask_ped(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane,
                                               self.args.safe_dist_threshold, self.args.safe_dist_threshold_side,
                                               check_safe=True)
                dang_mask = utils_ped.get_safe_mask_ped(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane,
                                               self.args.dang_dist_threshold, self.args.dang_dist_threshold_side,
                                               check_safe=False)
            else:
                true_actions = norm_actions * np.array([[4, 0.15]])
                safe_mask = utils_ngsim.get_safe_mask(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals], env_out_of_lane,
                                               self.args.safe_dist_threshold, self.args.safe_dist_threshold_side, check_safe=True)
                dang_mask = utils_ngsim.get_safe_mask(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals], env_out_of_lane,
                                               self.args.dang_dist_threshold, self.args.dang_dist_threshold_side, check_safe=False)

                # for npi in range(len(self.cbf_intervals)):
                #     true_obs[:, :, self.cbf_intervals[npi]] = safe_mask.reshape((true_obs.shape[0], true_obs.shape[1]))
                # for npi in range(1, len(self.cbf_intervals), 2):
                #     true_obs[:, :, self.cbf_intervals[npi]] = dang_mask.reshape((true_obs.shape[0], true_obs.shape[1]))
                # true_obs = true_obs*0.0
                # true_obs[:, :, self.cbf_intervals[0]] = coll_mask1.reshape((true_obs.shape[0], true_obs.shape[1]))
                # true_obs[:, :, self.cbf_intervals[1]] = out_mask1.reshape((true_obs.shape[0], true_obs.shape[1]))
                # true_obs[:, :, self.cbf_intervals[2]] = coll_mask2.reshape((true_obs.shape[0], true_obs.shape[1]))
                # true_obs[:, :, self.cbf_intervals[3]] = out_mask2.reshape((true_obs.shape[0], true_obs.shape[1]))



            medium_mask = np.logical_and(~safe_mask, ~dang_mask)

            safe_mask = safe_mask.astype(dtype=np.float32)
            dang_mask = dang_mask.astype(dtype=np.float32)
            medium_mask = medium_mask.astype(dtype=np.float32)

            if self.args.save_data:
                buffer["true_actions"] = true_actions
                buffer["safe_mask"] = safe_mask
                buffer["dang_mask"] = dang_mask
                buffer["medium_mask"] = medium_mask
            if self.args.local_data_path is not None:
                true_actions = buffer["true_actions"]
                safe_mask = buffer["safe_mask"]
                dang_mask = buffer["dang_mask"]
                medium_mask = buffer["medium_mask"]

            extra_input_values = (true_obs, true_actions, safe_mask, dang_mask, medium_mask)

            if self.args.cbf_iter_per_epoch==0:
                num_repeat=0
            else:
                num_repeat = self.args.cbf_iter_per_epoch if itr < self.args.cbf_decay_after else max(self.args.cbf_iter_per_epoch // 10, 1)
            total_loss_tmp_list=[]
            lr_cbf_loss_tmp_list=[]
            loss_crit_tmp_list=[]
            loss_grad_tmp_list=[]
            loss_safe_tmp_list=[]
            loss_dang_tmp_list=[]
            acc_safe_tmp_list=[]
            acc_dang_tmp_list=[]

            h_deriv_acc_safe_list=[]
            h_deriv_acc_dang_list=[]
            h_deriv_acc_medium_list=[]
            # save_h_deriv_list=[]

            if num_repeat>0:
                for ti in range(num_repeat):
                    if self.args.accumulation_mode:
                        for inner_i in range(true_obs.shape[1]):
                            np_dict, _ = tf.get_default_session().run([self.cbf_dict, self.cbf_train_op],
                                                                      feed_dict={_k: _v for _k, _v in
                                                                                 zip(
                                                                                     self.input_list + self.cbf_input_list,
                                                                                     [inner_x[:, inner_i:inner_i+1] for inner_x in all_input_values] +
                                                                                     [true_obs[:, inner_i:inner_i+1],
                                                                                      true_actions[:, inner_i:inner_i+1],
                                                                                      safe_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i],
                                                                                      dang_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i],
                                                                                      medium_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i]]
                                                                                 )})
                            total_loss_tmp_list.append(np_dict["total_loss"])
                            lr_cbf_loss_tmp_list.append(np_dict["lr_cbf_loss"])
                            loss_crit_tmp_list.append(np_dict["loss_crit"])
                            loss_grad_tmp_list.append(np_dict["loss_grad"])
                            loss_safe_tmp_list.append(np_dict["loss_safe"])
                            loss_dang_tmp_list.append(np_dict["loss_dang"])
                            acc_safe_tmp_list.append(np_dict["acc_safe"])
                            acc_dang_tmp_list.append(np_dict["acc_dang"])

                            h_deriv_acc_safe_list.append(np_dict["h_deriv_acc_safe"])
                            h_deriv_acc_dang_list.append(np_dict["h_deriv_acc_dang"])
                            h_deriv_acc_medium_list.append(np_dict["h_deriv_acc_medium"])
                            # save_h_deriv_list.append(np_dict["save_h_deriv_data"])


                    else:
                        np_dict, _ = tf.get_default_session().run([self.cbf_dict, self.cbf_train_op],
                                                 feed_dict={_k:_v for _k,_v in zip(self.input_list+self.cbf_input_list, all_input_values+extra_input_values)})



                        total_loss_tmp_list.append(np_dict["total_loss"])
                        lr_cbf_loss_tmp_list.append(np_dict["lr_cbf_loss"])
                        loss_crit_tmp_list.append(np_dict["loss_crit"])
                        loss_grad_tmp_list.append(np_dict["loss_grad"])
                        loss_safe_tmp_list.append(np_dict["loss_safe"])
                        loss_dang_tmp_list.append(np_dict["loss_dang"])
                        acc_safe_tmp_list.append(np_dict["acc_safe"])
                        acc_dang_tmp_list.append(np_dict["acc_dang"])

                        h_deriv_acc_safe_list.append(np_dict["h_deriv_acc_safe"])
                        h_deriv_acc_dang_list.append(np_dict["h_deriv_acc_dang"])
                        h_deriv_acc_medium_list.append(np_dict["h_deriv_acc_medium"])
                        # save_h_deriv_list.append(np_dict["save_h_deriv_data"])

                logger.log('CBF--loss:     %s' % (style(total_loss_tmp_list, lr_cbf_loss_tmp_list)))
                logger.log('loss crit/grad:%s' % (style(loss_crit_tmp_list, loss_grad_tmp_list)))
                logger.log('loss  reg:     %.4f' % (np_dict["loss_reg_policy"]))
                logger.log('loss safe/dang:%s' % (style(loss_safe_tmp_list, loss_dang_tmp_list)))
                logger.log('acc  safe/dang:%s' % (style(acc_safe_tmp_list, acc_dang_tmp_list)))
                logger.log('num  sa/med/da:%d %d %d' % (np_dict["num_safe"], np_dict["num_medium"], np_dict["num_dang"]))
                logger.log('dacc sa/med/da:%s' % (style(h_deriv_acc_safe_list, h_deriv_acc_medium_list, h_deriv_acc_dang_list)))

                # TODO(yue) save to files
                # np.savez("%s/h_%d.npz"%(self.args.full_log_dir, itr),
                #          safe_mask=safe_mask, dang_mask=dang_mask, medium_mask=medium_mask, h_deriv=np.stack(save_h_deriv_list))

        else:
            extra_input_values = tuple([])
        if self.args.ref_policy:
            total_loss_tmp_list = []
            loss_crit_tmp_list = []
            loss_grad_tmp_list = []
            loss_safe_tmp_list = []
            loss_dang_tmp_list = []
            acc_safe_tmp_list = []
            acc_dang_tmp_list = []

            loss_reg_tmp_list=[]

            h_deriv_acc_safe_list = []
            h_deriv_acc_dang_list = []
            h_deriv_acc_medium_list = []
            logger.log("\nJoint Optimization Phase")
            if self.args.joint_iter_per_epoch==0:
                num_repeat=0
            else:
                num_repeat = self.args.joint_iter_per_epoch if itr < self.args.joint_decay_after else max(self.args.joint_iter_per_epoch // 10, 1)

            if num_repeat>0:
                for ti in range(num_repeat):
                    if self.args.accumulation_mode:
                        accu_total_loss=0
                        accu_loss_crit=0
                        accu_loss_grad=0
                        accu_loss_safe=0
                        accu_loss_dang=0
                        accu_acc_safe=0
                        accu_acc_dang=0
                        accu_h_deriv_acc_safe=0
                        accu_h_deriv_acc_dang = 0
                        accu_h_deriv_acc_medium=0
                        accu_loss_reg_policy=0

                        time_len = true_obs.shape[1]
                        for inner_i in range(time_len):
                            np_dict, _ = tf.get_default_session().run([self.joint_dict, self.joint_train_op],
                                                                      feed_dict={_k: _v for _k, _v in
                                                                                 zip(
                                                                                     self.input_list + self.cbf_input_list,
                                                                                     [inner_x[:, inner_i:inner_i+1] for inner_x in all_input_values] +
                                                                                     [true_obs[:, inner_i:inner_i+1],
                                                                                      true_actions[:, inner_i:inner_i+1],
                                                                                      safe_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i],
                                                                                      dang_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i],
                                                                                      medium_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i]]
                                                                                 )})
                            accu_total_loss += np_dict["total_loss"]
                            accu_loss_crit += np_dict["loss_crit"]
                            accu_loss_grad += np_dict["loss_grad"]
                            accu_loss_safe += np_dict["loss_safe"]
                            accu_loss_dang += np_dict["loss_dang"]

                            accu_loss_reg_policy+=np_dict["loss_reg_policy"]

                            curr_num_safe = np.sum(safe_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i])
                            curr_num_dang = np.sum(dang_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i])
                            curr_num_medium = np.sum(medium_mask.reshape((true_obs.shape[0], true_obs.shape[1]))[:,inner_i])

                            accu_acc_safe += np_dict["acc_safe"] * curr_num_safe
                            accu_acc_dang += np_dict["acc_dang"] * curr_num_dang

                            accu_h_deriv_acc_safe += np_dict["h_deriv_acc_safe"] * curr_num_safe
                            accu_h_deriv_acc_dang += np_dict["h_deriv_acc_dang"] * curr_num_dang
                            accu_h_deriv_acc_medium += np_dict["h_deriv_acc_medium"] * curr_num_medium


                        total_loss_tmp_list.append(accu_total_loss/time_len)
                        # lr_cbf_loss_tmp_list.append(np_dict["lr_cbf_loss"])
                        loss_crit_tmp_list.append(accu_loss_crit/time_len)
                        loss_grad_tmp_list.append(accu_loss_grad/time_len)
                        loss_safe_tmp_list.append(accu_loss_safe/time_len)
                        loss_dang_tmp_list.append(accu_loss_dang/time_len)
                        acc_safe_tmp_list.append(accu_acc_safe / (np.sum(safe_mask)+ 1e-8))
                        acc_dang_tmp_list.append(accu_acc_dang / (np.sum(dang_mask)+ 1e-8))

                        h_deriv_acc_safe_list.append(accu_h_deriv_acc_safe / (np.sum(safe_mask)+ 1e-8))
                        h_deriv_acc_dang_list.append(accu_h_deriv_acc_dang / (np.sum(dang_mask)+ 1e-8))
                        h_deriv_acc_medium_list.append(accu_h_deriv_acc_medium / (np.sum(medium_mask)+ 1e-8))

                        loss_reg_tmp_list.append(accu_loss_reg_policy/time_len)
                    else:
                        if self.args.alternative_update:
                            if self.args.run_both:
                                train_ops = [self.train_op_h, self.train_op_a]
                            else:
                                train_ops = [self.train_op_h if (itr//self.args.alternative_t)%2==0 else self.train_op_a]
                        else:
                            train_ops = [self.joint_train_op]

                        utils_cbf.dbg_print_cbf_params_values(self.args)
                        print()
                        for train_op_item in train_ops:
                            np_dict, _ = tf.get_default_session().run([self.joint_dict, train_op_item],
                                                              feed_dict={_k: _v for _k, _v in
                                                                         zip(self.input_list + self.cbf_input_list,
                                                                             all_input_values + extra_input_values)})

                        grad_keys = [grad_key for grad_key in np_dict if "gradients" in grad_key]
                        grad_len = len(grad_keys) // 2
                        for gi in range(grad_len):
                            print(gi, np_dict["gradients_%d" % gi].flatten())#,
                                  #np_dict["clipped_gradients_%d" % gi].flatten()[:3])
                        print()
                        utils_cbf.dbg_print_cbf_params_values(self.args)
                        print("LOSS=",np_dict["lr_cbf_loss"])
                        print("s=",np_dict["state_input"][0].flatten())
                        for key in np_dict:
                            if "dbgpt" in key:
                                print("%-12s" % key, np_dict[key][0].shape, list(np_dict[key][0].flatten()))

                        # print score (partial)
                        print("score", list(np_dict["h_scores"].flatten()[::50]))

                        # print loss full (partial)
                        print("loss", list(np_dict["debug_loss_full"].flatten()[::50]))
                        np.savez("%s/debug.npz" % (self.args.full_log_dir), score=np_dict["h_scores"], loss=np_dict["debug_loss_full"])

                        exit()



                        total_loss_tmp_list.append(np_dict["total_loss"])
                        # lr_cbf_loss_tmp_list.append(np_dict["lr_cbf_loss"])
                        loss_crit_tmp_list.append(np_dict["loss_crit"])
                        loss_grad_tmp_list.append(np_dict["loss_grad"])
                        loss_safe_tmp_list.append(np_dict["loss_safe"])
                        loss_dang_tmp_list.append(np_dict["loss_dang"])
                        acc_safe_tmp_list.append(np_dict["acc_safe"])
                        acc_dang_tmp_list.append(np_dict["acc_dang"])

                        h_deriv_acc_safe_list.append(np_dict["h_deriv_acc_safe"])
                        h_deriv_acc_dang_list.append(np_dict["h_deriv_acc_dang"])
                        h_deriv_acc_medium_list.append(np_dict["h_deriv_acc_medium"])

                        loss_reg_tmp_list.append(np_dict["loss_reg_policy"])

                logger.log('CBF--loss:     %s' % (style(total_loss_tmp_list)))
                logger.log('loss crit/grad:%s' % (style(loss_crit_tmp_list, loss_grad_tmp_list)))
                logger.log('loss  reg:     %s' % (style(loss_reg_tmp_list)))
                logger.log('loss safe/dang:%s' % (style(loss_safe_tmp_list, loss_dang_tmp_list)))
                logger.log('acc  safe/dang:%s' % (style(acc_safe_tmp_list, acc_dang_tmp_list)))
                # logger.log('num  sa/med/da:%d %d %d' % (np_dict["num_safe"], np_dict["num_medium"], np_dict["num_dang"]))
                logger.log('num  sa/med/da:%d %d %d' % (np.sum(safe_mask), np.sum(medium_mask), np.sum(dang_mask)))
                logger.log('dacc sa/med/da:%s' % (style(h_deriv_acc_safe_list, h_deriv_acc_medium_list, h_deriv_acc_dang_list)))

                einfo = samples_data["env_infos"]
                self.log_t = time.time()
                if self.log_curr is not None:
                    self.log_curr.write("[%04d] %.4fs loss %.4f %.4f %.4f %.4f (%4d %4d %4d) acc %.4f %.4f %.4f %.4f %.4f "
                                        "out %.4f col %.4f trj %.4f err %.4f \n" % (
                        itr, self.log_t - self.log_prev_t,
                        total_loss_tmp_list[0], loss_crit_tmp_list[0], loss_grad_tmp_list[0], loss_reg_tmp_list[0],
                        np.sum(safe_mask), np.sum(medium_mask), np.sum(dang_mask),
                        acc_safe_tmp_list[0], acc_dang_tmp_list[0], h_deriv_acc_safe_list[0], h_deriv_acc_medium_list[0],
                        h_deriv_acc_dang_list[0], np.mean(einfo["out_of_lane"]), np.mean(einfo["is_colliding"]),
                        np.mean(np.sum(einfo["is_colliding"], axis=1) > 0.5), np.mean(einfo["rmse_pos"])))

                    self.log_curr.flush()

                self.log_prev_t = self.log_t

            if self.args.save_data:
                np.savez("%s/buffer_%d.npz" % (self.args.full_log_dir, itr), buffer=buffer)


        else:
            loss_before = self.optimizer.loss(all_input_values, extra_inputs=extra_input_values)
            mean_kl_before = self.optimizer.constraint_val(all_input_values, extra_inputs=extra_input_values)
            self.optimizer.optimize(all_input_values, extra_inputs=extra_input_values)
            mean_kl = self.optimizer.constraint_val(all_input_values, extra_inputs=extra_input_values)
            loss_after = self.optimizer.loss(all_input_values, extra_inputs=extra_input_values)
            logger.record_tabular('LossBefore', loss_before)
            logger.record_tabular('LossAfter', loss_after)
            logger.record_tabular('MeanKLBefore', mean_kl_before)
            logger.record_tabular('MeanKL', mean_kl)
            logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()



    #TODO(yue)
    def refine_policy(self, norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var, dbg_traj_i, dbg_step):
        def style(a, b=None):
            if b is None:
                return "%.4f -> %.4f (%.4f)" % (a[0], a[-1], np.mean(a))
            else:
                return "%.4f %.4f -> %.4f %.4f (%.4f %.4f)" % (
                    a[0], b[0], a[-1], b[-1], np.mean(a), np.mean(b)
                )

        if self.args.clip_affordance:
            raise NotImplementedError

        # TODO (compute normalized features)
        env_is_colliding = e_info["is_colliding"]
        env_out_of_lane = e_info["out_of_lane"]
        true_obs = norm_obs * np.sqrt(obs_var) + obs_mean
        if self.args.use_mono:
            true_actions = norm_actions * np.array([[4,]])
            # TODO (compute save/dang masks)
            safe_mask = utils_mono.get_safe_mask_mono(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                           env_out_of_lane, self.args.safe_dist_threshold,
                                           self.args.safe_dist_threshold_side, check_safe=True)
            dang_mask = utils_mono.get_safe_mask_mono(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                           env_out_of_lane, self.args.dang_dist_threshold,
                                           self.args.dang_dist_threshold_side, check_safe=False)
        elif self.args.use_ped:
            true_actions = norm_actions * np.array([[4, 4]])
            safe_mask = utils_ped.get_safe_mask_ped(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane, self.args.safe_dist_threshold,
                                               self.args.safe_dist_threshold_side, check_safe=True)
            dang_mask = utils_ped.get_safe_mask_ped(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane, self.args.dang_dist_threshold,
                                               self.args.dang_dist_threshold_side, check_safe=False)
        else:
            true_actions = norm_actions * np.array([[4, 0.15]])
            # TODO (compute save/dang masks)
            safe_mask = utils_ngsim.get_safe_mask(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                           env_out_of_lane, self.args.safe_dist_threshold, self.args.safe_dist_threshold_side, check_safe=True)
            dang_mask = utils_ngsim.get_safe_mask(self.args, true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                           env_out_of_lane, self.args.dang_dist_threshold, self.args.dang_dist_threshold_side, check_safe=False)
        medium_mask = np.logical_and(~safe_mask, ~dang_mask)
        safe_mask = safe_mask.astype(dtype=np.float32)
        dang_mask = dang_mask.astype(dtype=np.float32)
        medium_mask = medium_mask.astype(dtype=np.float32)

        #checkpoint debug
        np.set_printoptions(edgeitems=20, precision=5, linewidth=300)
        np.core.arrayprint._line_width = 360
        if self.args.debug_refine:
            print("safe_mask", safe_mask)  # TODO(debug)
            print("dang_mask", dang_mask)  # TODO(debug)
            print("medium_mask", medium_mask)  # TODO(debug)
        # print("true obs", true_obs)  # TODO(debug)
        # print("true actions", true_actions)  # TODO(debug)
        true_obs = np.expand_dims(true_obs, axis=1)
        true_actions = np.expand_dims(true_actions, axis=1)

        # print("true_action.shape",true_actions.shape)

        refine_input_values = (true_obs, true_actions, safe_mask, dang_mask, medium_mask)

        num_repeat = self.args.refine_n_iter
        loss_grad_tmp_list = []
        _ = tf.get_default_session().run([self.u_init], feed_dict={})

        u_res = 0.0 * true_actions
        # print()
        # print("traj-%d t=%d"%(dbg_traj_i, dbg_step), e_info["is_colliding"])
        # print(e_info["is_colliding"])

        # if dbg_traj_i>=1 or dbg_step>25:
        # if dbg_traj_i >= 1:
        # if dbg_traj_i>=1 or dbg_step>50:
        #     exit()

        for ti in range(num_repeat):
            # _ = tf.get_default_session().run([self.u_res], feed_dict={self.u_res: u_res})
            # if np.mean(safe_mask)==1.0:
            #     # print("continue")
            #     continue

            # u_res, np_dict, _ = tf.get_default_session().run([self.u_res, self.refine_dict, self.refine_op],
            #                                           feed_dict={_k: _v for _k, _v in
            #                                                      zip(self.refine_input_list, refine_input_values)})

            u_res, np_dict = tf.get_default_session().run([self.u_res, self.refine_dict],
                                                             feed_dict={_k: _v for _k, _v in
                                                                    zip(self.refine_input_list,
                                                                        refine_input_values)})

            case_i = 9
            # TODO(debug)
            # if dbg_step>25 and ti==0:
            #     print("traj-%d t=%d" % (dbg_traj_i, dbg_step), e_info["is_colliding"][case_i])
            #     dbg_s = np_dict["true_obs_var"][case_i, 0, :]
            #     for ni in range(8):
            #         print("observation-%d: %.4f %.4f %.4f %.4f"%(ni, dbg_s[0+ni*4], dbg_s[1+ni*4], dbg_s[2+ni*4], dbg_s[3+ni*4]))
            #     print("action: %.4f %.4f" % (np_dict["action"][case_i, 0], np_dict["action"][case_i, 1]))
            #     print("score: %.4f future:%.4f" % (np_dict["h_scores"][case_i], np_dict["h_scores_future"][case_i]))
            #     print("grad: [%.4f %.4f]" % (np_dict["gradients0"][case_i, 0, 0], np_dict["gradients0"][case_i, 0, 1]))
            #     print("u_res: [%.4f %.4f] -> [%.4f %.4f]" % (np_dict["u_res_prev"][case_i, 0, 0], np_dict["u_res_prev"][case_i, 0, 1],
            #                                                  u_res[case_i, 0, 0], u_res[case_i, 0, 1]))
            #     print("old====")
            #     mod_s = np_dict["old_state_tplus1"][case_i]
            #     for ni in range(2):
            #         print("old old obs-%d: %.4f %.4f %.4f %.4f" % (
            #         ni, mod_s[0 + ni * 4], mod_s[1 + ni * 4], mod_s[2 + ni * 4], mod_s[3 + ni * 4]))
            #     print("old fut: %.4f" % (np_dict["old_h_scores_future"][case_i]))
            #     print("--------")
            #
            #     print("mod====")
            #     mod_s = np_dict["mod_state_tplus1"][case_i]
            #     for ni in range(2):
            #         print("modified obs-%d: %.4f %.4f %.4f %.4f"%(ni, mod_s[0+ni*4], mod_s[1+ni*4], mod_s[2+ni*4], mod_s[3+ni*4]))
            #     print("mod fut: %.4f"%(np_dict["mod_h_scores_future"][case_i]))
            #     prev_fut_s = np_dict["h_scores_future"][case_i]

            # if dbg_step > 25:
            #     # np_dict, = tf.get_default_session().run([self.refine_dict], feed_dict={_k: _v for _k, _v in
            #     #                                                          zip(self.refine_input_list, refine_input_values)})
            #
            #     print("repeat-%d "
            #           "act: %.4f %.4f "
            #           "u: %.4f %.4f "
            #           "a+u: %.4f %.4f "
            #           "score: %.4f future: %.4f -> %.4f"
            #           "grad : %.4f %.4f"%(ti,
            #                               np_dict["action"][case_i, 0], np_dict["action"][case_i, 1],
            #                               np_dict["u_res_prev"][case_i, 0, 0], np_dict["u_res_prev"][case_i, 0, 1],
            #                               np_dict["action"][case_i, 0] + np_dict["u_res_prev"][case_i, 0, 0], np_dict["action"][case_i, 1] + np_dict["u_res_prev"][case_i, 0, 1],
            #                               np_dict["h_scores"][case_i], prev_fut_s, np_dict["h_scores_future"][case_i],
            #                               np_dict["gradients0"][case_i, 0, 0], np_dict["gradients0"][case_i, 0, 1]))

            # print("repeat %d"%ti, e_info["is_colliding"][case_i])
            # print(np_dict["gradients0"].shape, np_dict["safe_mask"].shape, np_dict["dang_mask"].shape, np_dict["medium_mask"].shape,
            #       np_dict["true_obs_var"].shape, np_dict["h_scores"].shape, np_dict["h_scores_future"].shape)
            # dbg_s = np_dict["true_obs_var"][:,0,:]
            # # for case_i in range(10):
            #
            # print("safe:%d  med:%d  dang:%d"%(np_dict["safe_mask"][case_i], np_dict["medium_mask"][case_i], np_dict["dang_mask"][case_i]))
            # for ni in range(8):
            #     print("observation-%d: %.4f %.4f %.4f %.4f"%(ni, dbg_s[case_i, 0+ni*4], dbg_s[case_i, 1+ni*4], dbg_s[case_i, 2+ni*4], dbg_s[case_i, 3+ni*4]))
            # print("action: %.4f %.4f"%(np_dict["action"][case_i, 0], np_dict["action"][case_i, 1]))
            # print("score: %.4f future:%.4f"%(np_dict["h_scores"][case_i], np_dict["h_scores_future"][case_i]))
            # print("grad: [%.4f %.4f]"%(np_dict["gradients0"][case_i, 0, 0], np_dict["gradients0"][case_i, 0, 1]))
            # print("u_res: [%.4f %.4f]"%(u_res[case_i, 0, 0], u_res[case_i, 0, 1]))
            # print()

            # exit()

            loss_grad_tmp_list.append(np_dict["loss_grad"])

        return u_res

    #TODO(yue)
    def refine_policy_qp(self, norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var, dbg_traj_i, dbg_step):
        if self.args.clip_affordance:
            raise NotImplementedError

        # TODO (compute normalized features)
        true_obs = norm_obs * np.sqrt(obs_var) + obs_mean
        if self.args.use_mono:
            true_actions = norm_actions * np.array([[4,]])
        elif self.args.use_ped:
            true_actions = norm_actions * np.array([[4, 4]])
        else:
            true_actions = norm_actions * np.array([[4, 0.15]])

        true_obs = np.expand_dims(true_obs, axis=1)
        true_actions = np.expand_dims(true_actions, axis=1)

        refine_input_values = (true_obs,)
        np_dict, = tf.get_default_session().run([self.refine_dict], feed_dict={_k: _v for _k, _v in
                                                                zip(self.refine_input_list, refine_input_values)})

        # min u_r^T H u_r + F u_r
        # s.t. dh/dx * (f(x) + g(x) (u+u_r)) >= - alpha * (h(x))
        # for fixed x, dh/dt is a 1*32 matrix f(x) is 32*1 matrix, and g(x) is 32*2 matrix, u is 2*1 (ped case)
        # standard form: min  1/2 * x^T P x + q^T x
        #                s.t. Gx <= h
        dhdx = np_dict["dhdx"][0]
        hx = np_dict["hx"]
        n_veh = hx.shape[0]

        # TODO naive 'for-loop' for each individual
        u_res_list = []
        P = matrix(np.diag([1.0, 1.0]), tc='d')  # quadratic term (regularization)
        q = matrix([0.0, 0.0], tc='d')  # linear term
        B = numpy.matlib.repmat([[0, 0], [0, 0], [-1, 0], [0, -1.]], self.args.num_neighbors, 1)
        alpha = self.args.qp_alpha
        feat = true_obs[:, 0, self.cbf_intervals]
        vxvy = feat.reshape((n_veh, 8, 4))[:, :, 2:]
        Ax = np.concatenate((vxvy, np.zeros_like(vxvy)), axis=-1).reshape((n_veh, 8*4))
        u = true_actions[:, 0, :]

        for i in range(n_veh):
            G = matrix(-dhdx[i:i+1] @ B, tc='d')
            h = matrix((alpha * hx[i:i+1] + dhdx[i:i+1] @ (Ax[i:i+1].T + B @ u[i:i+1].T)[:, 0]), tc='d')
            sol = solvers.qp(P, q, G, h)
            u_res_list.append(sol['x'].T)

        u_res = np.stack(u_res_list, axis=0)


        # # TODO parallel solution
        # P = matrix(np.diag([1.0 for _ in range(2*n_veh)]), tc='d')
        # q = matrix([0.0 for _ in range(2*n_veh)], tc='d')
        # _B = np.array([[0, 0], [0, 0], [-1, 0], [0, -1.]])
        # B = np.concatenate([_B for _ in range(self.args.num_neighbors)], axis=0)
        # dhdx_B = [dhdx[i:i+1] @ B for i in range(n_veh)]
        # dhdx_B = scipy.linalg.block_diag(*(dhdx_B))
        # alpha_hx = self.args.qp_alpha * hx.reshape((-1,1))
        # aff = true_obs[:, 0, self.cbf_intervals]
        # vxvy=aff.reshape((n_veh, 8, 4))[:, :, 2:]
        # Ax=np.concatenate((vxvy, vxvy*0), axis=-1).reshape((n_veh, 8*4))
        # dhdx_Ax = np.array([dhdx[i:i+1] @ Ax[i:i+1].T for i in range(n_veh)]).reshape((n_veh, 1))
        # dhdx_B_u = dhdx_B @ (true_actions.reshape((2*n_veh, 1)))
        #
        # G = matrix(-dhdx_B, tc='d')
        # h = matrix(alpha_hx + dhdx_Ax + dhdx_B_u, tc='d')
        #
        # sol = solvers.qp(P, q, G, h)
        # u_res = np.array(sol['x']).reshape((n_veh, 1, 2))

        return u_res

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )