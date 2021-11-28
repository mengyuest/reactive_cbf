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

import scipy
import numpy.matlib
import sandbox.rocky.tf.algos.utils_debug as utils_debug


class Object:
    pass

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
        if "args" in kwargs:
            self.args = kwargs["args"]  # TODO(yue)
        else:
            self.args=Object()
            self.args.joint_cbf=False
            self.args.use_policy_reference=False
            self.args.high_level=False
            self.args.full_log_dir=None
            self.args.refine_policy=False
            self.args.ref_policy=False
            self.args.quiet=False
            self.args.use_my_policy=False
            self.args.residual_u=False
            self.args.debug_render=False
            self.args.save_data=False
            self.args.cbf_iter_per_epoch=0
            self.args.accumulation_mode=False

        if "summary_writer" in kwargs:
            self.summary_writer = kwargs["summary_writer"]
        else:
            self.summary_writer = None
        self.kwargs = kwargs  # TODO(yue)
        if self.args.use_policy_reference: # TODO(yue)
            self.policy_as_ref = kwargs["policy_as_ref"]
        else:
            self.policy_as_ref = None
        if self.args.high_level:
            self.high_level_policy = kwargs["high_level_policy"]
        else:
            self.high_level_policy = None
        # TODO(yue)
        if self.args.full_log_dir is not None:
            if os.path.exists(ospj(self.args.full_log_dir, "train_log_curr.txt")):
                self.log_curr = None
                self.log_avg = None
            else:
                self.log_curr = open(ospj(self.args.full_log_dir, "train_log_curr.txt"), "a+")
                self.log_avg = open(ospj(self.args.full_log_dir, "train_log_avg.txt"), "a+")
        self.log_t = time.time()
        self.log_prev_t = time.time()

        #TODO(yue)
        self.grad_norm_rescale = 40.
        self.grad_norm_clip = 10000.

        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        if self.args.refine_policy:
            if self.args.qp_solve:
                self.cbf_intervals, self.cbf_ctrl_intervals, self.refine_input_list, self.refine_dict = \
                        utils_cbf.init_opt_refine_qp(self.args, self.policy.recurrent, self.env, self.kwargs["network"])
            else:
                self.cbf_intervals, self.cbf_ctrl_intervals, self.refine_input_list, \
                    self.u_res, self.u_init, self.refine_op, self.refine_dict = \
                    utils_cbf.init_opt_refine(self.args, self.policy.recurrent, self.env,
                                              self.kwargs["network"], self.grad_norm_rescale, self.grad_norm_clip)
        else:
            if self.args.ref_policy:
                self.init_opt_ref()
            else:
                self.init_opt_base()

        self.merged_summary_op = tf.summary.merge_all()



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

        input_list = [obs_var, action_var, advantage_var,] + state_info_vars_list + old_dist_info_vars_list

        if is_recurrent:
            input_list.append(valid_var)

        # TODO(yue)
        if self.args.joint_cbf:
            true_obs_var = self.env.observation_space.new_tensor_variable('true_obs', extra_dims=1 + is_recurrent)
            true_action_var = self.env.action_space.new_tensor_variable('true_action', extra_dims=1 + is_recurrent)

            safe_mask = tf.placeholder(tf.float32, [None], name="safe_mask")  # safe labels
            dang_mask = tf.placeholder(tf.float32, [None], name="dang_mask")  # dangerous labels
            medium_mask = tf.placeholder(tf.float32, [None], name="medium_mask")  # medium labels

            self.cbf_intervals, self.cbf_ctrl_intervals, cbf_loss, cbf_loss_full, cbf_network_params, self.joint_dict = \
                utils_cbf.setup_cbf(self.args, self.kwargs["network"], true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask)
            lr_cbf_loss = tf.reduce_sum(lr * cbf_loss_full * valid_var)
            surr_loss = surr_loss + lr_cbf_loss

            #TODO(yue) moved to extra list, because some of it is not sample-dividable
            extra_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]
            cbf_optimizer = tf.train.AdamOptimizer(self.args.cbf_learning_rate)
            self.cbf_train_op = self.get_train_op(surr_loss, cbf_network_params, cbf_optimizer, 'cbf/global_step')

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
        action_var = self.env.action_space.new_tensor_variable('action', extra_dims=1 + is_recurrent)

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

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
        elif self.args.high_level:
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
        if self.args.agent_cbf:
            if self.args.new_cbf_pol:
                safe_mask = tf.placeholder(tf.float32, [None, 2], name="safe_mask")  # safe labels
                dang_mask = tf.placeholder(tf.float32, [None, 2], name="dang_mask")  # dangerous labels
                medium_mask = tf.placeholder(tf.float32, [None, 2], name="medium_mask")  # medium labels
            else:
                safe_mask = tf.placeholder(tf.float32, [None, 6], name="safe_mask")  # safe labels
                dang_mask = tf.placeholder(tf.float32, [None, 6], name="dang_mask")  # dangerous labels
                medium_mask = tf.placeholder(tf.float32, [None, 6], name="medium_mask")  # medium labels
        else:
            safe_mask = tf.placeholder(tf.float32, [None], name="safe_mask")  # safe labels
            dang_mask = tf.placeholder(tf.float32, [None], name="dang_mask")  # dangerous labels
            medium_mask = tf.placeholder(tf.float32, [None], name="medium_mask")  # medium labels

        if self.args.use_my_policy:
            dist_info_vars = self.policy.dist_info_sym(true_obs_var)
        else:
            dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

        if self.args.high_level:
            self.high_level_policy.dist_info_sym(true_obs_var)


        self.cbf_intervals, self.cbf_ctrl_intervals, cbf_loss, cbf_loss_full, cbf_network_params, self.joint_dict = \
            utils_cbf.setup_cbf(self.args, self.kwargs["network"], true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask,
                           dist_info_vars, ref_action_var, ref_mean_var, ref_log_std_var)
        # self.joint_dict["loss_prev"] = cbf_loss
        #TODO(yue) moved to extra list, because some of it is not sample-dividable
        extra_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]
        self.input_list = input_list
        self.cbf_input_list = extra_list

        policy_params = self.policy.get_params(trainable=True)

        if self.args.alternative_update:
            assert self.args.joint_for_policy_only==False and self.args.joint_for_cbf_only==False
            optimizer_h = tf.train.AdamOptimizer(self.args.joint_learning_rate)
            self.train_op_h = self.get_train_op(cbf_loss, cbf_network_params, optimizer_h, 'joint/global_step_b')

            optimizer_a = tf.train.AdamOptimizer(self.args.joint_learning_rate)
            self.train_op_a = self.get_train_op(cbf_loss, policy_params, optimizer_a, 'joint/global_step_a')

        else:
            joint_optimizer = tf.train.AdamOptimizer(self.args.joint_learning_rate)
            if self.args.joint_for_policy_only:
                joint_params = policy_params
            elif self.args.joint_for_cbf_only:
                joint_params = cbf_network_params
            else:
                joint_params = cbf_network_params + policy_params
            self.joint_train_op = self.get_train_op(cbf_loss, joint_params, joint_optimizer, 'joint/global_step')
        # self.joint_dict["loss_after"] = cbf_loss
        return dict()

    def get_train_op(self, loss, params, optimizer, name):
        global_step = tf.Variable(0, name=name, trainable=False)
        gradients = tf.gradients(loss, params)
        clipped_grads = hgail.misc.tf_utils.clip_gradients(gradients, self.grad_norm_rescale, self.grad_norm_clip)
        train_op = optimizer.apply_gradients(list(zip(clipped_grads, params)), global_step=global_step)
        return train_op


    # def optimize_policy_debug(self, itr, samples_data):
    #
    #     agent_infos = samples_data["agent_infos"]
    #     env_infos = samples_data["env_infos"]
    #
    #     all_input_values = tuple(ext.extract(samples_data,
    #         "observations", "actions",))
    #
    #     state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
    #     all_input_values += tuple(state_info_list)
    #
    #     ref_action = agent_infos["lk_actions"] * agent_infos["choice"][:, :, 0:1]
    #     all_input_values += tuple([ref_action,])
    #
    #     all_input_values += (samples_data["valids"],)
    #
    #     obs_mean = samples_data["env_infos"]["obs_mean"]
    #     obs_var = samples_data["env_infos"]["obs_var"]
    #     env_is_colliding = env_infos["is_colliding"]
    #     env_out_of_lane = env_infos["out_of_lane"]
    #
    #     norm_obs = all_input_values[0]
    #     norm_actions = all_input_values[1]
    #     true_obs = utils_cbf.get_true_observations(self.args, norm_obs, obs_mean, obs_var)
    #     true_actions = utils_cbf.get_true_actions(norm_actions, self.args)
    #
    #     safe_mask, medium_mask, dang_mask = \
    #         utils_cbf.get_masks(self.args, true_obs, self.cbf_intervals, env_out_of_lane)
    #
    #     safe_mask = np.stack((safe_mask[:, 0], safe_mask[:, 3]), axis=-1)
    #     medium_mask = np.stack((medium_mask[:, 0], medium_mask[:, 3]), axis=-1)
    #     dang_mask = np.stack((dang_mask[:, 0], dang_mask[:, 3]), axis=-1)
    #
    #     extra_input_values = (true_obs, true_actions, safe_mask, dang_mask, medium_mask)
    #
    #     np_dict, _, sum_op = tf.get_default_session().run([self.joint_dict, self.joint_train_op, self.merged_summary_op],
    #                                                       feed_dict={_k: _v for _k, _v in
    #                                                                  zip(self.input_list + self.cbf_input_list,
    #                                                                      all_input_values + extra_input_values)})
    #
    #     logger.log('[%04d] L %.14f c %.14f g %.14f r %.14f [%3d %3d %3d] acc %.4f %.4f dacc %.4f %.4f %.4f' % (
    #         itr,
    #         np_dict["total_loss"],
    #         np_dict["loss_crit"],
    #         np_dict["loss_grad"],
    #         np_dict["loss_reg_policy"],
    #         np.sum(safe_mask),
    #         np.sum(medium_mask),
    #         np.sum(dang_mask),
    #         np_dict["acc_safe"],
    #         np_dict["acc_dang"],
    #         np_dict["h_deriv_acc_safe"],
    #         np_dict["h_deriv_acc_medium"],
    #         np_dict["h_deriv_acc_dang"]
    #     ), with_prefix=False, with_timestamp=True, end_word=" ", concised=True)
    #
    #     # TODO(debug)
    #     # print("loss_total_policy:")
    #     # for key in np_dict:
    #     #     if "debug_pol" in key:
    #     #         print("%s: %.14f"%(key, np.sum(np_dict[key])))
    #     #
    #     # print("loss_reg_policy:")
    #     # for key in np_dict:
    #     #     if "debug_reg_pol" in key:
    #     #         print("%s: %.14f"%(key, np.sum(np_dict[key])))
    #     #
    #     # print("loss_der_policy:")
    #     # for key in np_dict:
    #     #     if "debug_der_pol" in key:
    #     #         print("%s: %.14f"%(key, np.sum(np_dict[key])))
    #
    #     # TODO(debug)
    #     # if itr<=2:
    #     #     logger.log(" ".join(["%.14f"%xx for xx in list(np.sum(np.reshape(np_dict["debug_score"],(10,200,2)), axis=(1,2)))]),with_prefix=False, with_timestamp=True, end_word=" ", concised=True)
    #     #     logger.log("score next")
    #     #     logger.log(" ".join(
    #     #         ["%.14f" % xx for xx in list(np.sum(np.reshape(np_dict["debug_score_next"], (10, 200, 2)), axis=(1, 2)))]),
    #     #                with_prefix=False, with_timestamp=True, end_word=" ", concised=True)
    #
    #     # TODO(debug)
    #     # logger.log("A:%.14f\nB:%.14f\nC:%.14f\nAB:%.14f\nAC:%.14fBC:%.14f\nAB+C:%.14f\nABC:%.14f\nloss:%.14f\n"
    #     #            # "sum(s)+sum(d)=%.14f\n"
    #     #            # "sum(s+d)=%.14f\n"
    #     #            # "sum(ds)+sum(dm)+sum(dd)=%.14f\n"
    #     #            # "sum(ds+dm+dd)=%.14f\n"
    #     #            # "sum(s)+sum(d)+sum(ds)+sum(dm)+sum(dd)=%.14f\n"
    #     #            # "sum(s+d+ds+dm+dd)=%.14f\n"
    #     #            % (
    #     #     np_dict["A"], np_dict["B"], np_dict["C"], np_dict["AB"], np_dict["AC"], np_dict["BC"],
    #     #     np_dict["AB+C"], np_dict["ABC"], np_dict["total_loss"],
    #     #     # np_dict["sum(s)+sum(d)"],
    #     #     # np_dict["sum(s+d)"],
    #     #     # np_dict["sum(ds)+sum(dm)+sum(dd)"],
    #     #     # np_dict["sum(ds+dm+dd)"],
    #     #     # np_dict["sum(s)+sum(d)+sum(ds)+sum(dm)+sum(dd)"],
    #     #     # np_dict["sum(s+d+ds+dm+dd)"]
    #     #     # np_dict["loss_prev"], np_dict["loss_after"],
    #     #     # np_dict["loss*1"], np_dict["loss2"]
    #     # ), with_prefix=False, with_timestamp=False)
    #
    #     return {}

    # def optimize_policy_simple(self, itr, samples_data):
    #     obs = np.concatenate([path["observations"] for path in paths], axis=0)
    #     actions = np.concatenate([path["actions"] for path in paths], axis=0)
    #     agent_infos = {}
    #     env_infos = {}
    #
    #     for k in paths[0]["agent_infos"]:
    #         agent_infos[k] = np.concatenate([path["agent_infos"][k] for path in paths], axis=0)
    #     for k in paths[0]["env_infos"]:
    #         env_infos[k] = np.concatenate([path["env_infos"][k] for path in paths], axis=0)
    #
    #     env_is_colliding = env_infos["is_colliding"]
    #     env_out_of_lane = env_infos["out_of_lane"]
    #
    #     cbf_intervals = utils_cbf.get_ext_indices(args.cbf_intervals)
    #
    #     safe_mask, medium_mask, dang_mask = \
    #         utils_cbf.get_masks(args, obs, cbf_intervals, env_out_of_lane)
    #
    #     safe_mask = np.stack((safe_mask[:, 0], safe_mask[:, 3]), axis=-1)
    #     medium_mask = np.stack((medium_mask[:, 0], medium_mask[:, 3]), axis=-1)
    #     dang_mask = np.stack((dang_mask[:, 0], dang_mask[:, 3]), axis=-1)
    #
    #     state = obs[:, cbf_intervals]
    #     uref = agent_infos["ref_actions"]
    #
    #     np_dict, _, sum_op = tf.get_default_session().run(
    #         [tf_dict, train_op, summary_op],
    #         feed_dict={
    #             pl_dict["state"]: state,
    #             pl_dict["uref"]: uref,
    #             pl_dict["safe_mask"]: safe_mask,
    #             pl_dict["dang_mask"]: dang_mask,
    #             pl_dict["medium_mask"]: medium_mask,
    #         })
    #
    #
    #     return {}

    @overrides
    def optimize_policy(self, itr, samples_data):
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

            #TODO for reference_contrl
            if self.args.residual_u and self.args.reference_control:
                # print(all_input_values[1].shape, samples_data.keys(), samples_data["agent_infos"]["ref_actions"].shape)
                # for t in range(50):
                #     print(t, all_input_values[1][0, t,:],samples_data["agent_infos"]["ref_actions"][0,t,:])
                # exit()
                all_input_values = tuple([
                    all_input_values[0],
                    all_input_values[1] - samples_data["agent_infos"]["ref_actions"], all_input_values[2]])


        agent_infos = samples_data["agent_infos"]

        assert self.args.cbf_iter_per_epoch == 0
        assert not self.args.accumulation_mode

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
                        actions_from_ref, agent_infos_from_ref = self.policy_as_ref.get_actions(all_input_values[0][:, ti])
                        ref_action.append(actions_from_ref)
                        ref_mean.append(agent_infos_from_ref["mean"])
                        ref_log_std.append(agent_infos_from_ref["log_std"])

                    ref_action = np.stack(ref_action, axis=1)
                    ref_mean = np.stack(ref_mean, axis=1)
                    ref_log_std = np.stack(ref_log_std, axis=1)
                all_input_values += tuple([ref_action, ref_mean, ref_log_std])
            elif self.args.use_nominal_controller:
                assert any([self.args.use_ped, self.args.use_easy, self.args.use_pedcyc, self.args.use_round, self.args.use_high])
                ref_action = agent_infos["ref_actions"]
                all_input_values += tuple([ref_action,])
                if self.args.save_data:
                    buffer["ref_action"] = ref_action
            elif self.args.high_level:
                ref_action = agent_infos["lk_actions"] * agent_infos["choice"][:, :, 0:1]
                all_input_values += tuple([ref_action,])
            else:
                raise NotImplementedError

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
            true_obs = utils_cbf.get_true_observations(self.args, norm_obs, obs_mean, obs_var)
            true_actions = utils_cbf.get_true_actions(norm_actions, self.args)

            if self.args.save_data:
                buffer["true_obs"] = true_obs
            if self.args.local_data_path is not None:
                buffer = np.load(ospj(self.args.local_data_path, "buffer_%d.npz"%(itr+1)))["buffer"].item()
                true_obs = buffer["true_obs"]

            if self.args.clip_affordance:
                if any([self.args.use_mono, self.args.use_ped, self.args.use_easy, self.args.use_pedcyc, self.args.use_round, self.args.use_high]):
                    raise NotImplementedError
                else:
                    true_obs[:, :, self.cbf_intervals] = utils_cbf.post_process_affordance(true_obs[:, :, self.cbf_intervals])

            # TODO (compute save/medium/dang masks)
            safe_mask, medium_mask, dang_mask = \
                utils_cbf.get_masks(self.args, true_obs, self.cbf_intervals, env_out_of_lane)

            if self.args.new_cbf_pol:
                safe_mask = np.stack((safe_mask[:, 0], safe_mask[:, 3]), axis=-1)
                medium_mask = np.stack((medium_mask[:, 0], medium_mask[:, 3]), axis=-1)
                dang_mask = np.stack((dang_mask[:, 0], dang_mask[:, 3]), axis=-1)

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
            if self.args.quiet==False:
                logger.log("Joint Optimization Phase RAM: %.3f GB"%(utils_debug.get_memory()))
            if self.args.joint_iter_per_epoch==0:
                num_repeat=0
            else:
                num_repeat = self.args.joint_iter_per_epoch if itr < self.args.joint_decay_after else max(self.args.joint_iter_per_epoch // 10, 1)

            if num_repeat>0:
                for ti in range(num_repeat):
                    if self.args.alternative_update:
                        train_ops = [self.train_op_h if (itr//self.args.alternative_t)%2==0 else self.train_op_a]
                    else:
                        train_ops = [self.joint_train_op]

                    sess_t0 = time.time()
                    for train_op_item in train_ops:
                        np_dict, _, sum_op = tf.get_default_session().run([self.joint_dict, train_op_item, self.merged_summary_op],
                                                          feed_dict={_k: _v for _k, _v in
                                                                     zip(self.input_list + self.cbf_input_list,
                                                                         all_input_values + extra_input_values)})
                        if self.summary_writer is not None:
                            self.summary_writer.add_summary(sum_op, itr * num_repeat + ti)
                            self.summary_writer.flush()


                    if self.args.print_debug:
                        utils_debug.debug_in_optimize_policy(np_dict=np_dict, args=self.args)

                    sess_t = time.time()-sess_t0

                    total_loss_tmp_list.append(np_dict["total_loss"])
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

                var_sizes = [np.product(list(map(int, v.get_shape()))) * v.dtype.size
                             for v in tf.global_variables()]
                if self.args.quiet==False:
                    logger.log('Session Runtime: %.4f s\tRAM: %.4f MB'%(sess_t, sum(var_sizes) / (1024 ** 2)))
                    logger.log('CBF--loss:     %s' % (utils_debug.style(total_loss_tmp_list)))
                    logger.log('loss crit/grad:%s' % (utils_debug.style(loss_crit_tmp_list, loss_grad_tmp_list)))
                    logger.log('loss  reg:     %s' % (utils_debug.style(loss_reg_tmp_list)))
                    logger.log('loss safe/dang:%s' % (utils_debug.style(loss_safe_tmp_list, loss_dang_tmp_list)))
                    logger.log('acc  safe/dang:%s' % (utils_debug.style(acc_safe_tmp_list, acc_dang_tmp_list)))
                    logger.log('num  sa/med/da:%d %d %d' % (np.sum(safe_mask), np.sum(medium_mask), np.sum(dang_mask)))
                    logger.log('dacc sa/med/da:%s' % (utils_debug.style(h_deriv_acc_safe_list, h_deriv_acc_medium_list, h_deriv_acc_dang_list)))
                else:
                    logger.log('[%04d] L %.4f c %.4f g %.4f r %.4f [%3d %3d %3d] acc %.4f %.4f dacc %.4f %.4f %.4f'%(
                        itr,
                        total_loss_tmp_list[0],
                        loss_crit_tmp_list[0],
                        loss_grad_tmp_list[0],
                        loss_reg_tmp_list[0],
                        np.sum(safe_mask),
                        np.sum(medium_mask),
                        np.sum(dang_mask),
                        acc_safe_tmp_list[0],
                        acc_dang_tmp_list[0],
                        h_deriv_acc_safe_list[0],
                        h_deriv_acc_medium_list[0],
                        h_deriv_acc_dang_list[0]
                    ), with_prefix=False, with_timestamp=True, end_word=" ", concised=True)
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
        if self.args.clip_affordance:
            raise NotImplementedError

        # TODO (compute normalized features)
        env_out_of_lane = e_info["out_of_lane"]
        true_obs = utils_cbf.get_true_observations(self.args, norm_obs, obs_mean, obs_var)
        true_actions = utils_cbf.get_true_actions(norm_actions, self.args)

        # TODO (compute save/dang masks)
        safe_mask, medium_mask, dang_mask = \
            utils_cbf.get_masks(self.args, true_obs, self.cbf_intervals, env_out_of_lane)

        if self.args.new_cbf_pol:
            safe_mask = np.stack((safe_mask[:, 0], safe_mask[:, 3]), axis=-1)
            medium_mask = np.stack((medium_mask[:, 0], medium_mask[:, 3]), axis=-1)
            dang_mask = np.stack((dang_mask[:, 0], dang_mask[:, 3]), axis=-1)

        refine_input_values = (true_obs, true_actions, safe_mask, dang_mask, medium_mask)

        num_repeat = self.args.refine_n_iter
        loss_grad_tmp_list = []
        _ = tf.get_default_session().run([self.u_init], feed_dict={})

        u_res = 0.0 * true_actions

        for ti in range(num_repeat):
            u_res, np_dict = tf.get_default_session().run([self.u_res, self.refine_dict],
                                                             feed_dict={_k: _v for _k, _v in
                                                                    zip(self.refine_input_list, refine_input_values)})

            loss_grad_tmp_list.append(np_dict["loss_grad"])

        return u_res



    #TODO(yue)
    def refine_policy_qp(self, norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var, dbg_traj_i, dbg_step):
        if self.args.use_ped:
            u_res = self.refine_policy_qp_ped(norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var, dbg_traj_i, dbg_step)
        elif self.args.use_mono:
            raise NotImplementedError
        elif self.args.use_easy:
            raise NotImplementedError
        elif self.args.use_pedcyc:
            raise NotImplementedError
        elif self.args.use_round:
            raise NotImplementedError
        elif self.args.use_high:
            raise NotImplementedError
        else:
            u_res = self.refine_policy_qp_ngsim(norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var, dbg_traj_i, dbg_step)
        return u_res

    def refine_policy_qp_ped(self, norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var, dbg_traj_i, dbg_step):
        if self.args.clip_affordance:
            raise NotImplementedError

        from cvxopt import matrix
        from cvxopt import solvers
        solvers.options['show_progress'] = False

        true_obs = utils_cbf.get_true_observations(self.args, norm_obs, obs_mean, obs_var)
        true_actions = utils_cbf.get_true_actions(norm_actions, self.args)

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

        return u_res

    #TODO(yue)
    def refine_policy_qp_ngsim(self, norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var, dbg_traj_i, dbg_step):
        if self.args.clip_affordance:
            raise NotImplementedError

        from cvxopt import matrix
        from cvxopt import solvers
        solvers.options['show_progress'] = False

        true_obs = utils_cbf.get_true_observations(self.args, norm_obs, obs_mean, obs_var)
        true_actions = utils_cbf.get_true_actions(norm_actions, self.args)

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
        alpha = self.args.qp_alpha
        u_res_list = []
        P = matrix(np.diag([self.args.qp_accel_weight, self.args.qp_omega_weight]), tc='d')  # quadratic term (regularization)
        q = matrix([0.0, 0.0], tc='d')  # linear term

        feat = true_obs[:, 0, self.cbf_intervals]
        ego_feat = feat[:, :11]
        nei_feat = feat[:, 11:].reshape((-1, 6, 9))
        u = true_actions[:, 0, :]

        for i in range(n_veh):

            ego_fx = np.zeros((11, 1))
            ego_g = np.zeros((11, 2))
            ego_fx[1, 0] = -ego_feat[i, 6] * np.sin(ego_feat[i, 5])
            ego_fx[2, 0] = ego_feat[i, 6] * np.sin(ego_feat[i, 5])
            ego_fx[3, 0] = -ego_feat[i, 6] * np.sin(ego_feat[i, 5])
            ego_fx[4, 0] = ego_feat[i, 6] * np.sin(ego_feat[i, 5])
            ego_g[6, 0] = 1
            ego_g[5, 1] = 1

            nei_fx=np.zeros((6,9))
            nei_g = np.zeros((6,9,2))
            for j in range(6):
                if nei_feat[i, j, 0] > 0.5:
                    nei_fx[j, 1] = - nei_feat[i, j, 4] * np.sin(nei_feat[i, j, 3])
                    nei_fx[j, 2] = nei_feat[i, j, 4] * np.cos(nei_feat[i, j, 3]) - ego_feat[i, 6]
                    nei_fx[j, 3] = nei_feat[i, j, 6]
                    nei_fx[j, 4] = nei_feat[i, j, 5]
                    nei_g[j, 3, 1] = -1
            mat_fx = np.concatenate((ego_fx, np.reshape(nei_fx, (54, 1))))
            mat_g = np.concatenate((ego_g, np.reshape(nei_g, (54, 2))), axis=0)

            G = matrix(-dhdx[i:i+1] @ mat_g, tc='d')
            h = matrix((alpha * hx[i:i+1] + dhdx[i:i+1] @ (mat_fx+ mat_g @ u[i:i+1].T)[:, 0]), tc='d')
            sol = solvers.qp(P, q, G, h)
            u_res_list.append(sol['x'].T)

        u_res = np.stack(u_res_list, axis=0)

        return u_res

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )