from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf

import numpy as np #TODO(yue)
import hgail

import geo_check

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
        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        if self.args.refine_policy:
            self.init_opt_refine()
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

            cbf_loss, cbf_loss_full, cbf_network_params, self.cbf_dict, self.joint_dict = \
                self.setup_cbf(true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask)
            self.tf_dict["lr_cbf_loss"] = tf.reduce_sum(lr * cbf_loss_full * valid_var)
            surr_loss = surr_loss + self.tf_dict["lr_cbf_loss"]
            #TODO(yue) moved to extra list, because some of it is not sample-dividable
            extra_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]
            self.cbf_optimizer = tf.train.RMSPropOptimizer(self.args.cbf_learning_rate)
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
            self.save_mask_data = []
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

        cbf_loss, cbf_loss_full, cbf_network_params, self.cbf_dict, self.joint_dict = \
            self.setup_cbf(true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask,
                           dist_info_vars, ref_action_var, ref_mean_var, ref_log_std_var)
        self.tf_dict["lr_cbf_loss"] = tf.reduce_sum(cbf_loss_full * valid_var)

        #TODO(yue) moved to extra list, because some of it is not sample-dividable
        extra_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]
        self.cbf_optimizer = tf.train.RMSPropOptimizer(self.args.cbf_learning_rate)
        cbf_gradients = tf.gradients(self.tf_dict["lr_cbf_loss"], cbf_network_params)
        cbf_clipped_gradients = hgail.misc.tf_utils.clip_gradients(
            cbf_gradients, self.grad_norm_rescale, self.grad_norm_clip)
        self.cbf_global_step = tf.Variable(0, name='cbf/global_step', trainable=False)
        self.cbf_train_op = self.cbf_optimizer.apply_gradients([(g, v)
                          for (g, v) in zip(cbf_clipped_gradients, cbf_network_params)], global_step=self.cbf_global_step)
        self.input_list = input_list
        self.cbf_input_list = extra_list

        self.joint_optimizer = tf.train.RMSPropOptimizer(self.args.joint_learning_rate)
        if self.args.joint_for_policy_only:
            joint_network_params = self.policy.get_params(trainable=True)
        else:
            joint_network_params = cbf_network_params + self.policy.get_params(trainable=True)
        joint_gradients = tf.gradients(self.tf_dict["lr_cbf_loss"], joint_network_params)
        joint_clipped_gradients = hgail.misc.tf_utils.clip_gradients(
            joint_gradients, self.grad_norm_rescale, self.grad_norm_clip)
        self.joint_global_step = tf.Variable(0, name='joint/global_step', trainable=False)
        self.joint_train_op = self.joint_optimizer.apply_gradients([(g, v)
                                                                for (g, v) in
                                                                zip(joint_clipped_gradients, joint_network_params)],
                                                               global_step=self.joint_global_step)
        #TODO(yue) files to save
        self.save_mask_data = []
        self.save_h_deriv_data=[]

        return dict()

    def post_process_affordance(self, true_obs):
        aff = true_obs[:, :, self.cbf_intervals]
        _n, _t, _k = aff.shape
        # curve too large, clip to value
        aff[:, :, 0] = np.clip(aff[:, :, 0], -1, 1)

        # lld,rld,lrd,rrd too large, clip
        aff[:, :, 1:3] = np.clip(aff[:, :, 1:3], -5, 5)
        aff[:, :, 3:5] = np.clip(aff[:, :, 3:5], -30, 30)

        # as long as neighbor invalid, zero them
        aff_nei_flat = aff[:, :, 11:].reshape((_n*_t*6, 9))
        aff_nei_flat = (aff_nei_flat[:, 0:1] > 0.5).astype(np.float32) * aff_nei_flat

        aff[:, :, 11:] = aff_nei_flat.reshape((_n, _t, 6 * 9))

        true_obs[:, :, self.cbf_intervals] = aff
        return true_obs

    def normalize_affordance(self, state):
        info_norm = [1.0, 5.0, 5.0, 30.0, 30.0]
        ego_norm = [1.0, 30.0, 4.0, 0.15, 10.0, 3.0]
        nei_norm = [1.0, 15.0, 30.0, 1.0, 30.0, 4.0, 0.15, 10.0, 3.0] * 6
        self.normalizer = tf.constant([
            info_norm + ego_norm + nei_norm
        ])
        return state / self.normalizer

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

            if self.args.clip_affordance:
                if self.args.use_mono:
                    raise NotImplementedError
                else:
                    true_obs = self.post_process_affordance(true_obs)

            # TODO (compute save/dang masks)
            if self.args.use_mono:
                true_actions = norm_actions * np.array([[4,]])
                safe_mask = self.get_safe_mask_mono(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane,
                                               self.args.safe_dist_threshold, self.args.safe_dist_threshold_side,
                                               check_safe=True)
                dang_mask = self.get_safe_mask_mono(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                               env_out_of_lane,
                                               self.args.dang_dist_threshold, self.args.dang_dist_threshold_side,
                                               check_safe=False)
            else:
                true_actions = norm_actions * np.array([[4, 0.15]])
                safe_mask = self.get_safe_mask(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals], env_out_of_lane,
                                               self.args.safe_dist_threshold, self.args.safe_dist_threshold_side, check_safe=True)
                dang_mask = self.get_safe_mask(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals], env_out_of_lane,
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
            save_h_deriv_list=[]

            if num_repeat>0:
                for ti in range(num_repeat):
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
                    save_h_deriv_list.append(np_dict["save_h_deriv_data"])

                logger.log('CBF--loss:     %s' % (style(total_loss_tmp_list, lr_cbf_loss_tmp_list)))
                logger.log('loss crit/grad:%s' % (style(loss_crit_tmp_list, loss_grad_tmp_list)))
                logger.log('loss  reg:     %.4f' % (np_dict["loss_reg_policy"]))
                logger.log('loss safe/dang:%s' % (style(loss_safe_tmp_list, loss_dang_tmp_list)))
                logger.log('acc  safe/dang:%s' % (style(acc_safe_tmp_list, acc_dang_tmp_list)))
                logger.log('num  sa/med/da:%d %d %d' % (np_dict["num_safe"], np_dict["num_medium"], np_dict["num_dang"]))
                logger.log('dacc sa/med/da:%s' % (style(h_deriv_acc_safe_list, h_deriv_acc_medium_list, h_deriv_acc_dang_list)))

                #TODO(yue) save to files
                np.savez("%s/h_%d.npz"%(self.args.full_log_dir, itr),
                         safe_mask=safe_mask, dang_mask=dang_mask, medium_mask=medium_mask, h_deriv=np.stack(save_h_deriv_list))

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
                    np_dict, _ = tf.get_default_session().run([self.joint_dict, self.joint_train_op],
                                                          feed_dict={_k: _v for _k, _v in
                                                                     zip(self.input_list + self.cbf_input_list,
                                                                         all_input_values + extra_input_values)})
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
                logger.log('num  sa/med/da:%d %d %d' % (np_dict["num_safe"], np_dict["num_medium"], np_dict["num_dang"]))
                logger.log('dacc sa/med/da:%s' % (style(h_deriv_acc_safe_list, h_deriv_acc_medium_list, h_deriv_acc_dang_list)))

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
    def init_opt_refine(self):
        is_recurrent = int(self.policy.recurrent)
        true_obs_var = self.env.observation_space.new_tensor_variable('true_obs', extra_dims=1 + is_recurrent)
        true_action_var = self.env.action_space.new_tensor_variable('true_action', extra_dims=1 + is_recurrent)
        safe_mask = tf.placeholder(tf.float32, [None], name="safe_mask")  # safe labels
        dang_mask = tf.placeholder(tf.float32, [None], name="dang_mask")  # dangerous labels
        medium_mask = tf.placeholder(tf.float32, [None], name="medium_mask")  # medium labels

        self.refine_input_list = [true_obs_var, true_action_var, safe_mask, dang_mask, medium_mask]

        if self.args.use_mono:
            u_res = tf.Variable(tf.zeros([self.args.n_envs, 1, 1]), name='u_res')
        else:
            u_res = tf.Variable(tf.zeros([self.args.n_envs, 1, 2]), name='u_res')
        self.u_res = u_res
        self.u_init = tf.assign(u_res, tf.zeros_like(u_res))

        cbf_loss, cbf_loss_full, _, self.refine_dict, _ = \
            self.setup_cbf(true_obs_var, true_action_var + u_res, safe_mask, dang_mask, medium_mask)

        refine_optimizer = tf.train.RMSPropOptimizer(self.args.refine_learning_rate)
        gradients = tf.gradients(self.refine_dict["loss_grad"], u_res)
        clipped_gradients = hgail.misc.tf_utils.clip_gradients(gradients, self.grad_norm_rescale, self.grad_norm_clip)
        refine_global_step = tf.Variable(0, name='cbf/refine_global_step', trainable=False)

        # clipped_gradients = tf.Print(clipped_gradients, [clipped_gradients[0].shape], "clipped_gradients shape:", summarize=-1)
        self.refine_op = refine_optimizer.apply_gradients([(clipped_gradients[0], u_res)], global_step=refine_global_step)

    #TODO(yue)
    def refine_policy(self, norm_obs, e_info, norm_actions, a_info, obs_mean, obs_var):
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
            safe_mask = self.get_safe_mask_mono(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                           env_out_of_lane, self.args.safe_dist_threshold,
                                           self.args.safe_dist_threshold_side, check_safe=True)
            dang_mask = self.get_safe_mask_mono(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                           env_out_of_lane, self.args.dang_dist_threshold,
                                           self.args.dang_dist_threshold_side, check_safe=False)
        else:
            true_actions = norm_actions * np.array([[4, 0.15]])
            # TODO (compute save/dang masks)
            safe_mask = self.get_safe_mask(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
                                           env_out_of_lane, self.args.safe_dist_threshold, self.args.safe_dist_threshold_side, check_safe=True)
            dang_mask = self.get_safe_mask(true_obs.reshape((-1, true_obs.shape[-1]))[:, self.cbf_intervals],
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

        sel_idx = 4
        for ti in range(num_repeat):
            u_res, np_dict, _ = tf.get_default_session().run([self.u_res, self.refine_dict, self.refine_op],
                                                      feed_dict={_k: _v for _k, _v in
                                                                 zip(self.refine_input_list, refine_input_values)})


            loss_grad_tmp_list.append(np_dict["loss_grad"])

            if self.args.debug_refine:
                if ti==0:
            #         print("h_scores all", np_dict["h_scores"])
            #         print("h_scores_future", np_dict["h_scores_future"])
            #         print("condition:", (np_dict["h_scores_future"] - 0.99 * np_dict["h_scores"]))
                    print("h_scores", np_dict["h_scores"][sel_idx])

            # print("state_tplus1", np_dict["state_tplus1"])
            if self.args.debug_refine:
                print("h_scores_future", np_dict["h_scores_future"][sel_idx])
                #
                # # violation
                print("condition:", (np_dict["h_scores_future"] - 0.99 * np_dict["h_scores"])[sel_idx])
                # # print("gt loss:", np_dict["deriv_total"])
                # print("loss check:",
                #       np.sum(np.maximum(self.args.grad_medium_thres - np_dict["h_scores_future"] + 0.99 * np_dict["h_scores"],
                #                  0) * medium_mask / (1e-12 + np.sum(medium_mask))) * self.args.medium_deriv_loss_weight
                #
                #       +
                #
                #       np.sum(np.maximum(self.args.grad_safe_thres - np_dict["h_scores_future"] + 0.99 * np_dict["h_scores"],
                #                  0) * safe_mask / (1e-12 + np.sum(safe_mask))) * self.args.safe_deriv_loss_weight
                #
                #       +
                #
                #       np.sum(np.maximum(self.args.grad_dang_thres - np_dict["h_scores_future"] + 0.99 * np_dict["h_scores"],
                #                  0) * dang_mask / (1e-12 + np.sum(dang_mask))) * self.args.dang_deriv_loss_weight
                #
                #       )
                do_nothing=-1

            # state vectors (here just for the second last vehicle)
            # print(true_obs.shape)
            if self.args.debug_refine:
                np.set_printoptions(suppress=True)
            # print(true_obs[0,0,self.cbf_intervals])
            # exit()
            # print(np_dict["state_tplus1"].shape)
            # sel_nei_id=-1
            # print("==========dynamics=========")
            # for tt in range(4):
            #     print("tt=%d"%tt)
            #     print("accel:%.4f " % (np_dict["dyna_accel_%d" % tt][sel_idx]))
            #     print("omega:%.4f " % (np_dict["dyna_omega_%d" % tt][sel_idx]))
            #     print("V:%.4f " % (np_dict["dyna_v_%d" % tt][sel_idx]))
            #     print("Th:%.4f " % (np_dict["dyna_theta_%d" % tt][sel_idx]))
            #     print("dx:%.4f " % (np_dict["dyna_dx_%d" % tt][sel_idx]))
            #     print("ds:%.4f " % (np_dict["dyna_ds_%d" % tt][sel_idx]))
            #     print("newV:%.4f " % (np_dict["dyna_new_v_%d" % tt][sel_idx]))
            #     print("newTh:%.4f " % (np_dict["dyna_new_theta_%d" % tt][sel_idx]))
            #     print("neiid:%s " % (np_dict["dyna_nei_ind_%d" % tt].reshape((10,6))[sel_idx][sel_nei_id]))
            #     print("nei_x:%.4f " % (np_dict["dyna_nei_x_%d" % tt].reshape((10,6))[sel_idx][sel_nei_id]))
            #     print("nei_y:%.4f " % (np_dict["dyna_nei_y_%d" % tt].reshape((10,6))[sel_idx][sel_nei_id]))
            #     print("new_nei_x:%.4f " % (np_dict["dyna_new_nei_x_%d" % tt].reshape((10,6))[sel_idx][sel_nei_id]))
            #     print("new_nei_y:%.4f " % (np_dict["dyna_new_nei_y_%d" % tt].reshape((10,6))[sel_idx][sel_nei_id]))
            #     print("new_nei_x_new:%.4f " % (np_dict["dyna_new_nei_x_new_%d" % tt].reshape((10,6))[sel_idx][sel_nei_id]))
            #     print("new_nei_y_new:%.4f " % (np_dict["dyna_new_nei_y_new_%d" % tt].reshape((10,6))[sel_idx][sel_nei_id]))

            # to=true_obs[sel_idx, 0, self.cbf_intervals]
            # ne=np_dict["state_tplus1"][sel_idx]
            # if self.args.debug_refine:
            #     print("lld:%.4f %.4f\trld:%.4f %.4f\tlrd:%.4f %.4f\trrd:%.4f %.4f"%(to[1], ne[1],
            #                                                                         to[2], ne[2],
            #                                                                         to[3], ne[3],
            #                                                                         to[4], ne[4]))
            #     print("theta: %.4f %.4f\tv: %.4f %.4f\tL:%.4f W:%.4f"%(to[5], ne[5],
            #                                                            to[6], ne[6],
            #                                                            to[9],
            #                                                            to[10]))
            #     for nei_i in range(6):
            #         n_off=11+nei_i*9
            #         print("nei-%d %.4f x:%.4f %.4f\ty:%.4f %.4f\tdist:%.4f %.4f"%(nei_i, to[n_off+0],
            #                                                       to[n_off+1], ne[n_off+1],
            #                                                       to[n_off+2], ne[n_off+2],
            #                                                       (to[n_off+1]**2+to[n_off+2]**2)**0.5,
            #                                                       (ne[n_off + 1] ** 2 + ne[n_off + 2] ** 2) ** 0.5,
            #                                                                       ))
            #
            #     print("u_res", u_res.squeeze().T[:,sel_idx])
            #     print()



        # logger.log('loss grad:%s' % (style(loss_grad_tmp_list)))
        # logger.log('num  safe/dang:%d %d' % (np_dict["num_safe"], np_dict["num_dang"]))
        # logger.log('loss  safe/dang:%.4f %.4f' % (np_dict["loss_safe"], np_dict["loss_dang"]))
        # logger.log('acc  safe/dang:%.4f %.4f' % (np_dict["acc_safe"], np_dict["acc_dang"]))






        # import sys
        # np.set_printoptions(threshold=sys.maxsize, precision=8)

        # print("d_fut_score", np_dict["fut_score_gradients"])
        # print("d_s_tp1", np_dict["state_tplus1_gradients"])
        # print("d_s_stp1", np_dict["score_state_gradients"])
        # print("var_list", self.network.var_list)
        # print("last-1", np_dict["last1_gradient"][0])
        # print("val -1", self.network.get_param_values()[-1])
        # print("last-2", np_dict["last2_gradient"][0][:,0])
        # print("val -2", self.network.get_param_values()[-2][:,0])
        # print("last-3", np_dict["last3_gradient"][0])
        # print("last-4", np_dict["last4_gradient"][0][:10,:3])
        # print("last-5", np_dict["last5_gradient"][0])
        # print("last-6", np_dict["last6_gradient"][0][:10,:3])
        # print("last-7", np_dict["last7_gradient"][0])
        # print("last-8", np_dict["last8_gradient"][0][:10,:3])

        # for iii in range(len(self.network.debug)):
        #     print("debug%d"%(iii), np_dict["debug_%d"%(iii)][0])


        # print("params", self.network.get_param_values())
        # print("d_policy", np_dict["policy_output_gradients"])
        # print("d_u_res", np_dict["u_res_gradients"])
        #[<tf.Variable 'cbfer/cbfer/hidden/fully_connected/weights:0' shape=(65, 128) dtype=float32_ref>,
        # <tf.Variable 'cbfer/cbfer/hidden/fully_connected/biases:0' shape=(128,) dtype=float32_ref>,
        # <tf.Variable 'cbfer/cbfer/hidden/fully_connected_1/weights:0' shape=(128, 128) dtype=float32_ref>,
        # <tf.Variable 'cbfer/cbfer/hidden/fully_connected_1/biases:0' shape=(128,) dtype=float32_ref>,
        # <tf.Variable 'cbfer/cbfer/hidden/fully_connected_2/weights:0' shape=(128, 64) dtype=float32_ref>,
        # <tf.Variable 'cbfer/cbfer/hidden/fully_connected_2/biases:0' shape=(64,) dtype=float32_ref>,
        # <tf.Variable 'cbfer/cbfer/scores/fully_connected/weights:0' shape=(64, 1) dtype=float32_ref>,
        # <tf.Variable 'cbfer/cbfer/scores/fully_connected/biases:0' shape=(1,) dtype=float32_ref>]


        return u_res

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )


    def setup_cbf_indices_and_shapes(self):
        args = self.args
        self.cbf_intervals = get_ext_indices(args.cbf_intervals)
        if any([args.lane_control, args.multilane_control, args.naive_control]) and args.cbf_ctrl_intervals != "":
            print("Using Primal Control and PolicyNet as Rectification!")
            self.cbf_ctrl_intervals = get_ext_indices(args.cbf_ctrl_intervals)
        if args.attractive:
            self.n_ego_feat = 8
            self.n_nei_feat = 9
            if len(self.cbf_intervals)>self.n_ego_feat+self.n_nei_feat*6:
                self.n_ego_feat = 11
        elif args.use_mono:
            self.n_ego_feat = 1
            self.n_nei_feat = 2  # 1+2+2=5
        else:
            raise NotImplementedError

        self.feat_x_def = tf.constant([0.0, -15.0, 15.0, 0.0, -15.0, 15.0])
        self.feat_y_def = tf.constant([30.0, 30.0, 30.0, -30.0, -30.0, -30.0])



    def setup_cbf(self, obs, _action, safe_mask, dang_mask, medium_mask, dist_info_vars=None,
                  ref_action_var=None, ref_mean_var=None, ref_log_std_var=None):
        self.reduced = False

        self.tf_dict = {}
        joint_dict = {}

        self.network = self.kwargs["network"]

        self.grad_norm_rescale = 40.
        self.grad_norm_clip = 10000.
        self.setup_cbf_indices_and_shapes()
        n_dim = tf.cast(tf.shape(obs)[0], dtype=tf.float32)
        t_dim = tf.cast(tf.shape(obs)[1], dtype=tf.float32)
        obs = tf.reshape(obs, [-1, obs.shape[2]])


        if self.args.ref_policy and self.args.refine_policy==False:
            if self.args.deterministic_policy_for_cbf:
                action = dist_info_vars["mean"]
            else:
                rnd_sym = tf.random.normal(shape=tf.shape(dist_info_vars["mean"]))
                action = rnd_sym * dist_info_vars["log_std"] + dist_info_vars["mean"]
            action = tf.reshape(action, [-1, tf.shape(action)[2]])

            if self.args.use_mono:
                action = action * tf.constant([[4.0]])
            else:
                action = action * tf.constant([[4.0, 0.15]])
        else:
            action = _action
            action = tf.reshape(action, [-1, action.shape[2]])

        # self.h_func_input = tf.gather(obs, self.cbf_intervals, axis=1)  # needs un-normalization
        self.state_input = tf.gather(obs, self.cbf_intervals, axis=1)

        ### 2. FIND THE CONTROLLER OUTPUT
        if self.args.cbf_ctrl_intervals == "":
            primal_control = action * 0.0
        else:
            primal_control = tf.gather(obs, self.cbf_ctrl_intervals, axis=1)

        ### 3. DEFINE MONITORING VALUES
        self.loss_safe_list = []
        self.loss_dang_list = []
        self.acc_safe_list = []
        self.acc_dang_list = []
        self.num_safe_list = []
        self.num_dang_list = []
        # self.input_list = []
        self.mean_list = []
        self.var_list = []

        def dbg(x):
            return tf.strings.format('{}', x, summarize=-1)

        ### 4. DEFINE COMPUTATION GRAPH
        if self.args.normalize_affordance:
            if self.args.use_mono:
                raise NotImplementedError
            h_scores = tf.squeeze(self.network(self.normalize_affordance(self.state_input)))
        else:
            h_scores = tf.squeeze(self.network(self.state_input))
        self.tf_dict["state_input"] = self.state_input
        self.tf_dict["action"] = action
        self.tf_dict["h_scores"] = h_scores

        num_safe = tf.reduce_sum(safe_mask)
        num_dang = tf.reduce_sum(dang_mask)
        num_medium = tf.reduce_sum(medium_mask)

        loss_safe_full = tf.math.maximum(-h_scores + self.args.h_safe_thres, 0) * safe_mask / (1e-5 + num_safe)
        loss_safe_full = tf.reshape(loss_safe_full, [n_dim, t_dim])

        loss_dang_full = tf.math.maximum(h_scores + self.args.h_dang_thres, 0) * dang_mask / (1e-5 + num_dang)
        loss_dang_full = tf.reshape(loss_dang_full, [n_dim, t_dim])

        loss_safe = tf.reduce_sum(loss_safe_full)
        loss_dang = tf.reduce_sum(loss_dang_full)

        acc_dang = tf.reduce_sum(tf.cast(tf.less_equal(h_scores, 0), tf.float32) * dang_mask) / (
                1e-12 + num_dang)
        acc_safe = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores, 0), tf.float32) * safe_mask) / (
                1e-12 + num_safe)

        acc_dang = tf.cond(tf.greater(num_dang, 0), lambda: acc_dang, lambda: -tf.constant(1.0))
        acc_safe = tf.cond(tf.greater(num_safe, 0), lambda: acc_safe, lambda: -tf.constant(1.0))

        if self.args.use_mono:
            state_tplus1 = self.dynamics_mono(self.state_input, action, primal_control)
        else:
            state_tplus1 = self.dynamics_attr(self.state_input, action, primal_control)
        self.tf_dict["state_tplus1"] = state_tplus1

        if self.args.normalize_affordance:
            if self.args.use_mono:
                raise NotImplementedError
            h_scores_future = tf.squeeze(self.network(self.normalize_affordance(state_tplus1)))
        else:
            h_scores_future = tf.squeeze(self.network(state_tplus1))
        self.tf_dict["h_scores_future"] = h_scores_future

        loss_safe_deriv_full = tf.math.maximum(self.args.grad_safe_thres - h_scores_future + 0.99 * h_scores, 0) * safe_mask / (1e-12 + num_safe)
        loss_safe_deriv_full = tf.reshape(loss_safe_deriv_full, [n_dim, t_dim])
        loss_safe_deriv = tf.reduce_sum(loss_safe_deriv_full)

        loss_dang_deriv_full = tf.math.maximum(self.args.grad_dang_thres - h_scores_future + 0.99 * h_scores, 0) * dang_mask / (1e-12 + num_dang)
        loss_dang_deriv_full = tf.reshape(loss_dang_deriv_full, [n_dim, t_dim])
        loss_dang_deriv = tf.reduce_sum(loss_dang_deriv_full)

        loss_medium_deriv_full = tf.math.maximum(self.args.grad_medium_thres - h_scores_future + 0.99 * h_scores, 0) * medium_mask / (1e-12 + num_medium)
        loss_medium_deriv_full = tf.reshape(loss_medium_deriv_full, [n_dim, t_dim])
        loss_medium_deriv = tf.reduce_sum(loss_medium_deriv_full)

        self.tf_dict["save_h_deriv_data"] = h_scores_future - 0.99 * h_scores
        self.tf_dict["h_deriv_acc_safe"] = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                                                 tf.float32) * safe_mask) / (1e-12 + num_safe)
        joint_dict["h_deriv_acc_safe"] = self.tf_dict["h_deriv_acc_safe"]
        self.tf_dict["h_deriv_acc_dang"] = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                                                 tf.float32) * dang_mask) / (1e-12 + num_dang)
        joint_dict["h_deriv_acc_dang"] = self.tf_dict["h_deriv_acc_dang"]
        self.tf_dict["h_deriv_acc_medium"] = tf.reduce_sum(tf.cast(tf.greater_equal(h_scores_future - 0.99 * h_scores, 0),
                                                                 tf.float32) * medium_mask) / (1e-12 + num_medium)
        joint_dict["h_deriv_acc_medium"] = self.tf_dict["h_deriv_acc_medium"]

        self.tf_dict["h_deriv_acc_safe"] = tf.cond(tf.greater(num_safe, 0), lambda: self.tf_dict["h_deriv_acc_safe"], lambda: -tf.constant(1.0))
        joint_dict["h_deriv_acc_safe"] = self.tf_dict["h_deriv_acc_safe"]
        self.tf_dict["h_deriv_acc_dang"] = tf.cond(tf.greater(num_dang, 0), lambda: self.tf_dict["h_deriv_acc_dang"], lambda: -tf.constant(1.0))
        joint_dict["h_deriv_acc_dang"] = self.tf_dict["h_deriv_acc_dang"]
        self.tf_dict["h_deriv_acc_medium"] = tf.cond(tf.greater(num_medium, 0), lambda: self.tf_dict["h_deriv_acc_medium"],
                                                   lambda: -tf.constant(1.0))
        joint_dict["h_deriv_acc_medium"] = self.tf_dict["h_deriv_acc_medium"]
        if self.args.use_policy_reference and self.args.refine_policy==False:  # TODO since they maintain the shape (N/T, T, 2)
            loss_reg_policy_full = tf.reduce_sum(
                tf.math.square(dist_info_vars["mean"] - ref_mean_var) / (n_dim * t_dim), axis=[2]
            ) + tf.reduce_sum(
                tf.math.square(dist_info_vars["log_std"] - ref_log_std_var) / (n_dim * t_dim), axis=[2]
            )
        else:
            if self.args.reg_for_all_control:
                loss_reg_policy_full = tf.reduce_sum(tf.math.square(action+primal_control) / (n_dim * t_dim), axis=[1])
            else:
                loss_reg_policy_full = tf.reduce_sum(tf.math.square(action) / (n_dim*t_dim), axis=[1])

            loss_reg_policy_full = tf.reshape(loss_reg_policy_full, [n_dim, t_dim])
        loss_reg_policy = tf.reduce_sum(loss_reg_policy_full)

        self.tf_dict["deriv_total"] = loss_safe_deriv * self.args.safe_deriv_loss_weight \
                     + loss_dang_deriv * self.args.dang_deriv_loss_weight \
                     + loss_medium_deriv * self.args.medium_deriv_loss_weight

        total_loss_full = loss_safe_full * self.args.safe_loss_weight \
                     + loss_dang_full * self.args.dang_loss_weight \
                     + loss_safe_deriv_full * self.args.safe_deriv_loss_weight \
                     + loss_dang_deriv_full * self.args.dang_deriv_loss_weight \
                     + loss_medium_deriv_full * self.args.medium_deriv_loss_weight \
                     + loss_reg_policy_full * self.args.reg_policy_loss_weight

        total_loss = loss_safe * self.args.safe_loss_weight \
                     + loss_dang * self.args.dang_loss_weight \
                     + loss_safe_deriv * self.args.safe_deriv_loss_weight \
                     + loss_dang_deriv * self.args.dang_deriv_loss_weight \
                     + loss_medium_deriv * self.args.medium_deriv_loss_weight \
                     + loss_reg_policy * self.args.reg_policy_loss_weight


        self.tf_dict["total_loss"] = total_loss
        joint_dict["total_loss"] = self.tf_dict["total_loss"]
        self.tf_dict["loss_safe"] = loss_safe * self.args.safe_loss_weight
        joint_dict["loss_safe"] = self.tf_dict["loss_safe"]
        self.tf_dict["loss_dang"] = loss_dang * self.args.dang_loss_weight
        joint_dict["loss_dang"] = self.tf_dict["loss_dang"]
        self.tf_dict["loss_crit"] = loss_safe * self.args.safe_loss_weight + loss_dang * self.args.dang_loss_weight
        joint_dict["loss_crit"] = self.tf_dict["loss_crit"]
        self.tf_dict["loss_grad"] = loss_safe_deriv * self.args.safe_deriv_loss_weight \
                                    + loss_dang_deriv * self.args.dang_deriv_loss_weight \
                                    + loss_medium_deriv * self.args.medium_deriv_loss_weight
        joint_dict["loss_grad"] = self.tf_dict["loss_grad"]
        # self.tf_dict["loss_grad"] = tf.Print(self.tf_dict["loss_grad"], self.gradients, "d_u_res", summarize=-1)


        self.tf_dict["loss_safe_deriv"] = loss_safe_deriv * self.args.safe_deriv_loss_weight
        joint_dict["loss_safe_deriv"] = self.tf_dict["loss_safe_deriv"]
        self.tf_dict["loss_dang_deriv"] = loss_dang_deriv * self.args.dang_deriv_loss_weight
        joint_dict["loss_dang_deriv"] = self.tf_dict["loss_dang_deriv"]
        self.tf_dict["loss_medium_deriv"] = loss_medium_deriv * self.args.medium_deriv_loss_weight
        joint_dict["loss_medium_deriv"] = self.tf_dict["loss_medium_deriv"]
        self.tf_dict["loss_reg_policy"] = loss_reg_policy * self.args.reg_policy_loss_weight
        joint_dict["loss_reg_policy"] = self.tf_dict["loss_reg_policy"]
        self.tf_dict["num_dang"] = num_dang
        joint_dict["num_dang"] = self.tf_dict["num_dang"]
        self.tf_dict["num_safe"] = num_safe
        joint_dict["num_safe"] = self.tf_dict["num_safe"]
        self.tf_dict["num_medium"] = num_medium
        joint_dict["num_medium"] = self.tf_dict["num_medium"]
        self.tf_dict["acc_dang"] = acc_dang
        joint_dict["acc_dang"] = self.tf_dict["acc_dang"]
        self.tf_dict["acc_safe"] = acc_safe
        joint_dict["acc_safe"] = self.tf_dict["acc_safe"]

        return total_loss, total_loss_full, self.network.var_list, self.tf_dict, joint_dict


    # check feature definition at julia_pkgs/AutoRisk/src/extraction/feature_extractors.jl
    # function AutomotiveDrivingModels.pull_features!(
    #         ext::AttractiveExtractor,...
    def dynamics_attr(self, state, control, primal_control):
        next_state = state * 1.0
        # perform precised updates (numerically)
        dT = 0.1
        discrete_num = 4
        dt = dT / discrete_num

        for tt in range(discrete_num):
            # symbol table
            if self.n_ego_feat==11:
                curve = next_state[:, 0]
                lld = next_state[:, 1]
                rld = next_state[:, 2]
                lrd = next_state[:, 3]
                rrd = next_state[:, 4]
                theta = next_state[:, 2+3]
                v = next_state[:, 3+3]
                length = next_state[:, 6+3]
                width = next_state[:, 7+3]
            else:
                x1 = next_state[:, 0]
                x2 = next_state[:, 1]
                theta = next_state[:, 2]
                v = next_state[:, 3]
                length=next_state[:, 6]
                width=next_state[:, 7]

            accel = control[:, 0] + primal_control[:, 0]
            omega = control[:, 1] + primal_control[:, 1]

            #TODO(debug)
            self.tf_dict["dyna_accel_%d" % tt] = accel
            self.tf_dict["dyna_omega_%d" % tt] = omega
            self.tf_dict["dyna_v_%d" % tt] = v
            self.tf_dict["dyna_theta_%d" % tt] = theta

            # 1-order dynamic
            dx = v * tf.sin(theta) * dt
            ds = v * dt
            self.tf_dict["dyna_dx_%d" % tt] = dx
            self.tf_dict["dyna_ds_%d" % tt] = ds

            # 2-order dynamic
            dv = accel * dt
            dtheta = omega * dt
            self.tf_dict["dyna_dv_%d" % tt] = dv
            self.tf_dict["dyna_dtheta_%d" % tt] = dtheta

            # updates
            if self.n_ego_feat==11: #TODO lane related updates
                new_curve = curve
                new_lld = lld - dx
                new_rld = rld + dx
                new_lrd = lrd - dx
                new_rrd = rrd + dx
            else:
                new_x1 = x1 + dx
                new_x2 = x2 + dx
            new_theta = theta + dtheta
            new_v = v + dv
            self.tf_dict["dyna_new_v_%d" % tt] = new_v
            self.tf_dict["dyna_new_theta_%d" % tt] = new_theta
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

            self.tf_dict["dyna_nei_ind_%d"%tt] = nei_ind
            self.tf_dict["dyna_nei_x_%d" % tt] = nei_x
            self.tf_dict["dyna_nei_y_%d" % tt] = nei_y

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
            self.tf_dict["dyna_new_nei_x_%d" % tt] = new_nei_x
            self.tf_dict["dyna_new_nei_y_%d" % tt] = new_nei_y
            # (TODO) consider transformation caused by ego rotation
            new_nei_x, new_nei_y = \
                new_nei_x * tf.cos(dtheta_cp) + new_nei_y * tf.sin(dtheta_cp), \
                - new_nei_x * tf.sin(dtheta_cp) + new_nei_y * tf.cos(dtheta_cp)
            self.tf_dict["dyna_new_nei_x_new_%d" % tt] = new_nei_x
            self.tf_dict["dyna_new_nei_y_new_%d" % tt] = new_nei_y
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
            if self.n_ego_feat==11:
                new_ego = tf.stack([new_curve, new_lld, new_rld, new_lrd, new_rrd, new_theta, new_v, new_accel, new_omega, new_length, new_width], axis=-1)
            else:
                new_ego = tf.stack([new_x1, new_x2, new_theta, new_v, new_accel, new_omega, new_length, new_width], axis=-1)
            new_nei = tf.stack(
                [nei_ind, new_nei_x, new_nei_y, new_nei_theta, new_nei_v, new_nei_accel, new_nei_omega, new_nei_length, new_nei_width], axis=-1)

            new_nei = tf.reshape(new_nei, [-1, 6 * self.n_nei_feat])
            next_state = tf.concat([new_ego, new_nei], axis=-1)

        return next_state

    def dynamics_mono(self, state, control, primal_control):
        next_state = state * 1.0
        dT = 0.1
        discrete_num = 4
        dt = dT / discrete_num

        for tt in range(discrete_num):
            # symbol table
            ego_v = next_state[:, 0]
            head_x = next_state[:, 1]
            head_v = next_state[:, 2]
            rear_x = next_state[:, 3]
            rear_v = next_state[:, 4]

            accel = control[:, 0] + primal_control[:, 0]

            # updates
            head_x = head_x - ego_v * dt + head_v * dt
            rear_x = rear_x - ego_v * dt + rear_v * dt
            ego_v = ego_v + accel * dt

            next_state = tf.stack([ego_v, head_x, head_v, rear_x, rear_v], axis=-1)

        return next_state

    def get_safe_mask(self, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
        # get the safe mask from the states
        # input: (batchsize, feat_dim)
        # return: (batchsize, 1)

        bs = s.shape[0]
        neighbor_s = s[:, self.n_ego_feat:].reshape((bs, 6, self.n_nei_feat))
        nei_ind = neighbor_s[:, :, 0]

        # TODO out-of-road check (this is just heur. for straight lanes)
        if self.n_ego_feat == 11:
            outbound_mask = env_out_of_lane.reshape((bs,))  # total road width + 2*lanewidth

        if self.n_ego_feat==11:
            ego_l = s[:, 6+3]
            ego_w = s[:, 7+3]
        else:
            ego_l = s[:, 6]
            ego_w = s[:, 7]
        ego_r=(((ego_l ** 2 + ego_w ** 2) ** 0.5) / 2)
        nei_l=neighbor_s[:, :, 7]
        nei_w=neighbor_s[:, :, 8]
        nei_r=((nei_l ** 2 + nei_w ** 2) ** 0.5) / 2
        dx=neighbor_s[:, :, 1]
        dy=neighbor_s[:, :, 2]

        if self.args.bbx_collision_check:
            nei_dist = (dx ** 2 + dy ** 2) ** 0.5
            nei_thres = ego_r[:, None] + nei_r
            nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
            poss_collide = (np.sum(nei_collide, axis=1) > 0.5)
            for v_i in range(nei_collide.shape[0]):
                if not poss_collide[v_i]:
                    continue
                for nei_i in range(nei_collide.shape[1]):  # neighbors,
                    if nei_collide[v_i, nei_i]:  # fine-grained check to flip possible dang->safe
                        hl0 = ego_l[v_i] / 2 + dist_threshold / 2
                        hw0 = ego_w[v_i] / 2 + dist_threshold_side/2 #dist_threshold / 2

                        x1=dx[v_i, nei_i]
                        y1=dy[v_i, nei_i]
                        phi1 = neighbor_s[v_i, nei_i, 3] + np.pi/2
                        hl1 = nei_l[v_i, nei_i] / 2 + dist_threshold / 2
                        hw1 = nei_w[v_i, nei_i] / 2 + dist_threshold_side/2 #dist_threshold / 2

                        nei_collide[v_i, nei_i] = geo_check.check_rot_rect_collision(0, 0, np.pi/2, hl0, hw0, x1, y1, phi1, hl1, hw1)

            coll_mask = (np.sum(nei_collide, axis=1) > 0.5)

        else:
            nei_dist = (dx ** 2 + dy ** 2) ** 0.5
            nei_thres = ego_r[:, None] + nei_r
            nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
            coll_mask = (np.sum(nei_collide, axis=1) > 0.5)

        if self.n_ego_feat==11:
            dang_mask = np.logical_or(coll_mask, outbound_mask)
        else:
            dang_mask = coll_mask

        if check_safe:  # no collision for each row, means that ego vehicle is safe
            return np.logical_not(dang_mask)
        else:  # one collision in rows means that ego vehicle is dang
            return dang_mask

    def get_safe_mask_mono(self, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
        # get the safe mask from the states
        # input: (batchsize, feat_dim)
        # return: (batchsize, 1)

        a_max=4.0

        ego_v=s[:, 0]
        fore_x=s[:, 1]
        fore_v=s[:, 2]
        rear_x=s[:, 3]
        rear_v=s[:, 4]
        if self.args.simple_safe_checking:
            fore_collide = fore_x-5 < dist_threshold
            rear_collide = -rear_x-5 < dist_threshold
        else:
            fore_collide = (fore_x < (ego_v ** 2 - fore_v ** 2) / (2 * a_max) + dist_threshold)
            rear_collide = (-rear_x < (rear_v ** 2 - ego_v ** 2) / (2 * a_max) + dist_threshold)

        dang_mask = np.logical_or(fore_collide >= 0.5, rear_collide > 0.5)

        if check_safe:  # no collision for each row, means that ego vehicle is safe
            return np.logical_not(dang_mask)
        else:  # one collision in rows means that ego vehicle is dang
            return dang_mask


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