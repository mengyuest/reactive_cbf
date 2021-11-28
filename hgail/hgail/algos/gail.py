
import numpy as np
import os
import tensorflow as tf

from rllab.misc.overrides import overrides

from sandbox.rocky.tf.algos.trpo import TRPO

import hgail.misc.utils
#TODO(debug)
import rllab.misc.logger as logger

import sandbox.rocky.tf.algos.utils_cbf as utils_cbf

import time

class Object:
    pass

class GAIL(TRPO):
    """
    Generative Adversarial Imitation Learning
    """
    def __init__(
            self,
            critic=None,
            recognition=None,
            reward_handler=hgail.misc.utils.RewardHandler(),
            saver=None,
            saver_filepath=None,
            validator=None,
            snapshot_env=True,
            cbfer=None, #TODO(debug)
            **kwargs):
        """
        Args:
            critic: 
            recognition:
        """
        self.critic = critic
        self.recognition = recognition
        self.reward_handler = reward_handler
        self.saver = saver
        self.saver_filepath = saver_filepath
        self.validator = validator
        self.snapshot_env = snapshot_env
        self.cbfer = cbfer  #TODO(debug)
        self.is_colliding_list = []# TODO(debug)
        self.rmse_list=[]# TODO(debug)
        self.new_coll_list=[] #TODO(debug) for vehicle traj-wise collision
        self.out_of_lane_list=[]

        if "args" in kwargs:
            self.args = kwargs["args"]
        else:
            self.args = Object()
            self.args.quiet=False
            self.args.ref_policy = False
            self.args.save_model_freq = 1
            self.args.joint_cbf=False
            self.args.use_my_policy=False
            self.args.high_level=False



        self.kwargs = kwargs
        if "summary_writer" in kwargs:
            self.summary_writer = kwargs["summary_writer"]
        else:
            self.summary_writer = None
        super(GAIL, self).__init__(**kwargs)
    
    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Update the critic and recognition model in addition to the policy
        
        Args:
            itr: iteration counter
            samples_data: dictionary resulting from process_samples
                keys: 'rewards', 'observations', 'agent_infos', 'env_infos', 'returns', 
                      'actions', 'advantages', 'paths'
                the values in the infos dicts can be accessed for example as:

                    samples_data['agent_infos']['prob']
    
                and the returned value will be an array of shape (batch_size, prob_dim)
        """
        if self.args.skip_optimize == True:
            self.show_collision_rmse(itr, samples_data)
        else:
            super(GAIL, self).optimize_policy(itr, samples_data)
            if self.cbfer is not None:  #TODO(debug)
                self.cbfer.train(itr, samples_data)
            if self.critic is not None:
                self.critic.train(itr, samples_data)
            if self.recognition is not None:
                self.recognition.train(itr, samples_data)

            # TODO(debug)
            self.show_collision_rmse(itr, samples_data)

    # TODO(debug)
    def show_collision_rmse(self, itr,samples_data):
        if "out_of_lane" in samples_data["env_infos"]:
            self.out_of_lane_list.append(np.mean(samples_data["env_infos"]["out_of_lane"]))
        self.is_colliding_list.append(np.mean(samples_data["env_infos"]["is_colliding"]))
        self.rmse_list.append(np.mean(samples_data["env_infos"]["rmse_pos"]))
        self.new_coll_list.append(
            np.mean(np.sum(samples_data["env_infos"]["is_colliding"], axis=1) > 0.5))
        if self.args.quiet==False:
            if "out_of_lane" in samples_data["env_infos"]:
                logger.log("out_lane : %.4f (%.4f)\t[best= %.4f at %d] {%.4f %.4f}" % (
                    self.out_of_lane_list[-1], np.mean(self.out_of_lane_list),
                    np.min(self.out_of_lane_list), np.argmin(self.out_of_lane_list),
                    np.mean(self.out_of_lane_list[-100:]),
                    np.mean(self.out_of_lane_list[-20:]),
                ))
            logger.log("collision: %.4f (%.4f)\t[best= %.4f at %d] {%.4f %.4f}" % (
                self.is_colliding_list[-1], np.mean(self.is_colliding_list),
                np.min(self.is_colliding_list), np.argmin(self.is_colliding_list),
                np.mean(self.is_colliding_list[-100:]),
                np.mean(self.is_colliding_list[-20:]),
            ))
            logger.log("traj-coll: %.4f (%.4f)\t[best= %.4f at %d] {%.4f %.4f}" %(
                self.new_coll_list[-1], np.mean(self.new_coll_list),
                np.min(self.new_coll_list), np.argmin(self.new_coll_list),
                np.mean(self.new_coll_list[-100:]),
                np.mean(self.new_coll_list[-20:]),
            ))
            logger.log("rmse_pos : %.4f (%.4f)\t[best= %.4f at %d] {%.4f %.4f}" % (
                self.rmse_list[-1], np.mean(self.rmse_list),
                np.min(self.rmse_list), np.argmin(self.rmse_list),
                np.mean(self.rmse_list[-100:]),
                np.mean(self.rmse_list[-20:]),
            ))
        else:
            logger.log("out:%.4f(%.4f) coll:%.4f(%.4f) traj:%.4f(%.4f) rmse:%.4f(%.4f)"%(
                self.out_of_lane_list[-1], np.mean(self.out_of_lane_list),self.is_colliding_list[-1], np.mean(self.is_colliding_list),
                self.new_coll_list[-1], np.mean(self.new_coll_list),self.rmse_list[-1], np.mean(self.rmse_list),
            ), with_prefix=False, with_timestamp=False)

        if self.summary_writer is not None:
            summary = tf.Summary()
            summary.value.add(tag='0_out_lane', simple_value=self.out_of_lane_list[-1])
            self.summary_writer.add_summary(summary, itr)

            summary = tf.Summary()
            summary.value.add(tag='1_collision', simple_value=self.is_colliding_list[-1])
            self.summary_writer.add_summary(summary, itr)

            summary = tf.Summary()
            summary.value.add(tag='2_traj_coll', simple_value=self.new_coll_list[-1])
            self.summary_writer.add_summary(summary, itr)

            summary = tf.Summary()
            summary.value.add(tag='3_rmse_pos', simple_value=self.rmse_list[-1])
            self.summary_writer.add_summary(summary, itr)

    @overrides
    def process_samples(self, itr, paths):
        """
        Augment path rewards with critic and recognition model rewards
        
        Args:
            itr: iteration counter
            paths: list of dictionaries 
                each containing info for a single trajectory
                each with keys 'observations', 'actions', 'agent_infos', 'env_infos', 'rewards'
        """
        if self.args.ref_policy==False:
            # compute critic and recognition rewards and combine them with the path rewards
            critic_rewards = self.critic.critique(itr, paths) if self.critic else None
            recognition_rewards = self.recognition.recognize(itr, paths) if self.recognition else None
            paths = self.reward_handler.merge(paths, critic_rewards, recognition_rewards)
        return self.sampler.process_samples(itr, paths)

    def _save(self, itr):
        """
        Save a tf checkpoint of the session.
        """
        # using keep_checkpoint_every_n_hours as proxy for iterations between saves
        if self.saver and (itr + 1) % self.saver._keep_checkpoint_every_n_hours == 0 and \
                (hasattr(self.args, "save_model_freq") == False or itr % self.args.save_model_freq==0):

            # collect params (or stuff to keep in general)
            params = dict()
            if self.cbfer:
                params['cbfer'] = self.cbfer.network.get_param_values()
            if self.critic:
                params['critic'] = self.critic.network.get_param_values()
            if self.recognition:
                params['recognition'] = self.recognition.network.get_param_values()
            if self.args.joint_cbf:
                params['jcbfer'] = utils_cbf.get_cbf_param_values(self.args)
            if self.args.use_my_policy:
                params['policy'] = utils_cbf.get_policy_param_values(self.args)
            else:
                params['policy'] = self.policy.get_param_values()

            if self.args.high_level:
                params['high_level_policy'] = utils_cbf.get_high_level_policy_param_values(self.args)

            # if the environment is wrapped in a normalizing env, save those stats
            normalized_env = hgail.misc.utils.extract_normalizing_env(self.env)
            if normalized_env is not None:
                params['normalzing'] = dict(
                    obs_mean=normalized_env._obs_mean,
                    obs_var=normalized_env._obs_var
                )

            # save params 
            save_dir = os.path.split(self.saver_filepath)[0]
            hgail.misc.utils.save_params(save_dir, params, itr+1, max_to_keep=10000)

    def load(self, filepath):
        '''
        Load parameters from a filepath. Symmetric to _save. This is not ideal, 
        but it's easier than keeping track of everything separately.
        '''
        params = hgail.misc.utils.load_params(filepath)
        if self.cbfer and 'cbfer' in params.keys():  #TODO(debug)
            self.cbfer.network.set_param_values(params['cbfer'])
        if self.critic and 'critic' in params.keys():
            self.critic.network.set_param_values(params['critic'])
        if self.recognition and 'recognition' in params.keys():
            self.recognition.network.set_param_values(params['recognition'])
        if self.args.joint_cbf:  # TODO(yue)
            utils_cbf.set_cbf_param_values(params['jcbfer'], self.args)

        assert self.args.use_my_policy==False
        self.policy.set_param_values(params['policy'])

        if self.args.high_level:
            self.high_level_policy.set_param_values(params['high_level_policy'])

        normalized_env = hgail.misc.utils.extract_normalizing_env(self.env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

    def load_ref(self):
        print("Load nominal controller from PolicyNet...")
        policy_params = hgail.misc.utils.load_params(self.args.policy_reference_path)
        self.kwargs["policy_as_ref"].set_param_values(policy_params['policy'])  # TODO(yue)
        if not self.args.init_policy_from_scratch and not self.args.residual_u and not self.args.use_my_policy:
            print("Load same params to current learning PolicyNet...")
            self.policy.set_param_values(policy_params['policy'])

        normalized_env = hgail.misc.utils.extract_normalizing_env(self.env)
        if normalized_env is not None:
            normalized_env._obs_mean = policy_params['normalzing']['obs_mean']
            normalized_env._obs_var = policy_params['normalzing']['obs_var']

    def load_cbf(self):
        print("Load CBF pretrained weights...")
        cbf_params = hgail.misc.utils.load_params(self.args.cbf_pretrained_path)
        utils_cbf.set_cbf_param_values(cbf_params['jcbfer'], self.args)

    def load_policy(self):
        print("Load policy pretrained weights...")
        policy_params = hgail.misc.utils.load_params(self.args.policy_pretrained_path)
        # if self.args.use_my_policy:
        #     raise NotImplementedError
        # else:
        self.policy.set_param_values(policy_params['policy'])

    def load_high_level_policy(self):
        print("Load HIGH-LEVEL policy pretrained weights...")
        policy_params = hgail.misc.utils.load_params(self.args.high_level_policy_pretrained_path)
        self.kwargs["high_level_policy"].set_param_values(policy_params['high_level_policy'])

    def _validate(self, itr, samples_data):
        """
        Run validation functions.
        """
        if self.validator:
            objs = dict(
                policy=self.policy, 
                critic=self.critic,
                cbfer=self.cbfer, #TODO(debug)
                samples_data=samples_data,
                env=self.env)
            self.validator.validate(itr, objs)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        """
        Snapshot critic and recognition model as well
        """
        self._save(itr)
        self._validate(itr, samples_data)
        if self.args.use_my_policy:
            snapshot = dict(
                itr=itr,
            )
        else:
            snapshot = dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
            )
        if self.snapshot_env:
            snapshot['env'] = self.env
        if samples_data is not None:
            snapshot['samples_data'] = dict()
            if 'actions' in samples_data.keys():
                snapshot['samples_data']['actions'] = samples_data['actions'][:10]
            if 'mean' in samples_data.keys():
                snapshot['samples_data']['mean'] = samples_data['mean'][:10]

        return snapshot