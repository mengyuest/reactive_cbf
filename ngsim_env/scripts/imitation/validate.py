
import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os 
import sys
import tensorflow as tf
import time

backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
import matplotlib
matplotlib.use(backend)
import matplotlib.pyplot as plt

from contexttimer import Timer

import hgail.misc.simulation
import hgail.misc.utils

import hyperparams
import utils
from utils import str2bool

# TODO(debug)
from utils import get_exp_home
import sandbox.rocky.tf.algos.utils_cbf as utils_cbf

from datetime import datetime

def simulate(env, policy, max_steps, render=False, env_kwargs=dict(), gail=None):
    traj = hgail.misc.simulation.Trajectory()
    x = env.reset(**env_kwargs)
    policy.reset()
    for step in range(max_steps):
        if render: env.render()
        a, a_info = policy.get_action(x)
        nx, r, done, e_info = env.step(a)
        traj.add(
            policy.observation_space.flatten(x), 
            a, 
            r, 
            a_info,
            e_info
        )
        if done: break
        x = nx
    return traj.flatten()

def mutliagent_simulate(
        env, 
        policy, 
        max_steps, 
        render=False, 
        env_kwargs=dict(),
        gail=None,
        policy_as_ref=None,
        traj_i=None,
        pedcyc_cache=None,
        x_pl=None,
        est_u=None,
        saved_data=None,
):
    '''
    Description:
        - simulates a vectorized agent in a vectorized environment 

    Args:
        - env: env to simulate in, must be vectorized
        - policy: policy, must be vectorized
        - max_steps: max steps in env per episode
        - render: display visual
        - env_kwargs: key word arguments to pass to env 

    Returns:
        - a dictionary object with a bit of an unusual format:
            each value in the dictionary is an array with shape 
            (timesteps, n_env / n_veh, shape of attribute)
            i.e., first dim corresponds to time 
            second dim to the index of the agent
            third dim is for the attribute itself
    '''
    
    x = env.reset(**env_kwargs)
    n_agents = x.shape[0]
    traj = hgail.misc.simulation.Trajectory()
    dones = [True] * n_agents
    if not args.behavior_cloning:
        if args.use_pedcyc:
            policy1, obs_mean1, obs_var1, policy2, obs_mean2, obs_var2 = pedcyc_cache

            if policy1 is not None:
                policy1.reset(dones[:run_args.n_envs1])
            if policy2 is not None:
                policy2.reset(dones[run_args.n_envs1:])

        else:
            policy.reset(dones)
    if run_args.residual_u:
        if args.use_nominal_controller==False and args.high_level==False:
            policy_as_ref.reset(dones)
    cnt=0

    ttt_list=[]

    if saved_data is not None:  # TODO(video)
        # _env = hgail.misc.utils.extract_normalizing_env(env)
        if args.use_pedcyc:
            saved_data["reactive_ids1"] = env._wrapped_env._wrapped_env.egoids1
            saved_data["reactive_ids2"] = env._wrapped_env._wrapped_env.egoids2
        else:
            if hasattr(env._wrapped_env._wrapped_env, "egoids"):
                saved_data["reactive_ids"] = env._wrapped_env._wrapped_env.egoids
            else:  # TODO for NGSIM
                saved_data["reactive_ids"] = env._wrapped_env._wrapped_env.get_ego_vehs(signal=6)



    for step in range(max_steps):
        # print("step=",step)
        # if run_args.debug_refine:
        #     print("step=", step)  # TODO(yue)
        #     if step >= 2:  # TODO(yue)
        #         exit()  # TODO(yue)
        # print("step=",step)
        if render: env.render()
        # env.render()

        if args.use_my_policy:  # TODO(yue)
            normalized_env = hgail.misc.utils.extract_normalizing_env(env)
            obs_mean = normalized_env._obs_mean
            obs_var = normalized_env._obs_var
            gt_x = x * np.sqrt(obs_var) + obs_mean
            a, a_info = policy.get_actions(gt_x)
        else:
            if args.behavior_cloning:
                if args.use_pedcyc:
                    est_u1, est_u2 = est_u
                    x_pl1, x_pl2 = x_pl
                    a1, = tf.get_default_session().run([est_u1], feed_dict={x_pl1: x[:run_args.n_envs1]})
                    a2, = tf.get_default_session().run([est_u2], feed_dict={x_pl2: x[run_args.n_envs1:]})
                    a1_info={}
                    a2_info={}
                    a = np.concatenate((a1, a2), axis=0)
                    a_info = {}
                    for key in a1_info:
                        a_info[key] = np.concatenate((a1_info[key], a2_info[key]), axis=0)
                else:
                    a, =  tf.get_default_session().run([est_u], feed_dict={x_pl:x})
                    a_info={}
            elif args.mpc:
                continue
            elif args.use_pedcyc:
                # mean         (n_envs1, 2) + (n_envs2, 2)
                # log_std      (n_envs1, 2) + (n_envs2, 2)
                # prev_action  (n_envs1, 2) + (n_envs2, 2)
                if policy1 is not None:
                    norm_x1 = (x[:run_args.n_envs1] - obs_mean1) / (np.sqrt(obs_var1) + 1e-8)
                    a1, a1_info = policy1.get_actions(norm_x1)
                    if run_args.unclip_action==False:
                        a1=np.clip(a1, -1, 1)
                        a1_info["prev_action"] = np.clip(a1_info["prev_action"], -1, 1)

                else:
                    a1 = np.zeros((0, 2))
                    a1_info = {"mean": np.zeros((0, 2)),
                               "log_std": np.zeros((0, 2)),
                               "prev_action": np.zeros((0,2))}

                if policy2 is not None:
                    norm_x2 = (x[run_args.n_envs1:] - obs_mean2) / (np.sqrt(obs_var2) + 1e-8)
                    a2, a2_info = policy2.get_actions(norm_x2)

                    if run_args.unclip_action==False:
                        a2=np.clip(a2, -1, 1)
                        a2_info["prev_action"] = np.clip(a2_info["prev_action"], -1, 1)
                    # for the policy2 case, we might need to consider rescaling, when using ped and cyc together
                    # because we are using simulator with (-4, 4) range but here we have (-4, 4) (-.15, .15)
                    # so for the second dimension, we need to *0.15/4

                    if run_args.control_mode=="ped_cyc":
                        a2[:, 1] = a2[:, 1] * 0.15 / 4.0
                        a2_info["prev_action"][:, 1] = a2_info["prev_action"][:, 1] * 0.15 / 4.0

                else:
                    a2 = np.zeros((0, 2))
                    a2_info = {"mean": np.zeros((0, 2)),
                               "log_std": np.zeros((0, 2)),
                               "prev_action": np.zeros((0,2))}


                # merge two policies
                a = np.concatenate((a1, a2), axis=0)
                a_info = {}
                for key in a1_info:
                    a_info[key] = np.concatenate((a1_info[key], a2_info[key]), axis=0)

            else:
                if args.record_time:
                    ttt1=time.time()
                a, a_info = policy.get_actions(x)
                if args.record_time:
                    ttt2=time.time()
                    ttt_list.append(ttt2-ttt1)
                    print(ttt_list[-1], np.mean(ttt_list))

        if run_args.residual_u:
            if args.use_nominal_controller: # this only for use_ped/use_easy/use_pedcyc, and for deterministic only
                if step >= 1:
                    a_ref = e_info["controls"] / 4.0
                    a_ref_info={"mean": a_ref}
                else:
                    a_ref = a * 0.0
                    a_ref_info = {"mean": a_ref}
            elif args.high_level:
                choice, choice_infos = gail.high_level_policy.get_choice(gt_x)
                lk_actions = utils_cbf.lane_keeping_controller(gt_x, args)
                if args.no_action_scaling==False:
                # print("lk_actions", lk_actions.shape, lk_actions)
                    lk_actions = lk_actions / np.array([[4.0, 0.15]])
                # print("lk_actions", lk_actions.shape, lk_actions)
                a_ref = lk_actions * choice[:, 0:1]
                a_ref_info = {"mean": a_ref}
            else:
                a_ref, a_ref_info = policy_as_ref.get_actions(x)
            if run_args.without_res_u:
                a = a_ref
                for k in a_ref_info:
                    a_info[k] = a_ref_info[k]
            elif run_args.without_res_u_ref:
                a = a
            elif run_args.zero_policy:
                a = a*0.0
                for k in a_info:
                    a_info[k] = a_info[k] * 0.0
            else:
                a = a + a_ref
                a_info['ref_actions'] = a_ref
                for k in a_ref_info:
                    a_info["ref_" + k] = a_ref_info[k]

        # TODO(yue) refinement
        if run_args.refine_policy and step >= 1:
            normalized_env = hgail.misc.utils.extract_normalizing_env(env)
            obs_mean = normalized_env._obs_mean
            obs_var = normalized_env._obs_var

            #TODO refine logic
            # print("original acc is",a.flatten())
            if run_args.qp_solve:
                u_res = gail.refine_policy_qp(x, e_info, a, a_info, obs_mean, obs_var, traj_i, step)
            else:
                u_res = gail.refine_policy(x, e_info, a, a_info, obs_mean, obs_var, traj_i, step)
            #TODO measure difference
            if args.no_action_scaling==False:
                if args.use_mono:
                    u_res = u_res[:, 0, :] / np.array([[4,]])  # normalize
                elif args.use_ped:
                    u_res = u_res[:, 0, :] / np.array([[4, 4]])  # normalize
                elif args.use_pedcyc:
                    if args.control_mode=="ped_only":
                        u_res = u_res[:, 0, :] / np.array([[4, 4]])  # normalize
                    elif args.control_mode=="cyc_only":
                        u_res = u_res[:, 0, :] / np.array([[4, .15]])  # normalize
                    else:
                        raise NotImplementedError
                elif args.use_round:
                    u_res = u_res[:, 0, :] / np.array([[4, 4]])  # normalize
                elif args.use_high:
                    u_res = u_res[:, 0, :] / np.array([[4, .15]])
                elif args.use_easy:
                    u_res = u_res[:, 0, :] / np.array([[4, 4]])  # normalize
                else:
                    u_res = u_res[:, 0, :] / np.array([[4, 0.15]]) # normalize
            else:
                u_res = u_res[:, 0, :]
            a = a + u_res
            # print("now acc is", a.flatten())

        if saved_data is not None:   # TODO(video)
            if args.use_ped:
                the_t = env._wrapped_env._wrapped_env.t - 1
                saved_data["agents"][the_t] = {}
                saved_data["vehs"][the_t] = {}

                # all pedestrians (but egoids) from scene
                # all egoids from scene
                # all vehs from scene
                for agi in env._wrapped_env._wrapped_env.traj_data.ped_snapshots[the_t]:
                    if agi not in env._wrapped_env._wrapped_env.egoids:
                        agent0 = env._wrapped_env._wrapped_env.traj_data.ped_snapshots[the_t][agi]
                    else:
                        agent0 = env._wrapped_env._wrapped_env.ego_peds[env._wrapped_env._wrapped_env.backtrace[agi]]
                    saved_data["agents"][the_t][agi] = [agent0.id, agent0.x, agent0.y, agent0.vx, agent0.vy]

                for agi in env._wrapped_env._wrapped_env.traj_data.veh_snapshots[the_t]:
                    agent0 = env._wrapped_env._wrapped_env.traj_data.veh_snapshots[the_t][agi]
                    saved_data["vehs"][the_t][agi] = [agent0.id,
                                                      agent0.x, agent0.y,
                                                      agent0.psi, agent0.v,
                                                      None, None,
                                                      agent0.length, agent0.width]
            elif args.use_round:
                do_nothing=1
                the_t = env._wrapped_env._wrapped_env.t - 1
                saved_data["agents"][the_t] = {}
                saved_data["vehs"][the_t] = {}

                # all pedestrians (but egoids) from scene
                # all egoids from scene
                # all vehs from scene
                # for agi in range(env._wrapped_env._wrapped_env.n_veh):
                #     agent0 = env._wrapped_env._wrapped_env.ego_vehs[agi]
                #     saved_data["agents"][the_t][agent0.id] = \
                #         np.array([agent0.id, agent0.x, agent0.y, agent0.heading, agent0.v_lon, agent0.length, agent0.width])
                #
                # for agi in env._wrapped_env._wrapped_env.scene:
                #     if agi not in env._wrapped_env._wrapped_env.egoids:
                #         agent0 = env._wrapped_env._wrapped_env.scene[agi]
                #         saved_data["agents"][the_t][agent0.id] = \
                #             np.array(
                #                 [agent0.id, agent0.x, agent0.y, agent0.heading, agent0.v_lon, agent0.length, agent0.width])

                # all pedestrians (but egoids) from scene
                # all egoids from scene
                # all vehs from scene
                if the_t in env._wrapped_env._wrapped_env.traj_data.npc_snapshots:
                    for agi in env._wrapped_env._wrapped_env.traj_data.npc_snapshots[the_t]:
                        agent0 = env._wrapped_env._wrapped_env.traj_data.npc_snapshots[the_t][agi]
                        saved_data["agents"][the_t][agi] = [agent0.id, agent0.x, agent0.y, agent0.vx, agent0.vy]

                for agi in env._wrapped_env._wrapped_env.traj_data.snapshots[the_t]:
                    if agi not in env._wrapped_env._wrapped_env.egoids:
                        agent0 = env._wrapped_env._wrapped_env.traj_data.snapshots[the_t][agi]
                    else:
                        agent0 = env._wrapped_env._wrapped_env.ego_vehs[env._wrapped_env._wrapped_env.backtrace[agi]]
                    saved_data["vehs"][the_t][agi] = [agent0.id,
                                                      agent0.x, agent0.y,
                                                      agent0.heading, agent0.v_lon,
                                                      None, None,
                                                      agent0.length, agent0.width]

            elif args.use_high:
                the_t = env._wrapped_env._wrapped_env.t - 1
                saved_data["agents"][the_t] = {}
                saved_data["vehs"][the_t] = {}
                for agi in env._wrapped_env._wrapped_env.traj_data.snapshots_upper[the_t]:
                    if agi not in env._wrapped_env._wrapped_env.egoids:
                        agent0 = env._wrapped_env._wrapped_env.traj_data.snapshots_upper[the_t][agi]
                    else:
                        agent0 = env._wrapped_env._wrapped_env.ego_vehs[env._wrapped_env._wrapped_env.backtrace[agi]]
                    saved_data["vehs"][the_t][agi] = [agent0.id, agent0.x, agent0.y,
                                                      agent0.heading, agent0.v_lon,
                                                      None, None, agent0.length, agent0.width]
                for agi in env._wrapped_env._wrapped_env.traj_data.snapshots_lower[the_t]:
                    agent0 = env._wrapped_env._wrapped_env.traj_data.snapshots_lower[the_t][agi]
                    saved_data["vehs"][the_t][agi] = [agent0.id, agent0.x, agent0.y,
                                                      agent0.heading, agent0.v_lon,
                                                      None, None, agent0.length, agent0.width]

            elif args.use_pedcyc:
                the_t = env._wrapped_env._wrapped_env.t - 1
                saved_data["agents1"][the_t] = {}
                saved_data["agents2"][the_t] = {}

                # all pedestrians (but egoids) from scene
                # all egoids from scene
                # all vehs from scene
                for agi in env._wrapped_env._wrapped_env.traj_data["Pedestrian"]["snapshots"][the_t]:
                    if agi not in env._wrapped_env._wrapped_env.egoids1:
                        agent0 = env._wrapped_env._wrapped_env.traj_data["Pedestrian"]["snapshots"][the_t][agi]
                    else:
                        agent0 = env._wrapped_env._wrapped_env.ego_peds1[env._wrapped_env._wrapped_env.backtrace[agi]]

                    saved_data["agents1"][the_t][agi] = [agi, agent0[0], agent0[1], agent0[2], agent0[3]]

                for agi in env._wrapped_env._wrapped_env.traj_data["Biker"]["snapshots"][the_t]:
                    if agi not in env._wrapped_env._wrapped_env.egoids2:
                        agent0 = env._wrapped_env._wrapped_env.traj_data["Biker"]["snapshots"][the_t][agi]
                    else:
                        agent0 = env._wrapped_env._wrapped_env.ego_peds2[env._wrapped_env._wrapped_env.backtrace[agi] - env._wrapped_env._wrapped_env.n_veh1]
                    saved_data["agents2"][the_t][agi] = [agi, agent0[0], agent0[1], agent0[2], agent0[3]]
            else:
                assert args.use_mono==False and args.use_easy==False
                # TODO this is for NGSIM
                the_t = int(env._wrapped_env._wrapped_env.get_ego_vehs(signal=7)[0]) - 1
                saved_data["agents"][the_t] = {}
                saved_data["vehs"][the_t] = {}
                egoids = env._wrapped_env._wrapped_env.get_ego_vehs(signal=6)

                all_veh_states = env._wrapped_env._wrapped_env.get_ego_vehs(signal=4)
                ego_veh_states = env._wrapped_env._wrapped_env.get_ego_vehs(signal=5)

                all_veh_states = np.array(all_veh_states).reshape((-1, 9))
                ego_veh_states = np.array(ego_veh_states).reshape((-1, 9))

                rev_d={}
                for agi in range(ego_veh_states.shape[0]):
                    rev_d[ego_veh_states[agi, 0]] = agi

                for agi in range(all_veh_states.shape[0]):
                    v_id = all_veh_states[agi, 0]
                    if v_id in egoids:  # use ego veh
                        agent0 = ego_veh_states[rev_d[v_id], :]
                    else:  # use veh from scene
                        agent0 = all_veh_states[agi, :]
                    saved_data["vehs"][the_t][v_id] = list(agent0)

        nx, r, dones, e_info = env.step(a)
        # print(step, e_info["is_colliding"])
        # if np.sum(e_info["is_colliding"]>0):
        # cnt+=1
        # if cnt==100:
        #     exit()
        traj.add(x, a, r, a_info, e_info)
        if any(dones): break
        x = nx
    return traj.flatten()

def collect_trajectories(
        args,  
        params, 
        egoids, 
        starts,
        trajlist,
        pid,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed,
        x_pl=None,
        est_u=None,):

    #TODO(yue)
    if args.use_mono:
        env_fn = utils.build_mono_env
    elif args.use_ped:
        env_fn = utils.build_ped_env
    elif args.use_easy:
        env_fn = utils.build_easy_env
    elif args.use_pedcyc:
        env_fn = utils.build_pedcyc_env
    elif args.use_round:
        env_fn = utils.build_round_env
    elif args.use_high:
        env_fn = utils.build_high_env

    env, _, _ = env_fn(args, exp_dir=run_args.exp_dir, alpha=0., debug_args=debug_args)

    if args.behavior_cloning:
        policy = None
        policy1 = None
        policy2 = None
    else:
        if args.use_pedcyc:
            if args.control_mode=="ped_cyc":
                policy1=policy_fn(args, env, is_first=True)
                policy2=policy_fn(args, env, is_first=False)
            elif args.control_mode=="ped_only":
                policy1=policy_fn(args, env, is_first=True)
                policy2=None
            elif args.control_mode=="cyc_only":
                policy1=None
                policy2=policy_fn(args, env, is_first=False)
            else:
                raise NotImplementedError
        else:
            policy = policy_fn(args, env)




    # TODO(yue)
    if run_args.refine_policy:
        from hgail.algos.gail import GAIL
        from hgail.core.models import StateMLP
        gail = GAIL(
            critic=None, recognition=None, reward_handler=None,
            env=env, policy=policy,
            cbfer=None, baseline=None, validator=None,
            batch_size=args.batch_size, max_path_length=args.max_path_length,
            n_itr=args.n_itr, discount=args.discount, step_size=args.trpo_step_size,
            saver=None, saver_filepath=None, force_batch_sampler=False if args.vectorize else True,
            sampler_args=None,
            snapshot_env=False, plot=False, optimizer=None,
            optimizer_args=dict(
                max_backtracks=50,
                debug_nan=True
            ),
            args=args,  # TODO(yue)
            network = None,
            policy_as_ref=policy_fn(args, env, as_reference=True) if args.use_policy_reference else None,  # TODO(yue)
            high_level_policy=utils_cbf.MyPolicyHighLevel(env, args) if args.high_level else None,  # TODO(yue)
            summary_writer=None
        )
        if run_args.residual_u and args.use_nominal_controller==False:
            policy_as_ref = gail.policy_as_ref
        else:
            policy_as_ref = None
    else:
        if run_args.residual_u and args.use_nominal_controller==False:
            policy_as_ref = policy_fn(args, env, as_reference=True)
        else:
            policy_as_ref = None

    if args.high_level:
        true_obs_var = env.observation_space.new_tensor_variable('true_obs', extra_dims=1 + 1)
        gail.kwargs["high_level_policy"].dist_info_sym(true_obs_var)

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        pedcyc_cache=None
        if args.use_pedcyc:
            if policy1 is not None:
                params1 = hgail.misc.utils.load_params(run_args.policy1_path)
                policy1.set_param_values(params1["policy"])
                obs_mean1 = params1['normalzing']['obs_mean']
                obs_var1 = params1['normalzing']['obs_var']
            else:
                obs_mean1 = None
                obs_var1 = None

            if policy2 is not None:
                params2 = hgail.misc.utils.load_params(run_args.policy2_path)
                policy2.set_param_values(params2["policy"])
                obs_mean2=params2['normalzing']['obs_mean']
                obs_var2=params2['normalzing']['obs_var']
            else:
                obs_mean2 = None
                obs_var2 = None

            normalized_env = hgail.misc.utils.extract_normalizing_env(env)
            if normalized_env is not None:
                normalized_env._obs_mean = np.zeros_like(params['normalzing']['obs_mean'])
                normalized_env._obs_var = np.ones_like(params['normalzing']['obs_var'])
            pedcyc_cache = (policy1, obs_mean1, obs_var1, policy2, obs_mean2, obs_var2)

            policy=None
            gail=None

        else:
            if args.behavior_cloning:
                normalized_env = hgail.misc.utils.extract_normalizing_env(env)
                normalized_env._obs_mean = np.zeros_like(params['normalzing']['obs_mean'])
                normalized_env._obs_var = np.ones_like(params['normalzing']['obs_var'])
                params1 = hgail.misc.utils.load_params(run_args.bc_policy_path1)
                var_list1 = [v for v in tf.trainable_variables() if "policy1" in v.name]
                assign1 = tf.group(*[tf.assign(var, val) for (var, val) in zip(var_list1, params["policy"])])

                params2 = hgail.misc.utils.load_params(run_args.bc_policy_path2)
                var_list2 = [v for v in tf.trainable_variables() if "policy2" in v.name]
                assign2 = tf.group(*[tf.assign(var, val) for (var, val) in zip(var_list2, params["policy"])])

                session = tf.get_default_session()
                session.run(assign1)
                session.run(assign2)
                gail = None
            else:
                if use_hgail:
                    for i, level in enumerate(policy):
                        level.algo.policy.set_param_values(params[i]['policy'])
                    policy = policy[0].algo.policy
                else:
                    policy.set_param_values(params['policy'])
                    if args.high_level:
                        gail.kwargs["high_level_policy"].set_param_values(params["high_level_policy"])
                    if run_args.residual_u and args.use_nominal_controller==False and args.high_level==False:
                        if args.reference_control:
                            do_nothing=1
                        else:
                            params_ref = hgail.misc.utils.load_params(run_args.policy_reference_path)
                            policy_as_ref.set_param_values(params_ref['policy'])
                    if run_args.refine_policy:
                        utils_cbf.set_cbf_param_values(params['jcbfer'], args)
                    else:
                        gail = None
                normalized_env = hgail.misc.utils.extract_normalizing_env(env)
                if normalized_env is not None:
                    normalized_env._obs_mean = params['normalzing']['obs_mean']
                    normalized_env._obs_var = params['normalzing']['obs_var']



        # collect trajectories
        nids = len(egoids)
        for i, egoid in enumerate(egoids):
            # if not run_args.batch_ops:
            #     sys.stdout.write('\rpid: {} traj: {} / {}'.format(pid, i, nids))

            if args.save_traj_data:  # TODO(video)
                # metadata (exp_dir, cmd, etc)
                # pedestrian data t -> {id -> {id, x, y, vx, vy}}
                # vehicle data t -> {id, x, y, th, vlon, accel ,omega, L, W}
                # control slices {id list}
                if args.use_pedcyc:
                    saved_data = {"exp_dir": run_args.full_log_dir,
                                  "cmd": "python " + " ".join(sys.argv),
                                  "reactive_ids1": None,
                                  "reactive_ids2": None,
                                  "agents1": {}, "agents2": {}, }
                else:
                    saved_data = {"exp_dir": run_args.full_log_dir,
                                  "cmd": "python " + " ".join(sys.argv),
                                  "reactive_ids": None,
                                  "agents": {}, "vehs": {}, }
            else:
                saved_data = None

            if args.env_multiagent:
                kwargs = dict()
                if random_seed:
                    kwargs = dict(random_seed=random_seed+egoid)
                traj = mutliagent_simulate(
                    env,
                    policy,
                    max_steps=max_steps,
                    env_kwargs=kwargs,
                    gail=gail,
                    policy_as_ref=policy_as_ref,
                    traj_i=i,
                    render=run_args.validator_render,  # TODO(yue)
                    pedcyc_cache=pedcyc_cache,
                    x_pl=x_pl, est_u=est_u,
                    saved_data=saved_data
                )
                out_mean = np.mean(cal_mean([traj], "out_of_lane", args.env_H))
                rmse_mean = np.mean(cal_mean([traj], "rmse_pos", args.env_H))
                coll_mean = np.mean(cal_mean([traj], "is_colliding", args.env_H))
                new_coll = np.mean(cal_mean([traj], "is_colliding", args.env_H, traj_wise=True))
                out_mean_list.append(out_mean)
                rmse_mean_list.append(rmse_mean)
                coll_mean_list.append(coll_mean)
                new_coll_list.append(new_coll)

                if args.use_pedcyc:
                    # TODO
                    rmse_ped = np.mean(cal_mean([traj], "rmse_pos", args.env_H, is_first=True))
                    coll_ped = np.mean(cal_mean([traj], "is_colliding", args.env_H, is_first=True))
                    new_coll_ped = np.mean(cal_mean([traj], "is_colliding", args.env_H, is_first=True, traj_wise=True))
                    rmse_cyc = np.mean(cal_mean([traj], "rmse_pos", args.env_H, is_first=False))
                    coll_cyc = np.mean(cal_mean([traj], "is_colliding", args.env_H, is_first=False))
                    new_coll_cyc = np.mean(cal_mean([traj], "is_colliding", args.env_H, is_first=False, traj_wise=True))

                    rmse_ped_list.append(rmse_ped)
                    coll_ped_list.append(coll_ped)
                    new_coll_ped_list.append(new_coll_ped)
                    rmse_cyc_list.append(rmse_cyc)
                    coll_cyc_list.append(coll_cyc)
                    new_coll_cyc_list.append(new_coll_cyc)

                if not run_args.batch_ops:
                    if args.use_pedcyc:
                        log_str = "Traj%d/%d Mean:%.4f(%.4f)  Coll:%.4f(%.4f)  Traj:%.4f(%.4f) |" \
                                  "Mean:%.4f(%.4f)  Coll:%.4f(%.4f)  Traj:%.4f(%.4f) |" \
                                  "Mean:%.4f(%.4f)  Coll:%.4f(%.4f)  Traj:%.4f(%.4f)" % \
                                  (i, nids, rmse_mean, np.mean(rmse_mean_list),
                                   coll_mean, np.mean(coll_mean_list), new_coll, np.mean(new_coll_list),

                                   rmse_ped, np.mean(rmse_ped_list),
                                   coll_ped, np.mean(coll_ped_list), new_coll_ped, np.mean(new_coll_ped_list),

                                   rmse_cyc, np.mean(rmse_cyc_list),
                                   coll_cyc, np.mean(coll_cyc_list), new_coll_cyc, np.mean(new_coll_cyc_list),
                                   )
                    else:
                        log_str="Traj%d/%d Out: %.4f(%.4f)  Mean is: %.4f(%.4f)  Collision: %.4f(%.4f)  TrajColl: %.4f(%.4f)" % \
                                (i, nids, out_mean, np.mean(out_mean_list), rmse_mean, np.mean(rmse_mean_list),
                                 coll_mean, np.mean(coll_mean_list), new_coll, np.mean(new_coll_list))
                    print(log_str)
                    logger.write(log_str+"\n")
                    logger.flush()

                trajlist.append(traj)
            else:
                traj = simulate(
                    env,
                    policy,
                    max_steps=max_steps,
                    gail=gail,
                    policy_as_ref=policy_as_ref,
                    env_kwargs=dict(egoid=egoid, start=starts[egoid])
                )
                traj['egoid'] = egoid
                traj['start'] = starts[egoid]
                trajlist.append(traj)


            # TODO save my traj data
            if args.save_traj_data:  # TODO(video)
                print("SAVE TO",os.path.join(run_args.full_log_dir, "traj_data_%d.npz" % (i)))
                np.savez(os.path.join(run_args.full_log_dir, "traj_data_%d.npz" % (i)), data=saved_data)

    return trajlist

def parallel_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc,
        env_fn=utils.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None):
    # build manager and dictionary mapping ego ids to list of trajectories
    manager = mp.Manager()
    trajlist = manager.list()

    # set policy function
    policy_fn = utils.build_hierarchy if use_hgail else utils.build_policy
    
    # partition egoids 
    proc_egoids = utils.partition_list(egoids, n_proc)

    # pool of processes, each with a set of ego ids
    pool = mp.Pool(processes=n_proc)

    # run collection
    results = []
    for pid in range(n_proc):
        res = pool.apply_async(
            collect_trajectories,
            args=(
                args, 
                params, 
                proc_egoids[pid], 
                starts,
                trajlist, 
                pid,
                env_fn,
                policy_fn,
                max_steps,
                use_hgail,
                random_seed
            )
        )
        results.append(res)

    # wait for the processes to finish
    [res.get() for res in results]
    pool.close()
    # let the julia processes finish up
    time.sleep(5)
    return trajlist

def single_process_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc, 
        env_fn=utils.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None):
    '''
    This function for debugging purposes
    '''
    # build list to be appended to 
    trajlist = []

    # set policy function
    policy_fn = utils.build_hierarchy if use_hgail else utils.build_policy
    tf.reset_default_graph()

    if run_args.behavior_cloning is not None:
        args.behavior_cloning = run_args.behavior_cloning
        if run_args.behavior_cloning:
            x_pl, est_u = build_bc_graph(double=args.use_pedcyc)
    else:
        x_pl = None
        est_u = None
    if run_args.mpc is not None:
        args.mpc = run_args.mpc
        args.planning_horizon = run_args.planning_horizon

    # collect trajectories in a single process
    collect_trajectories(
        args, 
        params, 
        egoids, 
        starts,
        trajlist, 
        1,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed,
        x_pl=x_pl,
        est_u=est_u,
    )
    return trajlist    

def collect(
        egoids,
        starts,
        args,
        exp_dir,
        use_hgail,
        params_filename,
        n_proc,
        max_steps=200,
        collect_fn=parallel_collect_trajectories,
        random_seed=None):
    '''
    Description:
        - prepare for running collection in parallel
        - multiagent note: egoids and starts are not currently used when running 
            this with args.env_multiagent == True 
    '''
    # load information relevant to the experiment
    params_filepath = os.path.join(exp_dir, 'imitate/log/{}'.format(params_filename))
    params = hgail.misc.utils.load_params(params_filepath)

    # validation setup 
    validation_dir = os.path.join(exp_dir, 'imitate', 'validation')
    run_args.exp_dir = exp_dir
    #TODO multiple validation sets
    # if args.params_filename!="itr_200.npz":
    model_idx = run_args.params_filename.split("_")[1].split(".")[0]
    validation_dir += "_m" + model_idx

    # if args.n_envs != 50:
    validation_dir += "_n" + str(run_args.n_envs)

    validation_dir += "_t" + str(run_args.n_multiagent_trajs)

    validation_dir += run_args.suffix

    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_trajectories.npz'.format(
        args.ngsim_filename.split('.')[0]))

    with Timer():
        trajs = collect_fn(
            args, 
            params, 
            egoids, 
            starts,
            n_proc,
            max_steps=max_steps,
            use_hgail=use_hgail,
            random_seed=random_seed
        )

    out_mean = np.mean(cal_mean(trajs, "out_of_lane", args.env_H))
    rmse_mean = np.mean(cal_mean(trajs, "rmse_pos", args.env_H))
    coll_mean = np.mean(cal_mean(trajs, "is_colliding", args.env_H))
    new_coll = np.mean(cal_mean(trajs, "is_colliding", args.env_H, traj_wise=True))

    if args.use_pedcyc:
        metric_str = "Mean: %.4f  Coll: %.4f  TrajColl: %.4f | Mean: %.4f  Coll: %.4f  TrajColl: %.4f " \
                     "| Mean: %.4f  Coll: %.4f  TrajColl: %.4f " % (
            rmse_mean, coll_mean, new_coll, np.mean(rmse_ped_list), np.mean(coll_ped_list), np.mean(new_coll_ped_list),
            np.mean(rmse_cyc_list), np.mean(coll_cyc_list), np.mean(new_coll_cyc_list),
        )
    else:
        metric_str="Out is: %.6f\tMean is: %.6f\tCollision: %.6f\t TrajColl: %.6f" % (out_mean, rmse_mean, coll_mean, new_coll)
    print(metric_str)

    if not any([args.use_mono, args.use_ped, args.use_easy, args.use_pedcyc, args.use_round, args.use_high]):
        print()
        if args.env_H == 200:
            # print("Histograms")
            # print("estimate a", np.histogram([x['env_info_a'] for x in trajs if len(x['env_info_a'])==200], bins=[-6,-4,-2,0,2,4,6]))
            # print("estimate w", np.histogram([x['env_info_w'] for x in trajs if len(x['env_info_w'])==200], bins=[-0.25,-0.2,-0.15,-0.10,-0.05,0,0.05,0.10, 0.15, 0.20, 0.25]))
            #
            # print()
            # rmse pos from different time lens
            for time_len in [1, 10, 50, 200]:
                rm = np.mean(cal_mean(trajs, "rmse_pos", 200, time_len))
                print("TimeLen=%d mean is: %.6f" % (time_len, rm))
            print()
            # <state, action> discrepancy from diff time lens
            for time_len in [1,10,50,200]:
                da0 = cal_mean_pair(trajs, "env_info_a", "env_info_a_gt0", 200, time_len)
                dw0 = cal_mean_pair(trajs, "env_info_w", "env_info_w_gt0", 200, time_len)
                # da1 = cal_mean_pair(trajs, "env_info_a", "env_info_a_gt1", 200, time_len)
                # dw1 = cal_mean_pair(trajs, "env_info_w", "env_info_w_gt1", 200, time_len)
                print("TimeLen=%d, da0:%.4f\tdw0:%.4f"%(time_len, da0,dw0))
            print()
            # <state, action> discretized discrepancy from diff time lens
            for time_len in [1,10,50,200]:
                da0 = cal_mean_pair(trajs, "env_info_a", "env_info_a_gt0", 200, time_len, discrete=True)
                dw0 = cal_mean_pair(trajs, "env_info_w", "env_info_w_gt0", 200, time_len, discrete=True)
                # da1 = cal_mean_pair(trajs, "env_info_a", "env_info_a_gt1", 200, time_len, discrete=True)
                # dw1 = cal_mean_pair(trajs, "env_info_w", "env_info_w_gt1", 200, time_len, discrete=True)
                print("TimeLen=%d(discrete) da0:%.4f\tdw0:%.4f"%(time_len, da0,dw0))


    if run_args.refine_policy:
        from datetime import datetime
        prefix = datetime.fromtimestamp(time.time()).strftime("g%y%m%d_%H%M%S")
    else:
        prefix = ""

    if run_args.ped_sim_with_veh:
        prefix += "_wveh"

    from os.path import join as ospj
    if args.use_pedcyc:
        metric_filepath = ospj(exp_dir, "%s%sm%s_n%d_n%d_t%d_%.4f_%.4f_%.4f" % (
            prefix, args.prefix, model_idx, run_args.n_envs1, run_args.n_envs2,
            run_args.n_multiagent_trajs, rmse_mean, coll_mean, new_coll))
    else:
        metric_filepath = ospj(exp_dir, "%s%sm%s_n%d_t%d_%.4f_%.4f_%.4f" % (
            prefix, args.prefix, model_idx, run_args.n_envs, run_args.n_multiagent_trajs, rmse_mean, coll_mean, new_coll))
    with open(metric_filepath, "w") as f:
        f.write("python " + " ".join(sys.argv)+"\n")
        f.write(metric_str+"\n")
        lines= open(global_cache_filepath).readlines()
        for line in lines:
            f.write(line)

    if args.behavior_cloning==False:
        os.remove(global_cache_filepath)

    utils.write_trajectories(output_filepath, trajs)

def cal_mean(trajs, attr="rmse",length=200, cap_len=None, traj_wise=False, is_first=None):
    rmses = []
    for traj in trajs:
        if len(traj[attr]) == length:
            if is_first is not None:
                if is_first:
                    veh_range = range(0, run_args.n_envs1)
                else:
                    veh_range = range(run_args.n_envs1, run_args.n_envs1+run_args.n_envs2)
                if len(list(veh_range))==0:
                    return 0.0
            else:
                veh_range=range(traj[attr].shape[1])
            for veh_i in veh_range:
                if cap_len is None:
                    rmses.append(traj[attr][:, veh_i])
                else:
                    rmses.append(traj[attr][:cap_len, veh_i])
    if traj_wise:
        return (np.sum(np.array(rmses), axis=1)>0.5)
    else:
        return np.mean(np.array(rmses), axis=0)

def cal_mean_pair(trajs, attr0, attr1, length, cap_len, discrete=False):
    rmses = []
    for traj in trajs:
        if len(traj[attr0]) == length:
            for veh_i in range(traj[attr0].shape[1]):
                x0=traj[attr0][:cap_len, veh_i]
                x1=traj[attr1][:cap_len, veh_i]
                rmses.append((x0-x1)**2)
    return np.sqrt(np.mean(np.array(rmses)))

def load_egoids(filename, args, n_runs_per_ego_id=1, env_fn=utils.build_ngsim_env):
    offset = args.env_H + args.env_primesteps
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data/')
    ids_filename = filename.replace('.txt', '-index-{}-ids.h5'.format(offset))
    ids_filepath = os.path.join(basedir, ids_filename)
    if not os.path.exists(ids_filepath):
        # this should create the ids file
        env_fn(args, debug_args=debug_args)
        if not os.path.exists(ids_filepath):
            raise ValueError('file unable to be created, check args')
    ids = np.array(h5py.File(ids_filepath, 'r')['ids'].value)

    # we want to sample start times uniformly from the range of possible values 
    # but we also want these start times to be identical for every model we 
    # validate. So we sample the start times a single time, and save them.
    # if they exist, we load them in and reuse them
    start_times_filename = filename.replace('.txt', '-index-{}-starts.h5'.format(offset))
    start_times_filepath = os.path.join(basedir, start_times_filename)
    # check if start time filepath exists
    if os.path.exists(start_times_filepath):
        # load them in
        starts = np.array(h5py.File(start_times_filepath, 'r')['starts'].value)
    # otherwise, sample the start times and save them
    else:
        ids_file = h5py.File(ids_filepath, 'r')
        ts = ids_file['ts'].value
        # subtract offset gives valid end points
        te = ids_file['te'].value - offset
        starts = np.array([np.random.randint(s,e+1) for (s,e) in zip(ts,te)])
        # write to file
        starts_file = h5py.File(start_times_filepath, 'w')
        starts_file.create_dataset('starts', data=starts)
        starts_file.close()

    # create a dict from id to start time
    id2starts = dict()
    for (egoid, start) in zip(ids, starts):
        id2starts[egoid] = start

    ids = np.tile(ids, n_runs_per_ego_id)
    return ids, id2starts


def build_bc_graph(double=False):
    # build computation graph

    if args.use_ped:
        feat_dim=34
    elif args.use_pedcyc:
        feat_dim1=34
        feat_dim2=34
    elif args.use_high:
        feat_dim=56
    elif args.use_round:
        feat_dim=32
    else:
        feat_dim=65

    hiddens = [256, 256, 256, 64]
    if double:
        x_pl1 = tf.placeholder(tf.float32, shape=[None, feat_dim1], name="x1")
        x_pl2 = tf.placeholder(tf.float32, shape=[None, feat_dim2], name="x2")

        cat_x1 = tf.expand_dims(x_pl1, axis=1)
        cat_x2 = tf.expand_dims(x_pl2, axis=1)
        for i, hidden_num in enumerate(hiddens):
            cat_x1 = tf.contrib.layers.conv1d(inputs=cat_x1,
                                             num_outputs=hidden_num,
                                             kernel_size=1,
                                             reuse=tf.AUTO_REUSE,
                                             scope='policy1/conv%d' % i,
                                             activation_fn=tf.nn.relu)
            cat_x2 = tf.contrib.layers.conv1d(inputs=cat_x2,
                                              num_outputs=hidden_num,
                                              kernel_size=1,
                                              reuse=tf.AUTO_REUSE,
                                              scope='policy2/conv%d' % i,
                                              activation_fn=tf.nn.relu)

        cat_x1 = tf.contrib.layers.conv1d(inputs=cat_x1,
                                         num_outputs=2,
                                         kernel_size=1,
                                         reuse=tf.AUTO_REUSE,
                                         scope='policy1/conv%d' % len(hiddens),
                                         activation_fn=None)
        cat_x2 = tf.contrib.layers.conv1d(inputs=cat_x2,
                                          num_outputs=2,
                                          kernel_size=1,
                                          reuse=tf.AUTO_REUSE,
                                          scope='policy2/conv%d' % len(hiddens),
                                          activation_fn=None)

        est_u1 = tf.squeeze(cat_x1, axis=1)  # TODO(modified)
        est_u2 = tf.squeeze(cat_x2, axis=1)  # TODO(modified)

        return (x_pl1,x_pl2), (est_u1,est_u2)
    else:
        x_pl = tf.placeholder(tf.float32, shape=[None, feat_dim], name="x")
        cat_x = tf.expand_dims(x_pl, axis=1)
        for i, hidden_num in enumerate(hiddens):
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
                                         scope='policy/conv%d' % len(hiddens),
                                         activation_fn=None)
        est_u = tf.squeeze(cat_x, axis=1)  # TODO(modified)
        return x_pl, est_u


def write_cmd_to_file(log_dir, argv):
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    with open(os.path.join(log_dir, "cmd.txt"), "w") as f:
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

class Logger(object):
    def __init__(self, path):
        self._terminal = sys.stdout
        self._log = open(path, "w")

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, default=get_exp_home() + 'gail/')
    parser.add_argument('--params_filename', type=str, default='itr_2000.npz')
    parser.add_argument('--n_runs_per_ego_id', type=int, default=1)
    parser.add_argument('--use_hgail', type=str2bool, default=False)
    parser.add_argument('--use_multiagent', type=str2bool, default=False)
    parser.add_argument('--n_multiagent_trajs', type=int, default=10000)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--remove_ngsim_vehicles', type=str2bool, default=False)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--refine_policy', type=str2bool, default=False)
    parser.add_argument('--refine_learning_rate', type=float, default=None)
    parser.add_argument('--refine_n_iter', type=int, default=20)
    parser.add_argument('--remove_dropout', type=str2bool, default=False)
    parser.add_argument('--batch_ops', type=str2bool, default=False)
    parser.add_argument('--debug_refine', type=str2bool, default=False)
    parser.add_argument('--fake_net', type=str2bool, default=False)
    parser.add_argument('--use_mono', type=str2bool, default=False)
    parser.add_argument('--validator_render', type=str2bool, default=False)
    parser.add_argument('--unclip_action', type=str2bool, default=False)
    parser.add_argument('--fixed_trajectory', type=str2bool, default=False)
    parser.add_argument('--traj_idx_list', type=str, default="4,5,6,7,8,9,10")

    parser.add_argument('--ped_sim_with_veh', type=str2bool, default=False)  # ped_sim with vehicles
    parser.add_argument('--veh_mode', type=str, default="points", choices=["points", "radius", "bbox"])
    parser.add_argument('--veh_num_pts', type=int, default=4)
    parser.add_argument('--veh_pts_reso', type=float, default=0.30)

    parser.add_argument('--env_H', type=int, default=None)
    parser.add_argument('--qp_solve', type=str2bool, default=False)
    parser.add_argument('--qp_alpha', type=float, default=0.01)
    parser.add_argument('--qp_accel_weight', type=float, default=1.0)  # TODO only for NGSIM data
    parser.add_argument('--qp_omega_weight', type=float, default=1.0)  # TODO only for NGSIM data

    parser.add_argument('--without_res_u', type=str2bool, default=False)  # TODO only for residual_u case, test primal
    parser.add_argument('--without_res_u_ref', type=str2bool, default=False)  # TODO only for residual_u case, test primal
    parser.add_argument('--zero_policy', type=str2bool, default=False)

    parser.add_argument('--init_with_lcs', type=str2bool, default=False)
    parser.add_argument('--lcs_fixed_id', type=int, default=1)

    parser.add_argument('--control_mode', type=str, choices=["ped_cyc", "ped_only", "cyc_only"], default=None)
    parser.add_argument('--n_envs1', type=int, default=None)
    parser.add_argument('--n_envs2', type=int, default=None)

    parser.add_argument('--policy1_path', type=str, default=None)
    parser.add_argument('--policy2_path', type=str, default=None)

    parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument('--schedule_file', type=str, default=None)
    parser.add_argument('--env_primesteps', type=int, default=None)

    parser.add_argument('--behavior_cloning', type=str2bool, default=None)
    parser.add_argument('--bc_policy_path', type=str, default="")
    parser.add_argument('--bc_policy_path1', type=str, default="")
    parser.add_argument('--bc_policy_path2', type=str, default="")

    parser.add_argument('--no_action_scaling', type=str2bool, default="")

    parser.add_argument('--mpc', type=str2bool, default=None)
    parser.add_argument('--planning_horizon', type=str2bool, default=None)

    parser.add_argument('--idm_data', type=str2bool, default=None)
    parser.add_argument('--record_time', type=str2bool, default=None)

    parser.add_argument('--save_traj_data', type=str2bool, default=None)

    parser.add_argument('--video_mode', type=str2bool, default=None)
    parser.add_argument('--video_traj_idx', type=int, default=None)
    parser.add_argument('--video_egoids', type=str, default=None)
    parser.add_argument('--video_t', type=int, default=None)
    parser.add_argument('--video_h', type=int, default=None)
    parser.add_argument('--action_constraint', type=float, default=None)
    parser.add_argument('--speed_constraint', type=float, default=None)
    parser.add_argument('--allow_out', type=str2bool, default=None)
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--video_egoids1', type=str, default=None)
    parser.add_argument('--video_egoids2', type=str, default=None)

    parser.add_argument('--file_splits',type=str,default=None)
    parser.add_argument('--ext_intervals',type=str,default=None) # this is only for NGSIM, normally we shouldn't use this

    parser.add_argument('--use_ngsim',type=str2bool, default=None) # this is only for old version PS-GAIL for NGSIM

    parser.add_argument('--show_collision', type=str2bool, default=None)

    run_args = parser.parse_args()

    # TODO(yue) non-deterministic
    import random
    random.seed(1007)
    np.random.seed(1007)
    tf.set_random_seed(1007)

    if run_args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_args.gpus

    run_args.full_log_dir = os.path.join(run_args.exp_dir, "%sm%s_n%d_t%d" % (
            run_args.prefix, run_args.params_filename.split("_")[1].split(".")[0], run_args.n_envs, run_args.n_multiagent_trajs))
    os.makedirs(run_args.full_log_dir, exist_ok=True)
    # TODO cmd file
    write_cmd_to_file(run_args.full_log_dir, sys.argv)

    # TODO log file
    sys.stdout = Logger(os.path.join(run_args.full_log_dir, "log.txt"))



    print("validate for", run_args.exp_dir, "...")
    t1=time.time()
    args_filepath = os.path.join(run_args.exp_dir, 'imitate/log/args.npz')
    args = hyperparams.load_args(args_filepath)

    #TODO(debug)
    # debug_args = {
    #     "ext_intervals": args.ext_intervals,
    #     "affordance": args.affordance,
    #     "lane_control": args.lane_control,
    #     "attention": args.attention,
    #     "learn_cbf": args.learn_cbf,
    #     "cbf_intervals": args.cbf_intervals,
    # }

    debug_args={}

    if run_args.validator_render:
        # TODO viz file
        viz_dir = os.path.join(run_args.full_log_dir, 'viz')
        debug_args["viz_dir"] = viz_dir
        os.makedirs(viz_dir, exist_ok=True)
        print("viz_dir",viz_dir)
    else:
        viz_dir=""
        debug_args["viz_dir"] = viz_dir

    if run_args.ext_intervals is not None:
        args.ext_intervals = run_args.ext_intervals

    if args.affordance:
        debug_args["affordance"] = True
    if args.attention:
        debug_args["attention"] = True
    if args.learn_cbf:
        debug_args["learn_cbf"] = True
    if args.lane_control:
        debug_args["lane_control"] = True
    if args.multilane_control:
        debug_args["multilane_control"] = True
    if args.ext_intervals != "":
        debug_args["ext_intervals"] = args.ext_intervals
    if args.cbf_intervals != "":
        debug_args["cbf_intervals"] = args.cbf_intervals
    if args.attractive:
        debug_args["attractive"] = True
    if args.new_affordance:
        debug_args["new_affordance"] = True
    if args.naive_control:
        debug_args["naive_control"] = True
    if args.ctrl_intervals != "":
        debug_args["ctrl_intervals"] = args.ctrl_intervals
    if args.cbf_ctrl_intervals != "":
        debug_args["cbf_ctrl_intervals"] = args.cbf_ctrl_intervals
    if args.joint_cbf:
        debug_args["joint_cbf"] = args.joint_cbf
    if args.naive_control_clip:
        debug_args["naive_control_clip"] = args.naive_control_clip
    if args.aggressive:
        debug_args["aggressive"] = args.aggressive

    if run_args.unclip_action:
        args.unclip_action = run_args.unclip_action

    if run_args.show_collision:
        debug_args["show_collision"] = run_args.show_collision

    if args.use_ped:
        debug_args["traj_idx_list"] = run_args.traj_idx_list #TODO this we might want to change
        debug_args["num_neighbors"] = args.num_neighbors
        debug_args["ped_radius"] = args.ped_radius

        # TODO let run_args override args <+ in case we want to train without vehicles, but test with it
        if run_args.ped_sim_with_veh and args.ped_sim_with_veh==False:
            args.ped_sim_with_veh = True
            args.veh_mode = run_args.veh_mode
            args.veh_num_pts = run_args.veh_num_pts
            args.veh_pts_reso = run_args.veh_pts_reso

        debug_args["ped_sim_with_veh"] = args.ped_sim_with_veh

        # TODO load vehicles to ped sim
        if args.ped_sim_with_veh:
            debug_args["veh_mode"] = args.veh_mode
            debug_args["veh_num_pts"] = args.veh_num_pts
            debug_args["veh_pts_reso"] = args.veh_pts_reso

        debug_args["record_vxvy"] = args.record_vxvy

    if args.use_easy:
        debug_args["radius"] = args.radius
        debug_args["obs_radius"] = args.obs_radius

    if args.use_pedcyc:
        if run_args.traj_idx_list!="4,5,6,7,8,9,10":
            args.traj_idx_list = run_args.traj_idx_list
        debug_args["traj_idx_list"] = args.traj_idx_list  # TODO this we might want to change
        debug_args["num_neighbors"] = args.num_neighbors
        debug_args["ped_radius"] = args.ped_radius

        if run_args.control_mode is not None:
            args.control_mode = run_args.control_mode
        if run_args.n_envs1 is not None:
            args.n_envs1 = run_args.n_envs1
        if run_args.n_envs2 is not None:
            args.n_envs2 = run_args.n_envs2
        debug_args["control_mode"] = args.control_mode
        debug_args["n_veh1"] = args.n_envs1
        debug_args["n_veh2"] = args.n_envs2
        debug_args["record_vxvy"] = args.record_vxvy
        debug_args["fps"] = args.fps

    if args.use_round:
        debug_args["traj_idx_list"] = args.traj_idx_list
        debug_args["num_neighbors"] = args.num_neighbors
        debug_args["ped_radius"] = args.ped_radius
        debug_args["record_vxvy"] = args.record_vxvy
        debug_args["fps"] = args.fps

        debug_args["print_gt"] = args.print_gt
        debug_args["without_background"] = args.without_background
        debug_args["filtered_background"] = args.filtered_background
        debug_args["figure_ratio"] = args.figure_ratio
        debug_args["filter_sampling"] = args.filter_sampling
        debug_args["init_vmin"] = args.init_vmin
        debug_args["init_vmax"] = args.init_vmax
        debug_args["plot_circles"] = args.plot_circles
        debug_args["plot_gt_trajectory"] = args.plot_gt_trajectory
        debug_args["plot_plan_trajectory"] = args.plot_plan_trajectory

    if args.use_high:
        if run_args.traj_idx_list != "4,5,6,7,8,9,10":
            args.traj_idx_list = run_args.traj_idx_list
        debug_args["traj_idx_list"] = args.traj_idx_list
        debug_args["num_neighbors"] = args.num_neighbors
        debug_args["print_gt"] = args.print_gt
        debug_args["fps"] = args.fps
        debug_args["without_background"] = args.without_background
        debug_args["figure_ratio"] = args.figure_ratio
        debug_args["plot_gt_trajectory"] = args.plot_gt_trajectory
        debug_args["neighbor_feature"] = args.neighbor_feature

    if run_args.idm_data:
        args.idm_data = run_args.idm_data
        debug_args["idm_data"] = True

    if run_args.env_H is not None:
        print("over-write train.env_H %d with validate.env_H %d"%(args.env_H, run_args.env_H))
        args.env_H = run_args.env_H

    if run_args.env_primesteps is not None:
        args.env_primesteps = run_args.env_primesteps

    if run_args.schedule_file is not None:
        lines=open(run_args.schedule_file).readlines()
        sche_data=[]
        for line in lines:
            split_data=line.strip().split(" ")
            f_ts, f_te, f_num = int(split_data[0]), int(split_data[1]), int(split_data[2])
            if f_num > args.n_envs:
                possibles = [int(x) for x in split_data[3].split(",")]
                sche_data.append([])
                sche_data[-1].append(f_ts)
                sche_data[-1].append(f_te)
                sche_data[-1].append(f_num)
                sche_data[-1] += possibles[:args.n_envs]
        debug_args["schedule_data"] = np.array(sche_data)

    if run_args.record_time is not None:
        args.record_time = run_args.record_time

    if run_args.save_traj_data is not None:
        args.save_traj_data = run_args.save_traj_data
    if run_args.video_mode is not None:    # TODO(video)
        args.video_mode = run_args.video_mode
        args.video_traj_idx = run_args.video_traj_idx
        if run_args.use_ngsim:
            run_args.video_egoids = [int(xxx) for xxx in run_args.video_egoids.split(",")]
            args.video_egoids = run_args.video_egoids
        else:
            args.video_egoids = run_args.video_egoids
        args.video_t = run_args.video_t
        args.video_h = run_args.video_h
        args.video_egoids1 = run_args.video_egoids1
        args.video_egoids2 = run_args.video_egoids2
        debug_args["video_mode"] = args.video_mode
        debug_args["video_traj_idx"] = args.video_traj_idx
        debug_args["video_egoids"] = args.video_egoids
        debug_args["video_t"] = args.video_t
        debug_args["video_h"] = args.video_h
        debug_args["video_egoids1"] = args.video_egoids1
        debug_args["video_egoids2"] = args.video_egoids2

    if run_args.action_constraint is not None:
        args.action_constraint = run_args.action_constraint
        debug_args["action_constraint"] = args.action_constraint

    if run_args.speed_constraint is not None:
        args.speed_constraint = run_args.speed_constraint
        debug_args["speed_constraint"] = args.speed_constraint

    if run_args.allow_out is not None:
        args.allow_out = run_args.allow_out
        debug_args["allow_out"] = args.allow_out

    if run_args.file_splits is not None:
        args.file_splits = run_args.file_splits
        debug_args["file_splits"] = args.file_splits

    args.prefix = run_args.prefix

    args.ngsim_filename_list = None

    # if args.random_seed is not None:
    #     debug_args["random_seed"] = args.random_seed
    debug_args["random_seed"] = 1007


    if run_args.refine_policy:
        args.refine_policy = run_args.refine_policy
        args.refine_learning_rate = run_args.refine_learning_rate
        args.refine_n_iter = run_args.refine_n_iter
        args.remove_dropout = run_args.remove_dropout
        args.debug_refine = run_args.debug_refine
        args.fake_net = run_args.fake_net

    if run_args.qp_solve:
        if run_args.refine_policy==False:
            exit("qp_solve must be along with refine_policy")
        args.qp_solve = run_args.qp_solve
        if run_args.qp_alpha != 0.01:
            args.qp_alpha = run_args.qp_alpha
        args.qp_accel_weight = run_args.qp_accel_weight
        args.qp_omega_weight = run_args.qp_omega_weight

    if run_args.fixed_trajectory:
        args.fixed_trajectory = run_args.fixed_trajectory
        debug_args["fixed_trajectory"] = run_args.fixed_trajectory

    run_args.residual_u = args.residual_u
    if args.policy_reference_path is not None:
        run_args.policy_reference_path = args.policy_reference_path

    if run_args.use_multiagent:
        args.env_multiagent = True
        args.remove_ngsim_vehicles = run_args.remove_ngsim_vehicles

    args.validator_render=run_args.validator_render

    if run_args.debug:
        collect_fn = single_process_collect_trajectories
    else:
        collect_fn = parallel_collect_trajectories


    args.exp_dir = run_args.exp_dir
    args.full_log_dir = run_args.full_log_dir

    #TODO
    args.init_with_lcs = run_args.init_with_lcs
    args.lcs_fixed_id = run_args.lcs_fixed_id

    filenames = [
        "trajdata_i101_trajectories-0750am-0805am.txt",
        # "trajdata_i101_trajectories-0805am-0820am.txt",
        # "trajdata_i101_trajectories-0820am-0835am.txt",
        # "trajdata_i80_trajectories-0400-0415.txt",
        # "trajdata_i80_trajectories-0500-0515.txt",
        # "trajdata_i80_trajectories-0515-0530.txt"
    ]
    if run_args.n_envs:
        args.n_envs = run_args.n_envs
    if not run_args.batch_ops:
        sys.stdout.write('{} vehicles with H = {}'.format(args.n_envs, args.env_H))
    # TODO(yue)
    global_cache_filepath=os.path.join(run_args.exp_dir, "validation_log%s.txt"%(datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")))

    if run_args.behavior_cloning:
        global_cache_filepath = global_cache_filepath.replace("validation_log","val_bc_log")

    if run_args.mpc:
        global_cache_filepath = global_cache_filepath.replace("validation_log", "val_mpc_log")

    logger = open(global_cache_filepath, "w")
    if args.ped_sim_with_veh:
        logger.write("PedSim with vehicles\n")
        logger.flush()
    out_mean_list =[]
    rmse_mean_list = []
    coll_mean_list = []
    new_coll_list = []

    rmse_ped_list=[]
    coll_ped_list=[]
    new_coll_ped_list=[]

    rmse_cyc_list = []
    coll_cyc_list = []
    new_coll_cyc_list = []




    for fn in filenames:
        args.ngsim_filename = fn
        if args.env_multiagent:
            # args.n_envs gives the number of simultaneous vehicles 
            # so run_args.n_multiagent_trajs / args.n_envs gives the number 
            # of simulations to run overall
            egoids = list(range(int(run_args.n_multiagent_trajs / args.n_envs)))
            starts = dict()
        else:
            egoids, starts = load_egoids(fn, args, run_args.n_runs_per_ego_id)
        collect(
            egoids,
            starts,
            args,
            exp_dir=run_args.exp_dir,
            params_filename=run_args.params_filename,
            use_hgail=run_args.use_hgail,
            n_proc=run_args.n_proc,
            collect_fn=collect_fn,
            random_seed=run_args.random_seed
        )

    t2=time.time()
    print("Finished in %.4f s"%(t2-t1))