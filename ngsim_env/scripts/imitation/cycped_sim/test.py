import numpy as np
import tensorflow as tf
import argparse
import configs
import os
import random
import utils
import train
import sandbox.rocky.tf.algos.utils_cbf as utils_cbf
import hgail.misc.simulation
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
import rllab.spaces
import time
import sys

class Logger(object):
    def __init__(self, path):
        self._terminal=sys.stdout
        self._log = open(path,"w")
    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)
        self._log.flush()
    def flush(self):
        pass

def main():
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--params_filename', type=str, default='itr_2000.npz')
    parser.add_argument('--n_multiagent_trajs', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=3)
    parser.add_argument('--debug_render', type=configs.str2bool, default=False)
    parser.add_argument('--debug_render_freq', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument('--zero_policy', type=configs.str2bool, default=False)
    parser.add_argument('--dest_controller_type', type=str, default=None)

    parser.add_argument('--refinement', type=configs.str2bool, default=False)  # TODO
    parser.add_argument('--refine_n_iter', type=int, default=10)
    parser.add_argument('--refine_learning_rate', type=float, default=0.1)

    parser.add_argument('--pre_action_clip', type=float, default=None)
    parser.add_argument('--u_res_norm', type=float, default=100)
    parser.add_argument('--env_H', type=int, default=None)

    parser.add_argument('--control_mode', type=str, choices=["ped_cyc", "ped_only", "cyc_only"], default='ped_cyc')
    parser.add_argument('--n_envs1', type=int, default=None)
    parser.add_argument('--n_envs2', type=int, default=None)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--policy_pretrained_path1', type=str, default=None)
    parser.add_argument('--policy_pretrained_path2', type=str, default=None)

    parser.add_argument('--prefix', type=str, default="")

    parser.add_argument('--save_traj_data', type=configs.str2bool, default=None)
    parser.add_argument('--video_mode', type=configs.str2bool, default=None)
    parser.add_argument('--video_traj_idx', type=int, default=None)
    parser.add_argument('--video_egoids1', type=str, default=None)
    parser.add_argument('--video_egoids2', type=str, default=None)
    parser.add_argument('--video_t', type=int, default=None)
    parser.add_argument('--video_h', type=int, default=None)

    parser.add_argument('--action_constraint', type=float, default=None)
    parser.add_argument('--speed_constraint', type=float, default=None)

    parser.add_argument('--use_gt_trajs', type=configs.str2bool, default=False)

    run_args = parser.parse_args()
    if run_args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_args.gpus

    t1=time.time()

    args = configs.load_args(os.path.join(run_args.exp_dir, "imitate/log/args.npz"))

    args.debug_render = run_args.debug_render
    args.debug_render_freq = run_args.debug_render_freq
    if run_args.n_envs1 is not None:
        args.n_envs1 = run_args.n_envs1
    if run_args.n_envs2 is not None:
        args.n_envs2 = run_args.n_envs2
    if run_args.env_H is not None:
        args.env_H = run_args.env_H
    args.batch_size = args.env_H * (args.n_envs1+args.n_envs2)
    args.random_seed = run_args.random_seed

    args.zero_policy = run_args.zero_policy
    args.refinement = run_args.refinement
    args.refine_n_iter = run_args.refine_n_iter
    args.refine_learning_rate = run_args.refine_learning_rate

    args.u_res_norm = run_args.u_res_norm
    args.pre_action_clip = run_args.pre_action_clip

    if args.dest_controller_type is None:
        args.dest_controller_type = "none"

    if run_args.dest_controller_type is not None:
        args.dest_controller_type = run_args.dest_controller_type


    if args.refinement:
        target_dir = "val_%s_n%d_t%d_ref%d_%.4f%s"%(run_args.params_filename.split(".npz")[0],
                                           run_args.n_envs1 + run_args.n_envs2,
                                           run_args.n_multiagent_trajs,
                                           args.refine_n_iter, args.refine_learning_rate, run_args.prefix)
    else:
        target_dir = "val_%s_n%d_t%d%s" % (run_args.params_filename.split(".npz")[0],
                                           run_args.n_envs1 + run_args.n_envs2,
                                           run_args.n_multiagent_trajs, run_args.prefix)

    if run_args.save_traj_data is not None:    # TODO(video)
        args.save_traj_data = run_args.save_traj_data

    if run_args.video_mode is not None:    # TODO(video)
        args.video_mode = run_args.video_mode
        args.video_traj_idx = run_args.video_traj_idx
        args.video_egoids1 = run_args.video_egoids1
        args.video_egoids2 = run_args.video_egoids2
        args.video_t = run_args.video_t
        args.video_h = run_args.video_h

    if run_args.use_gt_trajs is not None:
        args.use_gt_trajs = run_args.use_gt_trajs

    if run_args.action_constraint is not None:
        args.action_constraint = run_args.action_constraint
    if run_args.speed_constraint is not None:
        args.speed_constraint = run_args.speed_constraint

    target_dir_full = os.path.join(run_args.exp_dir, target_dir)
    os.makedirs(target_dir_full, exist_ok=True)
    utils.write_cmd_to_file(target_dir_full, sys.argv)
    args.full_log_dir = target_dir_full

    sys.stdout = Logger(os.path.join(target_dir_full, "log.txt"))

    random.seed(run_args.random_seed)
    np.random.seed(run_args.random_seed)
    tf.set_random_seed(run_args.random_seed)

    debug_args = utils.match_for_args_dict(args, test=True)

    env, act_low, act_high = utils.build_pedcyc_env(args, debug_args, for_test=True)

    if args.n_envs1 > 0:
        policy1 = train.MyPolicy(env, args, name="myp1")
    else:
        policy1 = None
    if args.n_envs2 > 0:
        policy2 = train.MyPolicy(env, args, name="myp2")
    else:
        policy2 = None

    if args.policy_reference_path is not None and args.zero_reference == False and args.dest_controller_type == "none":
        ps_indices = train.get_ext_indices(args.ps_intervals)
        ref_obs_space = rllab.spaces.Box(low=env.spec.observation_space.low[ps_indices],
                                         high=env.spec.observation_space.high[ps_indices])
        ref_action_space = env.spec.action_space
        ref_env = train.MockEnv(ref_obs_space, ref_action_space)
        policy_ref = GaussianGRUPolicy(
            name="policy_ref",
            env_spec=ref_env,
            hidden_dim=args.recurrent_hidden_dim,
            output_nonlinearity=None,
            learn_std=True,
            args=args
        )
    else:
        policy_ref = None

    if run_args.refinement:
        pl_dict, tf_dict, u_init1, u_res_new1, u_init2, u_res_new2 \
            = train.build_computation_graph(policy1, policy2, policy_ref, args, refinement=run_args.refinement)
    else:
        _,_,_,_,_ = train.build_computation_graph(policy1, policy2, policy_ref, args, refinement=run_args.refinement)

    if args.refinement:
        refinement_cache = u_init1, u_res_new1, u_init2, u_res_new2, pl_dict
    else:
        refinement_cache = None
    coll_list = []
    coll1_list = []
    rmse_list = []

    traj_list=[]

    assert run_args.fps==10
    k_table = np.load("K_lqr_v_t0.1000.npz")["table"].item()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #TODO (load new weights)
        args.cbf_pretrained_path = os.path.join(run_args.exp_dir, "imitate/log", run_args.params_filename)
        args.policy_pretrained_path1 = os.path.join(run_args.exp_dir, "imitate/log", run_args.params_filename)
        args.policy_pretrained_path2 = os.path.join(run_args.exp_dir, "imitate/log", run_args.params_filename)
        if run_args.policy_pretrained_path1 is not None:
            args.policy_pretrained_path1 = run_args.policy_pretrained_path1
        if run_args.policy_pretrained_path2 is not None:
            args.policy_pretrained_path2 = run_args.policy_pretrained_path2
        obs_mean, obs_var = train.load_pretrained_weights(args, policy1, policy2, policy_ref)
        for ep_i in range(run_args.n_multiagent_trajs//(args.n_envs1+args.n_envs2)):
            if args.save_traj_data:   # TODO(video)
                # metadata (exp_dir, cmd, etc)
                # pedestrian data t -> {id -> {id, x, y, vx, vy}}
                # vehicle data t -> {id, x, y, th, vlon, accel ,omega, L, W}
                # control slices {id list}
                saved_data = {"exp_dir": args.full_log_dir,
                              "cmd": "python " + " ".join(sys.argv),
                              "reactive_ids1": None,
                              "reactive_ids2": None,
                              "agents1": {}, "agents2": {}, }
            else:
                saved_data = None

            paths = train.collect_samples(ep_i, env, policy1, policy2, policy_ref, args, None, obs_mean, obs_var,
                                          refinement=run_args.refinement,
                                          refinement_cache=refinement_cache,
                                          k_table=k_table, saved_data=saved_data)

            if args.save_traj_data:  # TODO(video)
                np.savez(os.path.join(args.full_log_dir, "traj_data_%d.npz" % (ep_i)), data=saved_data)

            obs = np.concatenate([path["observations"] for path in paths], axis=0)
            actions = np.concatenate([path["actions"] for path in paths], axis=0)
            agent_infos = {}
            env_infos = {}

            for k in paths[0]["agent_infos"]:
                agent_infos[k] = np.concatenate([path["agent_infos"][k] for path in paths], axis=0)
            for k in paths[0]["env_infos"]:
                env_infos[k] = np.concatenate([path["env_infos"][k] for path in paths], axis=0)

            coll_list.append(np.mean(env_infos["is_colliding"]))
            coll1_list.append(np.mean(np.sum(env_infos["is_colliding"].reshape(args.n_envs1+args.n_envs2, args.env_H), axis=1) > 0.5))
            rmse_list.append(np.mean(env_infos["rmse_pos"]))

            print("coll:%.4f(%.4f) traj:%.4f(%.4f) rmse:%.2f(%.2f)"%(
                coll_list[-1], np.mean(coll_list),
                coll1_list[-1], np.mean(coll1_list), rmse_list[-1], np.mean(rmse_list),
            ))

            # write to trajs
            traj = {}
            traj["observations"] = obs.reshape((args.n_envs1+args.n_envs2, args.env_H, -1))
            traj["actions"] = actions.reshape((args.n_envs1+args.n_envs2, args.env_H, -1))
            traj["agent_infos"] = {k:agent_infos[k].reshape((args.n_envs1+args.n_envs2, args.env_H) + agent_infos[k].shape[1:]) for k in agent_infos}
            traj["env_infos"] = {k:env_infos[k].reshape((args.n_envs1+args.n_envs2, args.env_H) + env_infos[k].shape[1:]) for k in env_infos}
            traj_list.append(traj)

        print()
        t2=time.time()
        print("Test: %.4f sec  coll:%.4f  traj:%.4f  rmse:%.4f"%(
            t2-t1, np.mean(coll_list), np.mean(coll1_list), np.mean(rmse_list)
        ))
        with open(os.path.join(target_dir_full, "log.txt"), "w") as f:
            f.write("Test: %.4f sec  coll:%.4f  traj:%.4f  rmse:%.4f\n"%(
            t2-t1, np.mean(coll_list), np.mean(coll1_list), np.mean(rmse_list)
        ))

        target_fname = "trajlist_full.npz"
        np.savez(os.path.join(target_dir_full, target_fname), traj_list=traj_list)

if __name__ == "__main__":
    main()