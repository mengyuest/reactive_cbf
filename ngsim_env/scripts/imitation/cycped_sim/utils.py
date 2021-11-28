import os
import time
from datetime import datetime
from os.path import join as ospj
from sandbox.rocky.tf.envs.base import TfEnv

def get_exp_home():
    return "../../../../train_exps"

def get_exp_name(exp_name):
    return exp_name

def write_cmd_to_file(log_dir, argv):
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
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


def match_for_args_dict(args, test=False):
    debug_args={}
    if args.attractive:
        debug_args["attractive"] = True
    if args.learn_cbf:
        debug_args["learn_cbf"] = True
    if args.lane_control:
        debug_args["lane_control"] = True
    if args.ext_intervals != "":
        debug_args["ext_intervals"] = args.ext_intervals
    if args.cbf_intervals != "":
        debug_args["cbf_intervals"] = args.cbf_intervals
    if args.ctrl_intervals != "":
        debug_args["ctrl_intervals"] = args.ctrl_intervals
    if args.random_seed is not None:
        debug_args["random_seed"] = args.random_seed
    if args.naive_control:
        debug_args["naive_control"] = True
    if args.save_h5:
        debug_args["save_h5"] = True
    if args.debug_extractor:
        debug_args["debug_extractor"] = True
    if args.cbf_ctrl_intervals != "":
        debug_args["cbf_ctrl_intervals"] = args.cbf_ctrl_intervals
    if args.naive_control_clip:
        debug_args["naive_control_clip"] = args.naive_control_clip
    if args.debug_one_step:
        debug_args["debug_one_step"] = True
    if args.fixed_trajectory:
        debug_args["fixed_trajectory"] = True

    if args.use_pedcyc:
        debug_args["num_neighbors"] = args.num_neighbors
        debug_args["ped_radius"] = args.ped_radius
        debug_args["print_gt"] = args.print_gt
        debug_args["record_vxvy"] = args.record_vxvy

        debug_args["traj_idx_list"] = args.traj_idx_list
        debug_args["n_envs1"] = args.n_envs1
        debug_args["n_envs2"] = args.n_envs2
        debug_args["control_mode"] = args.control_mode
        debug_args["fps"] = args.fps

    if args.video_mode is not None:
        debug_args["video_mode"] = args.video_mode
        debug_args["video_traj_idx"] = args.video_traj_idx
        debug_args["video_egoids1"] = args.video_egoids1
        debug_args["video_egoids2"] = args.video_egoids2
        debug_args["video_t"] = args.video_t
        debug_args["video_h"] = args.video_h

    if args.use_gt_trajs is not None:
        debug_args["use_gt_trajs"] = args.use_gt_trajs

    if test:
        if hasattr(args,"action_constraint") and args.action_constraint:
            debug_args["action_constraint"] = args.action_constraint
        if hasattr(args,"speed_constraint") and args.speed_constraint:
            debug_args["speed_constraint"] = args.speed_constraint

    return debug_args

def build_pedcyc_env(args, debug_args, for_test=False):
    import pedcyc_sim

    render_params = dict(
        viz_dir=ospj(args.full_log_dir, '../..', 'imitate/viz') if not for_test else args.full_log_dir+"/viz",
        is_render=args.validator_render,
        n_veh1=args.n_envs1,
        n_veh2=args.n_envs2,
        env_H=args.env_H,
        primesteps=args.env_primesteps
    )
    for key in debug_args:
        if key != "viz_dir":
            render_params[key] = debug_args[key]

    print("viz_dir",render_params["viz_dir"])

    env = pedcyc_sim.PedCycSim(render_params)
    env = TfEnv(env)
    low, high = env.action_space.low, env.action_space.high
    return env, low, high

