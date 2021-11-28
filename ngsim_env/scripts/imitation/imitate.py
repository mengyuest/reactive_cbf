
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#TODO(yue) non-deterministic
import hyperparams
args = hyperparams.parse_args()
import random
random.seed(args.random_seed)
np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)
import time
from datetime import datetime

from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp, ConjugateGradientOptimizer

from hgail.algos.gail import GAIL

import auto_validator
import utils
from hgail.core.models import StateMLP #TODO

if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


exp_dir = utils.set_up_experiment(exp_name=args.exp_name, phase='imitate')
saver_dir = os.path.join(exp_dir, 'imitate', 'log')
saver_filepath = os.path.join(saver_dir, 'checkpoint')
np.savez(os.path.join(saver_dir, 'args'), args=args)
# summary_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'imitate', 'summaries'))
summary_writer = tf.summary.FileWriter(exp_dir)

args.full_log_dir = saver_dir #TODO(yue)

#TODO save cmdline info
import sys
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
with open(os.path.join(saver_dir, "cmd.txt"),"w") as f:
    f.write(dt_string+"\n")
    seen = set()
    argv_list=[]
    for i, x in enumerate(sys.argv):
        if len(x)>0:
            argv_list.append(x)
        if x.startswith("--"):
            if x not in seen:
                seen.add(x)
            else:
                exit("duplicate flag: %s"%x)
        if x=="--params_filepath":
            if len(sys.argv[i+1])==0:
                argv_list.append("\"\"")
    f.write("python "+" ".join(argv_list))

# build components
debug_args={}  #TODO(debug)
if args.affordance:
    debug_args["affordance"] = True
if args.attention:
    debug_args["attention"] = True
if args.attractive:
    debug_args["attractive"] = True
if args.new_affordance:
    debug_args["new_affordance"] = True
if args.learn_cbf:
    debug_args["learn_cbf"] = True
if args.lane_control:
    debug_args["lane_control"] = True
if args.multilane_control:
    debug_args["multilane_control"] = True
if args.ext_intervals!="":
    debug_args["ext_intervals"]=args.ext_intervals
if args.cbf_intervals!="":
    debug_args["cbf_intervals"]=args.cbf_intervals
if args.ctrl_intervals!="":
    debug_args["ctrl_intervals"]=args.ctrl_intervals
if args.random_seed is not None:
    debug_args["random_seed"] = args.random_seed
if args.naive_control:
    debug_args["naive_control"] = True
if args.save_h5:
    debug_args["save_h5"] = True
if args.debug_extractor:
    debug_args["debug_extractor"] = True
if args.cbf_ctrl_intervals!="":
    debug_args["cbf_ctrl_intervals"] = args.cbf_ctrl_intervals
if args.naive_control_clip:
    debug_args["naive_control_clip"] = args.naive_control_clip
if args.aggressive:
    debug_args["aggressive"] = args.aggressive
if args.print_veh_states:
    debug_args["print_veh_states"] = args.print_veh_states
    debug_args["saver_dir"] = saver_dir

if args.debug_one_step:
    debug_args["debug_one_step"]=True

if args.nc_v1:
    debug_args["nc_v1"] = True
if args.nc_v2:
    debug_args["nc_v2"] = True
if args.nc_v3:
    debug_args["nc_v3"] = True

if args.fixed_trajectory:
    debug_args["fixed_trajectory"] = True

if args.idm_data:
    debug_args["idm_data"] = True



# pedestrian dataset
if args.use_ped:
    debug_args["traj_idx_list"] = args.traj_idx_list
    debug_args["num_neighbors"] = args.num_neighbors
    debug_args["ped_radius"] = args.ped_radius
    debug_args["print_gt"] = args.print_gt
    debug_args["ped_sim_with_veh"] = args.ped_sim_with_veh
    debug_args["record_vxvy"] = args.record_vxvy

    env, act_low, act_high = utils.build_ped_env(args, exp_dir, vectorize=args.vectorize, debug_args=debug_args)
    data = utils.load_data_ped(args.expert_filepath,
                               act_low=act_low, act_high=act_high,
                               min_length = args.env_H + args.env_primesteps,
                               clip_std_multiple=args.normalize_clip_std_multiple,
                               debug_args=debug_args)

elif args.use_pedcyc:
    debug_args["traj_idx_list"] = args.traj_idx_list
    debug_args["num_neighbors"] = args.num_neighbors
    debug_args["ped_radius"] = args.ped_radius
    debug_args["print_gt"] = args.print_gt

    debug_args["control_mode"] = args.control_mode
    debug_args["n_veh1"] = args.n_envs1
    debug_args["n_veh2"] = args.n_envs2
    debug_args["record_vxvy"] = args.record_vxvy
    debug_args["fps"] = args.fps

    env, act_low, act_high = utils.build_pedcyc_env(args, exp_dir, vectorize=args.vectorize, debug_args=debug_args)
    data = utils.load_data_pedcyc(args.expert_filepath,
                               act_low=act_low, act_high=act_high,
                               min_length=args.env_H + args.env_primesteps,
                               clip_std_multiple=args.normalize_clip_std_multiple,
                               debug_args=debug_args)

elif args.use_round:
    debug_args["traj_idx_list"] = args.traj_idx_list
    debug_args["num_neighbors"] = args.num_neighbors
    debug_args["ped_radius"] = args.ped_radius
    debug_args["print_gt"] = args.print_gt
    debug_args["record_vxvy"] = args.record_vxvy
    debug_args["fps"] = args.fps

    debug_args["without_background"] = args.without_background
    debug_args["filtered_background"] = args.filtered_background
    debug_args["figure_ratio"] = args.figure_ratio
    debug_args["filter_sampling"] = args.filter_sampling
    debug_args["init_vmin"] = args.init_vmin
    debug_args["init_vmax"] = args.init_vmax
    debug_args["plot_circles"] = args.plot_circles

    debug_args["plot_gt_trajectory"] = args.plot_gt_trajectory
    debug_args["plot_plan_trajectory"] = args.plot_plan_trajectory

    env, act_low, act_high = utils.build_round_env(args, exp_dir, vectorize=args.vectorize, debug_args=debug_args)
    data = utils.load_data_round(args.expert_filepath,
                                  act_low=act_low, act_high=act_high,
                                  min_length=args.env_H + args.env_primesteps,
                                  clip_std_multiple=args.normalize_clip_std_multiple,
                                  debug_args=debug_args)


elif args.use_high:
    debug_args["traj_idx_list"] = args.traj_idx_list
    debug_args["num_neighbors"] = args.num_neighbors
    debug_args["print_gt"] = args.print_gt
    debug_args["fps"] = args.fps

    debug_args["without_background"] = args.without_background
    debug_args["figure_ratio"] = args.figure_ratio

    debug_args["plot_gt_trajectory"] = args.plot_gt_trajectory
    debug_args["neighbor_feature"] = args.neighbor_feature

    env, act_low, act_high = utils.build_high_env(args, exp_dir, vectorize=args.vectorize, debug_args=debug_args)
    data = utils.load_data_high(args.expert_filepath,
                                 act_low=act_low, act_high=act_high,
                                 min_length=args.env_H + args.env_primesteps,
                                 clip_std_multiple=args.normalize_clip_std_multiple,
                                 debug_args=debug_args)


elif args.use_mono:
    env, act_low, act_high = utils.build_mono_env(args, exp_dir, vectorize=args.vectorize, debug_args=debug_args)
    data = utils.load_data_mono(
        args.expert_filepath,
        act_low=act_low,
        act_high=act_high,
        min_length=args.env_H + args.env_primesteps,
        clip_std_multiple=args.normalize_clip_std_multiple,
        ngsim_filename=args.ngsim_filename,
        debug_args=debug_args  # TODO(debug)
    )

elif args.use_easy:
    debug_args["radius"] = args.radius
    debug_args["obs_radius"] = args.obs_radius
    env, act_low, act_high = utils.build_easy_env(args, exp_dir, vectorize=args.vectorize, debug_args=debug_args)
else:
    env, act_low, act_high = utils.build_ngsim_env(args, exp_dir, vectorize=args.vectorize, debug_args=debug_args)
    data = utils.load_data(
        args.expert_filepath,
        act_low=act_low,
        act_high=act_high,
        min_length=args.env_H + args.env_primesteps,
        clip_std_multiple=args.normalize_clip_std_multiple,
        ngsim_filename=args.ngsim_filename,
        debug_args=debug_args  #TODO(debug)
    )

#TODO(debug)
if args.use_fixed_mean_var:
    raise NotImplementedError

if args.ref_policy:
    critic=None
else:
    critic = utils.build_critic(args, data, env, summary_writer)
policy = utils.build_policy(args, env)
if args.ref_policy:
    policy_as_ref = utils.build_policy(args, env, as_reference=True)

    recognition_model = None
    cbfer = None
    baseline = utils.build_baseline(args, env)
    reward_handler = None
    validator = None
else:
    recognition_model = utils.build_recognition_model(args, env, summary_writer)
    cbfer = None
    baseline = utils.build_baseline(args, env)
    reward_handler = utils.build_reward_handler(args, summary_writer)
    if any([args.use_ped, args.use_mono, args.use_easy, args.use_pedcyc, args.use_round, args.use_high]):
        validator = None
    else:
        validator = auto_validator.AutoValidator(
            summary_writer,
            data['obs_mean'],
            data['obs_std'],
            render=args.validator_render,
            render_every=args.render_every,
            flat_recurrent=args.policy_recurrent
        )

if args.high_level:
    high_level_policy = utils.build_high_level_policy(args, env)

# build algo 
saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=.5)
sampler_args = dict(n_envs=args.n_envs) if args.vectorize else None
if args.policy_recurrent:
    optimizer = ConjugateGradientOptimizer(
        max_backtracks=50,
        hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)
    )
else:
    optimizer = None
algo = GAIL(
    critic=critic,
    recognition=recognition_model,
    reward_handler=reward_handler,
    env=env,
    policy=policy,
    cbfer=cbfer, #TODO(cbfer)
    baseline=baseline,
    validator=validator,
    batch_size=args.batch_size,
    max_path_length=args.max_path_length,
    n_itr=args.n_itr,
    discount=args.discount,
    step_size=args.trpo_step_size,
    saver=saver,
    saver_filepath=saver_filepath,
    force_batch_sampler=False if args.vectorize else True,
    sampler_args=sampler_args,
    snapshot_env=False,
    plot=False,
    optimizer=optimizer,
    optimizer_args=dict(
        max_backtracks=50,
        debug_nan=True
    ),
    args=args,  #TODO(yue)
    network= None,  #TODO(yue)
    policy_as_ref=policy_as_ref if args.use_policy_reference else None,  #TODO(yue)
    high_level_policy=high_level_policy if args.high_level else None,
    summary_writer=summary_writer,  #TODO(yue)
)

# run it
with tf.Session() as session:
    
    # running the initialization here to allow for later loading
    # NOTE: rllab batchpolopt runs this before training as well 
    # this means that any loading subsequent to this is nullified 
    # you have to comment of that initialization for any loading to work
    session.run(tf.global_variables_initializer())

    # loading
    if args.params_filepath != '':
        algo.load(args.params_filepath)
    #TODO(yue)
    if args.policy_reference_path is not None:
        algo.load_ref()
    if args.cbf_pretrained_path is not None:
        algo.load_cbf()
    if args.policy_pretrained_path is not None:
        algo.load_policy()
    if args.high_level_policy_pretrained_path is not None:
        algo.load_high_level_policy()

    summary_writer.add_graph(session.graph)

    # TODO(debug)
    algo._save(-1)

    # run training
    algo.train(sess=session)
