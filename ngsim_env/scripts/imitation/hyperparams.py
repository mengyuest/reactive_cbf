'''
default hyperparameters for training
these are build as args to allow for command line options 
these args are also saved along with parameters during training to 
allow for rebuilding everything with the same settings
'''

import argparse
import numpy as np

from utils import str2bool

def parse_args(arglist=None):
    parser = argparse.ArgumentParser()
    # logistics
    parser.add_argument('--exp_name', type=str, default='NGSIM-gail')
    parser.add_argument('--params_filepath', type=str, default='')
    parser.add_argument('--expert_filepath', type=str, default='../../data/trajectories/ngsim.h5')
    parser.add_argument('--vectorize', type=str2bool, default=True)
    parser.add_argument('--n_envs', type=int, default=50)
    parser.add_argument('--normalize_clip_std_multiple', type=float, default=10.)

    # env
    parser.add_argument('--ngsim_filename', type=str, default='trajdata_i101_trajectories-0750am-0805am.txt')
    parser.add_argument('--env_H', type=int, default=200)
    parser.add_argument('--env_primesteps', type=int, default=50)
    parser.add_argument('--env_action_repeat', type=int, default=1)
    parser.add_argument('--env_multiagent', type=str2bool, default=True) # TODO False

    # reward handler
    parser.add_argument('--reward_handler_max_epochs', type=int, default=100)
    parser.add_argument('--reward_handler_recognition_final_scale', type=float, default=.2)

    # policy 
    parser.add_argument('--use_infogail', type=str2bool, default=False)  # TODO True
    parser.add_argument('--policy_mean_hidden_layer_dims', nargs='+', default=(128,128,64))
    parser.add_argument('--policy_std_hidden_layer_dims', nargs='+', default=(128,64))
    parser.add_argument('--policy_recurrent', type=str2bool, default=True)  # TODO False
    parser.add_argument('--recurrent_hidden_dim', type=int, default=64)

    # critic
    parser.add_argument('--use_critic_replay_memory', type=str2bool, default=True)
    parser.add_argument('--n_critic_train_epochs', type=int, default=40)
    parser.add_argument('--critic_learning_rate', type=float, default=.0004)
    parser.add_argument('--critic_dropout_keep_prob', type=float, default=.8)
    parser.add_argument('--gradient_penalty', type=float, default=2.)
    parser.add_argument('--critic_grad_rescale', type=float, default=40.)
    parser.add_argument('--critic_batch_size', type=int, default=1000)
    parser.add_argument('--critic_hidden_layer_dims', nargs='+', default=(128,128,64))

    # recognition
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--n_recognition_train_epochs', type=int, default=30)
    parser.add_argument('--scheduler_k', type=int, default=20)
    parser.add_argument('--recognition_learning_rate', type=float, default=.0005)
    parser.add_argument('--recognition_hidden_layer_dims', nargs='+', default=(128,64))

    # gail
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--trpo_step_size', type=float, default=.01)
    parser.add_argument('--n_itr', type=int, default=2000)
    parser.add_argument('--max_path_length', type=int, default=1000)
    parser.add_argument('--discount', type=float, default=.95)

    # render
    parser.add_argument('--validator_render', type=str2bool, default=False)
    parser.add_argument('--render_every', type=int, default=25)
    parser.add_argument('--remove_ngsim_veh', type=str2bool, default=False)

    #TODO(debug)
    parser.add_argument('--ext_intervals', type=str, default="") # can be something like "1,66"
    parser.add_argument('--affordance', type=str2bool, default=False)
    parser.add_argument('--lane_control', type=str2bool, default=False)

    parser.add_argument('--attention', type=str2bool, default=False)
    parser.add_argument('--learn_cbf', type=str2bool, default=False)
    parser.add_argument('--cbf_intervals', type=str, default="")
    parser.add_argument('--cbf_ctrl_intervals', type=str, default="")
    parser.add_argument('--cbf_batch_size', type=int, default=1000)
    parser.add_argument('--cbf_learning_rate', type=float, default=.0005)
    parser.add_argument('--n_cbf_train_epochs', type=int, default=40)
    parser.add_argument('--cbf_grad_rescale', type=float, default=40)

    parser.add_argument('--safe_loss_weight', type=float, default=1.0)
    parser.add_argument('--dang_loss_weight', type=float, default=1.0)
    parser.add_argument('--safe_deriv_loss_weight', type=float, default=1.0)
    parser.add_argument('--dang_deriv_loss_weight', type=float, default=1.0)
    parser.add_argument('--medium_deriv_loss_weight', type=float, default=3.0)
    parser.add_argument('--reg_policy_loss_weight', type=float, default=1.0)

    parser.add_argument('--multilane_control', type=str2bool, default=False)
    parser.add_argument('--ctrl_intervals', type=str, default="")

    parser.add_argument('--attractive', type=str2bool, default=False)
    parser.add_argument('--normalized_cbf_input', type=str2bool, default=False)

    parser.add_argument('--naive_control', type=str2bool, default=False)

    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--use_fixed_mean_var', type=str2bool, default=False)
    parser.add_argument('--save_h5', type=str2bool, default=False)
    parser.add_argument('--debug_extractor', type=str2bool, default=False)

    parser.add_argument('--joint_cbf', type=str2bool, default=False)
    parser.add_argument('--cbf_iter_per_epoch', type=int, default=0)
    parser.add_argument('--cbf_decay_after', type=int, default=20)
    parser.add_argument('--refine_learning_rate', type=float, default=None)
    parser.add_argument('--refine_n_iter', type=int, default=20)
    parser.add_argument('--refine_policy', type=str2bool, default=False)
    parser.add_argument('--remove_dropout', type=str2bool, default=False)
    parser.add_argument('--naive_control_clip', type=str2bool, default=False)
    parser.add_argument('--aggressive', type=str2bool, default=False)

    parser.add_argument('--safe_dist_threshold', type=float, default=1.0)
    parser.add_argument('--dang_dist_threshold', type=float, default=0.0)
    parser.add_argument('--safe_dist_threshold_side', type=float, default=0.0)
    parser.add_argument('--dang_dist_threshold_side', type=float,default=0.0)

    parser.add_argument('--bbx_collision_check', type=str2bool, default=False)

    parser.add_argument('--jcbf_dropout_keep_prob', type=float, default=.8)  # TODO not used anymore
    parser.add_argument('--jcbf_hidden_layer_dims', type=int, nargs='+', default=(128, 128, 64))
    parser.add_argument('--reg_for_all_control', type=str2bool, default=False)
    parser.add_argument('--full_log_dir', type=str, default=None)

    parser.add_argument('--h_safe_thres', type=float, default=0.001)  # TODO 0.1
    parser.add_argument('--h_dang_thres', type=float, default=0.05)  # TODO 0.1
    parser.add_argument('--grad_safe_thres', type=float, default=0.0)  # TODO 0.1
    parser.add_argument('--grad_dang_thres', type=float, default=0.08)  # TODO 0.1
    parser.add_argument('--grad_medium_thres', type=float, default=0.03)  # TODO 0.1

    parser.add_argument('--debug_refine', type=str2bool, default=False)
    parser.add_argument('--fake_net', type=str2bool, default=False)

    parser.add_argument('--use_mono', type=str2bool, default=False)
    parser.add_argument('--simple_safe_checking', type=str2bool, default=False)
    parser.add_argument('--print_veh_states', type=str2bool, default=False)
    parser.add_argument('--debug_one_step', type=str2bool, default=False)

    parser.add_argument('--safe_def_1', type=str2bool, default=False)
    parser.add_argument('--print_safe_def_1', type=str2bool, default=False)
    parser.add_argument('--heading_tendency', type=float, default=0.2)
    parser.add_argument('--offset_tendency', type=float, default=0.5)

    parser.add_argument('--nc_v1', type=str2bool, default=False) # just decay the omega range [-0.3,0.3]
    parser.add_argument('--nc_v2', type=str2bool, default=False) # adding tolerance to lane offset in range plus [-0.3,0.3]
    parser.add_argument('--nc_v3', type=str2bool, default=False) # when switch lane, larger lane offset

    parser.add_argument('--ref_policy', type=str2bool, default=False)
    parser.add_argument('--joint_learning_rate', type=float, default=None)
    parser.add_argument('--joint_iter_per_epoch', type=int, default=1)  # TODO 100
    parser.add_argument('--joint_decay_after', type=int, default=999999)   # TODO 20
    parser.add_argument('--use_policy_reference', type=str2bool, default=False)
    parser.add_argument('--policy_reference_path', type=str, default=None)
    parser.add_argument('--init_policy_from_scratch', type=str2bool, default=False)
    parser.add_argument('--deterministic_policy_for_cbf', type=str2bool, default=True)  # TODO False

    parser.add_argument('--cbf_pretrained_path', type=str, default=None)
    parser.add_argument('--no_estimate_statistics', type=str2bool, default=False)
    parser.add_argument('--fix_sampling', type=str2bool, default=False)
    parser.add_argument('--unclip_action', type=str2bool, default=False)

    parser.add_argument('--fixed_trajectory', type=str2bool, default=False) # always use the same traj in the simulator (only in NGSIM)

    parser.add_argument('--clip_affordance', type=str2bool, default=False)
    parser.add_argument('--normalize_affordance', type=str2bool, default=False)
    parser.add_argument('--ngsim_filename_list', type=str, default=None)

    parser.add_argument('--accumulation_mode', type=str2bool, default=None)

    parser.add_argument('--use_ped', type=str2bool, default=False)
    parser.add_argument('--traj_idx_list', type=str, default="4,5,6,7,8,9,10")
    parser.add_argument('--num_neighbors', type=int, default=8)
    parser.add_argument('--ped_radius', type=float, default=0.15)
    parser.add_argument('--print_gt', type=str2bool, default=False)
    parser.add_argument('--use_nominal_controller', type=str2bool, default=False)

    parser.add_argument('--save_data', type=str2bool, default=False)           # save the training tuple to files
    parser.add_argument('--joint_for_policy_only', type=str2bool, default=False)
    parser.add_argument('--joint_for_cbf_only', type=str2bool, default=False)  # update only for cbf network
    parser.add_argument('--use_my_cbf', type=str2bool, default=False)          # use mine defined cbf network  # TODO Always-Not used anymore
    parser.add_argument('--use_point_net', type=str2bool, default=False)           # use pointnet struct in my cbf
    parser.add_argument('--fc_then_max', type=str2bool, default=False)
    parser.add_argument('--alternative_update', type=str2bool, default=False)
    parser.add_argument('--alternative_t', type=int, default=10)
    parser.add_argument('--run_both', type=str2bool, default=False)
    parser.add_argument('--residual_u', type=str2bool, default=False)
    parser.add_argument('--zero_policy', type=str2bool, default=False)

    parser.add_argument('--ped_sim_with_veh', type=str2bool, default=False) # ped_sim with vehicles
    parser.add_argument('--veh_mode', type=str, default="points", choices=["points", "radius", "bbox"])
    parser.add_argument('--veh_num_pts', type=int, default=4)
    parser.add_argument('--veh_pts_reso', type=float, default=0.30)
    parser.add_argument('--qp_solve', type=str2bool, default=False)
    parser.add_argument('--qp_alpha', type=float, default=0.01)

    parser.add_argument('--local_data_path', type=str, default=None)
    parser.add_argument('--disable_joint_clip_norm', type=str2bool, default=False)

    parser.add_argument('--ellipse_collision_check', type=str2bool, default=False)
    parser.add_argument('--ellipse_factor', type=float, default=1.0)

    parser.add_argument('--skip_baseline_fit', type=str2bool, default=False)

    parser.add_argument('--use_my_policy', type=str2bool, default=False)
    parser.add_argument('--policy_hidden_layer_dims', type=int, nargs='+', default=(128, 128, 64))
    parser.add_argument('--policy_use_point_net', type=str2bool, default=False)

    parser.add_argument('--new_affordance', type=str2bool, default=False)
    parser.add_argument('--include_action', type=str2bool, default=False)  # TODO only for newaffordance features
    parser.add_argument('--cbf_discrete_num', type=int, default=4)  # TODO only use this for ngsim cbf for now
    parser.add_argument('--qp_accel_weight', type=float, default=1.0)  # TODO only for NGSIM data
    parser.add_argument('--qp_omega_weight', type=float, default=1.0)  # TODO only for NGSIM data

    parser.add_argument('--reg_with_safe_mask', type=str2bool, default=False)

    parser.add_argument('--policy_pretrained_path', type=str, default=None)

    parser.add_argument('--use_easy', type=str2bool, default=False)
    parser.add_argument('--radius', type=float, default=0.5)
    parser.add_argument('--obs_radius', type=float, default=0.5)

    parser.add_argument('--regularizer_weights', type=float, nargs='+', default=(1.0, 1.0))

    parser.add_argument('--high_level', type=str2bool, default=False)  # 0-keeping, 1-changing
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--high_level_policy_hiddens', type=int, nargs='+', default=(64, 8))
    parser.add_argument('--high_level_policy_pretrained_path', type=str, default=None)
    parser.add_argument('--quiet', type=str2bool, default=False)
    parser.add_argument('--print_debug', type=str2bool, default=False)
    parser.add_argument('--debug_loss_weight', type=float, default=0.0)
    parser.add_argument('--debug_accel_only', type=str2bool, default=False)
    parser.add_argument('--debug_render', type=str2bool, default=True)   # TODO True
    parser.add_argument('--debug_render_freq', type=int, default=100)     # TODO 1
    parser.add_argument('--debug_gradually', type=str2bool, default=False)
    parser.add_argument('--debug_gradually_omega', type=str2bool, default=False)

    parser.add_argument('--agent_cbf', type=str2bool, default=False)  # (N,T)->(N,T,6) for masks/scores

    parser.add_argument('--conv1d', type=str2bool, default=False)
    parser.add_argument('--second_fusion', type=str2bool, default=False)
    parser.add_argument('--policy_hidden_fusion_dims', type=int, nargs='+', default=(64, 128, 64))
    parser.add_argument('--always_keeping', type=str2bool, default=False)

    parser.add_argument('--save_model_freq', type=int, default=1)
    parser.add_argument('--debug_simple_cbf', type=str2bool, default=False)
    parser.add_argument('--new_cbf_pol', type=str2bool, default=False)

    parser.add_argument('--no_action_scaling', type=str2bool, default=False)
    parser.add_argument('--no_obs_normalize', type=str2bool, default=False)

    parser.add_argument('--init_with_lcs', type=str2bool, default=False)

    parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument('--record_vxvy', type=str2bool, default=False)


    # TODO for stanford drone dataset
    parser.add_argument('--use_pedcyc', type=str2bool, default=False)
    parser.add_argument('--control_mode', type=str, choices=["ped_cyc", "ped_only", "cyc_only"], default=None)
    parser.add_argument('--n_envs1', type=int, default=5)
    parser.add_argument('--n_envs2', type=int, default=5)
    parser.add_argument('--fps', type=int, default=10)

    # TODO for roundD dataset
    parser.add_argument('--use_round', type=str2bool, default=False)
    parser.add_argument('--without_background', type=str2bool, default=False)
    parser.add_argument('--filtered_background', type=float, default=None)
    parser.add_argument('--figure_ratio', type=float, default=0.2)
    parser.add_argument('--filter_sampling', type=str2bool, default=False)
    parser.add_argument('--init_vmin', type=float, default=5.0)
    parser.add_argument('--init_vmax', type=float, default=12.0)
    parser.add_argument('--plot_circles', type=str2bool, default=False)
    parser.add_argument('--plot_gt_trajectory', type=str2bool, default=False)
    parser.add_argument('--plot_plan_trajectory', type=str2bool, default=False)


    # TODO for highD dataset
    parser.add_argument('--use_high', type=str2bool, default=False)
    parser.add_argument('--neighbor_feature', type=str2bool, default=False)


    # TODO for reference control for PS-GAIL
    parser.add_argument('--dest_controller_type', type=str, default="dest")
    parser.add_argument('--reference_control', type=str2bool, default=False)
    parser.add_argument('--clip_policy_a', type=float, default=None)

    # TODO for behavior cloning baseline
    parser.add_argument('--behavior_cloning', type=str2bool, default=None)

    # TODO for model predictive control with reference control baseline
    parser.add_argument('--mpc', type=str2bool, default=None)
    parser.add_argument('--planning_horizon', type=int, default=10)
    parser.add_argument('--skip_optimize', type=str2bool, default=False)
    parser.add_argument('--zero_mpc', type=str2bool, default=False)

    parser.add_argument('--mpc_max_iters', type=int, default=1000)
    parser.add_argument('--consider_uref_init',type=str2bool, default=False)


    # TODO IDM
    parser.add_argument('--idm_data', type=str2bool, default=False)

    # TODO
    parser.add_argument('--record_time', type=str2bool, default=False)

    # TODO
    parser.add_argument('--save_traj_data', type=str2bool, default=None)
    parser.add_argument('--video_mode', type=str2bool, default=None)
    parser.add_argument('--video_traj_idx', type=int, default=None)
    parser.add_argument('--video_egoids', type=str, default=None)
    parser.add_argument('--video_egoids1', type=str, default=None)
    parser.add_argument('--video_egoids2', type=str, default=None)
    parser.add_argument('--video_t', type=int, default=None)
    parser.add_argument('--video_h', type=int, default=None)

    # parse and return
    if arglist is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arglist)

    return args

def load_args(args_filepath):
    '''
    This function enables backward-compatible usage of saved args files by 
    filling in missing values with default values.
    '''
    orig = np.load(args_filepath)['args'].item()
    new = parse_args(arglist=[])
    orig_keys = set(orig.__dict__.keys())
    new_keys = list(new.__dict__.keys())
    # replace all keys in both orig and new, in new, with orig values
    for k in new_keys:
        if k in orig_keys:
            new.__dict__[k] = orig.__dict__[k]
    return new
