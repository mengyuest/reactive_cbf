# Reactive CBF

[![Conference](https://img.shields.io/badge/IROS-Accepted-success)](https://www.iros2021.org/)
   
[![Arxiv](http://img.shields.io/badge/arxiv-cs:2109.06689-B31B1B.svg)](https://arxiv.org/abs/2109.06689.pdf)

This repository contains the most up-to-date code for our IROS2021 paper, "Reactive and Safe Road User Simulations using Neural Barrier Certificates."

```
@article{meng2021reactive,
  title={Reactive and Safe Road User Simulations using Neural Barrier Certificates},
  author={Meng, Yue and Qin, Zengyi and Fan, Chuchu},
  journal={arXiv preprint arXiv:2109.06689},
  year={2021}
}
```

## Prerequisite
Ubuntu 18.04 + NVidia RTX 2080Ti

Get the data files from Google drive: https://drive.google.com/drive/folders/1x_w17Pf6yGX31PGrMBo2RPL8Fd7Ogz0e?usp=sharing

1. Unzip the `data.zip` at the project directory
2. Put the `motion_dict_fps10.npz` into `ngsim_env/pedcyc_env` directory

Install Packages: `conda env create -f environment.yml` and activate the conda environment: `conda activate pg2`

Install local packages:
1. `cd rllab && pip install -e . && cd -`
2. `cd hgail && pip install -e . && cd -`
3. `cd ngsim_env/pedcyc_env && pip install -e . && cd -`




## Running examples (SDD)

In each subsection below, you can just run the `Test` part, or run the `Train` part and replace the model paths ("--model_path", "--exp_name", etc) correspondingly

### PS-GAIL
(under the `ngsim_env/scripts/imitation` directory)
#### Train-ped

``` python
python imitate.py --env_multiagent True --use_infogail False --exp_name pedcyc_psgail_ped --n_itr 1000 --policy_recurrent True --n_envs 5 --ext_intervals 1,34 --expert_filepath ../../../data/pedcyc_ped_only_ego_n8_t50.h5 --validator_render False --random_seed 711 --attractive True --env_H 50 --batch_size 2500 --env_primesteps 1 --use_pedcyc True --record_vxvy True --traj_idx_list 0,1 --control_mode ped_only --n_envs1 5 --n_envs2 0 --fps 10 --gpus 0
```

#### Train-cyc

``` python
python imitate.py --env_multiagent True --use_infogail False --exp_name pedcyc_psgail_cyc --n_itr 1000 --policy_recurrent True --n_envs 5 --ext_intervals 1,34 --expert_filepath ../../../data/pedcyc_cyc_only_ego_n8_t50.h5 --validator_render False --random_seed 711 --attractive True --env_H 50 --batch_size 2500 --env_primesteps 1 --use_pedcyc True --record_vxvy True --traj_idx_list 0,1 --control_mode cyc_only --n_envs1 0 --n_envs2 5 --fps 10 --gpus 0
```
#### Test: 

``` python
python validate.py --n_proc 20 --use_multiagent True --random_seed 3 --exp_dir ../../../data/pedcyc/psgail_ped --params_filename itr_1000.npz --n_envs 10 --n_multiagent_trajs 1000 --debug True --gpus 0 --control_mode ped_cyc --n_envs1 5 --n_envs2 5 --policy1_path ../../../data/pedcyc/psgail_ped/imitate/log/itr_1000.npz --policy2_path ../../../data/pedcyc/psgail_cyc/imitate/log/itr_1000.npz
```

Expected:  `Collision: 0.0170  RSME: 4.1487`

### CBF
(under the `ngsim_env/scripts/imitation/cycped_sim` directory)

#### Train: 

``` python
python train.py --exp_name pedcyc_cbf_model --n_itr 5000 --batch_size 500 --env_H 50 --control_mode ped_cyc --n_envs1 5 --n_envs2 5 --ext_intervals 1,34 --ps_intervals 1,32 --random_seed 711 --cbf_intervals 1,34 --safe_loss_weight 0.5 --dang_loss_weight 0.5 --safe_deriv_loss_weight 0.5 --medium_deriv_loss_weight 0.5 --dang_deriv_loss_weight 0.5 --reg_policy_loss_weight 0.01 --h_safe_thres 0.001 --h_dang_thres 0.05 --grad_safe_thres 0.0 --grad_medium_thres 0.03 --grad_dang_thres 0.08 --safe_dist_threshold 0.3 --dang_dist_threshold 0.15 --joint_learning_rate 0.0005 --jcbf_hidden_layer_dims 128 128 64 --fix_sampling True --use_pedcyc True --env_primesteps 1 --record_vxvy True --save_model_freq 20 --debug_render True --debug_render_freq 500 --use_policy_reference True --include_u_ref_feat True --reg_with_safe_mask True --include_speed_feat True --num_neighbors 8 --enable_radius True --obs_radius 5.0 --dest_controller_type even --print_gt True --traj_idx_list 0,1 --gpus 0
```

#### Test:

``` python
python test.py --exp_dir ../../../../data/pedcyc/cbf_model --params_filename itr_4981.npz --n_multiagent_trajs 1000 --debug_render False --debug_render_freq 10 --gpus 1 --n_envs1 5 --n_envs2 5 --refinement True --refine_n_iter 3 --refine_learning_rate 1500
```

Expected: `Collision:0.0072  RSME:0.6605`

### Behavior Cloning
(under the `ngsim_env/scripts/imitation` directory)

#### Train-ped:
``` python
python bc_tf.py --exp_name pedonlyC --dataset_path ../../../data/pedcyc_ped_only_ego_n8_t50.h5 --batch_size 64 --gpus 0 --print_freq 200 --lr 0.0001 --epochs 50
```

#### Train-cyc

``` python
python bc_tf.py --exp_name cyconlyC --dataset_path ../../../data/pedcyc_ped_only_ego_n8_t50.h5 --batch_size 64 --gpus 0 --print_freq 200 --lr 0.0001 --epochs 50
```

#### Test:

``` python 
python validate.py --n_proc 20 --use_multiagent True --random_seed 3 --exp_dir ../../../data/pedcyc/bc_exp --params_filename itr_1000.npz --n_envs 10 --n_multiagent_trajs 1000 --debug True --gpus 2 --control_mode ped_cyc --n_envs1 5 --n_envs2 5 --behavior_cloning True --bc_policy_path1 ../../../data/pedcyc/bc_ped/itr_49.npz --bc_policy_path2 ../../../data/pedcyc/bc_cyc/itr_49.npz --no_action_scaling True
```
Expected: `Collision: 0.0149 RSME: 12.3568`


### MPC 
> (implemented in ./rllab/sandbox/rocky/tf/samplers/vectorized_sampler.py)

(under the `ngsim_env/scripts/imitation` directory)

#### Test:

``` python
python imitate.py --env_multiagent True --use_infogail False --exp_name pedcyc_mpc_demo --n_itr 100 --policy_recurrent True --n_envs 10 --ext_intervals 1,34 --expert_filepath ../../../data/pedcyc_ped_only_ego_n8_t50.h5 --validator_render False --random_seed 711 --attractive True --env_H 50 --batch_size 500 --env_primesteps 1 --use_pedcyc True --record_vxvy True --traj_idx_list 0,1 --control_mode ped_cyc --n_envs1 5 --n_envs2 5 --fps 10 --gpus 3 --residual_u True --reference_control True --dest_controller_type even --unclip_action True --print_gt True --mpc True --skip_optimize True --skip_baseline_fit True --no_estimate_statistics True --no_obs_normalize True --zero_policy True --quiet True --planning_horizon 5 --consider_uref_init True --mpc_max_iters 20000
```

Expected: `Collision: 0.0196 RMSE: 0.5657`
