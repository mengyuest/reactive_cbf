
import h5py
import numpy as np
import os
import tensorflow as tf

from rllab.envs.base import EnvSpec
from rllab.envs.normalized_env import normalize as normalize_env
import rllab.misc.logger as logger

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.spaces.discrete import Discrete

from hgail.algos.hgail_impl import Level
from hgail.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from hgail.critic.critic import WassersteinCritic
from hgail.envs.spec_wrapper_env import SpecWrapperEnv
from hgail.envs.vectorized_normalized_env import vectorized_normalized_env
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.policies.categorical_latent_sampler import CategoricalLatentSampler
from hgail.policies.gaussian_latent_var_gru_policy import GaussianLatentVarGRUPolicy
from hgail.policies.gaussian_latent_var_mlp_policy import GaussianLatentVarMLPPolicy
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import ObservationActionMLP
from hgail.policies.scheduling import ConstantIntervalScheduler
from hgail.recognition.recognition_model import RecognitionModel
from hgail.samplers.hierarchy_sampler import HierarchySampler
import hgail.misc.utils

from hgail.cbf.cbf import LearnableCBF
from hgail.core.models import StateMLP


import sandbox.rocky.tf.algos.utils_cbf as utils_cbf

'''
Const
'''
NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': 1,
    'trajdata_i101_trajectories-0805am-0820am.txt': 2,
    'trajdata_i101_trajectories-0820am-0835am.txt': 3,
    'trajdata_i80_trajectories-0400-0415.txt': 4,
    'trajdata_i80_trajectories-0500-0515.txt': 5,
    'trajdata_i80_trajectories-0515-0530.txt': 6
}

'''
Common 
'''
def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def partition_list(lst, n):
    sublists = [[] for _ in range(n)]
    for i, v in enumerate(lst):
        sublists[i % n].append(v)
    return sublists

def str2bool(v):
    if v.lower() == 'true':
        return True
    return False

def write_trajectories(filepath, trajs):
    np.savez(filepath, trajs=trajs)

def load_trajectories(filepath):
    return np.load(filepath)['trajs']

def filename2label(fn):
    s = fn.find('-') + 1
    e = fn.rfind('_')
    return fn[s:e]

def load_trajs_labels(directory, files_to_use=[0,1,2,3,4,5]):
    filenames = [
        'trajdata_i101_trajectories-0750am-0805am_trajectories.npz',
        'trajdata_i101_trajectories-0805am-0820am_trajectories.npz',
        'trajdata_i101_trajectories-0820am-0835am_trajectories.npz',
        'trajdata_i80_trajectories-0400-0415_trajectories.npz',
        'trajdata_i80_trajectories-0500-0515_trajectories.npz',
        'trajdata_i80_trajectories-0515-0530_trajectories.npz'
    ]
    filenames = [filenames[i] for i in files_to_use]
    labels = [filename2label(fn) for fn in filenames]
    filepaths = [os.path.join(directory, fn) for fn in filenames]
    trajs = [load_trajectories(fp) for fp in filepaths]
    return trajs, labels

'''
Component build functions
'''

'''
This is about as hacky as it gets, but I want to avoid editing the rllab 
source code as much as possible, so it will have to do for now.

Add a reset(self, kwargs**) function to the normalizing environment
https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
'''
def normalize_env_reset_with_kwargs(self, **kwargs):
    ret = self._wrapped_env.reset(**kwargs)
    if self._normalize_obs:
        return self._apply_normalize_obs(ret)
    else:
        return ret

def add_kwargs_to_reset(env):
    normalize_env = hgail.misc.utils.extract_normalizing_env(env)
    if normalize_env is not None:
        normalize_env.reset = normalize_env_reset_with_kwargs.__get__(normalize_env)

'''end of hack, back to our regularly scheduled programming'''

def build_ngsim_env(
        args, 
        exp_dir='/tmp', 
        alpha=0.001,
        vectorize=False,
        debug_args=None, #TODO(yue)
        render_params=None):
    from julia_env.julia_env import JuliaEnv  #TODO(debug)
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
    filepaths = [os.path.join(basedir, args.ngsim_filename)]

    #TODO(yue) enable multi traj files
    if args.ngsim_filename_list is not None:
        filepaths=[os.path.join(basedir, x) for x in args.ngsim_filename_list.split(",")]
        logger.log("Using multiple traj files!")
        for x in filepaths:
            logger.log(x)
        assert len(filepaths)<=6
        assert args.ngsim_filename in args.ngsim_filename_list.split(",")

    # TODO
    if args.init_with_lcs:
        debug_args["init_with_lcs"] = True
        lines = None
        lines=lines[1:]
        lcs_id = [int(l.strip().split(" ")[0]) for l in lines]
        lcs_t = [int(l.strip().split(" ")[1]) for l in lines]

        lcs_from = [int(l.strip().split(" ")[2]) for l in lines]
        lcs_to = [int(l.strip().split(" ")[3]) for l in lines]

        debug_args["lcs_id"] = lcs_id
        debug_args["lcs_t"] = lcs_t
        debug_args["lcs_from"] = lcs_from
        debug_args["lcs_to"] = lcs_to

        debug_args['lcs_fixed_id'] = args.lcs_fixed_id

    if render_params is None:
        if debug_args:
            if "notrain" in debug_args:
                render_params = dict(
                    viz_dir=os.path.join(exp_dir, 'imitate/%s'%(debug_args["viz_dir"])),
                    zoom=3.
                )
            else:
                render_params = dict(
                    viz_dir=os.path.join(exp_dir, 'imitate/viz'),
                    zoom=3.)
            for key in debug_args:
                if viz_dir_condition(debug_args["viz_dir"]) or key != "viz_dir":
                    render_params[key] = debug_args[key]
        else:
            render_params = dict(
                viz_dir=os.path.join(exp_dir, 'imitate/viz'),
                zoom=3.
            )
    env_params = dict(
        trajectory_filepaths=filepaths,
        H=args.env_H,
        primesteps=args.env_primesteps,
        action_repeat=args.env_action_repeat,
        terminate_on_collision=False,
        terminate_on_off_road=False,
        render_params=render_params,
        n_envs=args.n_envs,
        n_veh=args.n_envs,
        remove_ngsim_veh=args.remove_ngsim_veh
    )
    # order matters here because multiagent is a subset of vectorized
    # i.e., if you want to run with multiagent = true, then vectorize must 
    # also be true
    if args.env_multiagent:
        env_id = 'MultiagentNGSIMEnv'
        alpha = alpha * args.n_envs
        normalize_wrapper = vectorized_normalized_env


    elif vectorize:
        env_id = 'VectorizedNGSIMEnv'
        alpha = alpha * args.n_envs
        normalize_wrapper = vectorized_normalized_env

    else:
        env_id = 'NGSIMEnv'
        normalize_wrapper = normalize_env

    if args.no_estimate_statistics:
        alpha = 0

    env = JuliaEnv(
        env_id=env_id,
        env_params=env_params,
        using='AutoEnvs'
    )
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    # env = TfEnv(normalize_wrapper(env, normalize_obs=True, obs_alpha=alpha, args=args))
    env = TfEnv(normalize_wrapper(env, normalize_obs=not args.no_obs_normalize, obs_alpha=alpha, args=args))

    add_kwargs_to_reset(env)
    return env, low, high

def build_mono_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=False,
        debug_args=None, #TODO(yue)
        render_params=None):
    import mono_sim

    render_params = dict(viz_dir=os.path.join(exp_dir, 'imitate/viz'), zoom=3.,
                         n_veh=args.n_envs, env_H=args.env_H, primesteps=args.env_primesteps,
                         trajdata_path="None",
                         is_render=args.validator_render)
    for key in debug_args:
        if key != "viz_dir":
            render_params[key] = debug_args[key]

    if args.no_estimate_statistics:
        alpha = 0

    alpha = alpha * args.n_envs
    normalize_wrapper = vectorized_normalized_env

    env = mono_sim.MonoSim(render_params)
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=True, obs_alpha=alpha))
    add_kwargs_to_reset(env)
    return env, low, high

def viz_dir_condition(s):
    return "video" in s

def build_ped_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=False,
        debug_args=None, #TODO(yue)
        render_params=None):
    import ped_sim

    render_params = dict(viz_dir=os.path.join(exp_dir, 'imitate/viz'), is_render=args.validator_render,
                         n_veh=args.n_envs, env_H=args.env_H, primesteps=args.env_primesteps)
    for key in debug_args:

        if viz_dir_condition(debug_args["viz_dir"]) or key != "viz_dir":
            render_params[key] = debug_args[key]

    if args.no_estimate_statistics:
        alpha = 0

    alpha = alpha * args.n_envs
    normalize_wrapper = vectorized_normalized_env

    env = ped_sim.PedSim(render_params)
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=not args.no_obs_normalize, obs_alpha=alpha, args=args))
    add_kwargs_to_reset(env)
    return env, low, high


def build_pedcyc_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=False,
        debug_args=None, #TODO(yue)
        render_params=None):
    import pedcyc_sim

    render_params = dict(viz_dir=os.path.join(exp_dir, 'imitate/viz'), is_render=args.validator_render,
                         n_veh=args.n_envs, env_H=args.env_H, primesteps=args.env_primesteps)
    for key in debug_args:
        # TODO(temp)
        if ("viz_dir" in debug_args and viz_dir_condition(debug_args["viz_dir"])) or key != "viz_dir":
            render_params[key] = debug_args[key]

    if args.no_estimate_statistics:
        alpha = 0

    alpha = alpha * args.n_envs
    normalize_wrapper = vectorized_normalized_env

    env = pedcyc_sim.PedCycSim(render_params)
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=not args.no_obs_normalize, obs_alpha=alpha, args=args))
    add_kwargs_to_reset(env)
    return env, low, high


def build_round_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=False,
        debug_args=None, #TODO(yue)
        render_params=None):


    render_params = dict(viz_dir=os.path.join(exp_dir, 'imitate/viz'), is_render=args.validator_render,
                         n_veh=args.n_envs, env_H=args.env_H, primesteps=args.env_primesteps)
    for key in debug_args:
        if viz_dir_condition(debug_args["viz_dir"]) or key != "viz_dir":
            render_params[key] = debug_args[key]

    if args.no_estimate_statistics:
        alpha = 0

    alpha = alpha * args.n_envs
    normalize_wrapper = vectorized_normalized_env
    # import round_sim
    # env = round_sim.RoundSim(render_params)
    import round_sim_lite
    env = round_sim_lite.RoundSimLite(render_params)
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=not args.no_obs_normalize, obs_alpha=alpha, args=args))
    add_kwargs_to_reset(env)
    return env, low, high


def build_high_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=False,
        debug_args=None, #TODO(yue)
        render_params=None):

    render_params = dict(viz_dir=os.path.join(exp_dir, 'imitate/viz'), is_render=args.validator_render,
                         n_veh=args.n_envs, env_H=args.env_H, primesteps=args.env_primesteps)
    for key in debug_args:
        if viz_dir_condition(debug_args["viz_dir"]) or key != "viz_dir":
            render_params[key] = debug_args[key]

    if args.no_estimate_statistics:
        alpha = 0

    alpha = alpha * args.n_envs
    normalize_wrapper = vectorized_normalized_env
    import high_sim
    env = high_sim.HighSim(render_params)
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=not args.no_obs_normalize, obs_alpha=alpha, args=args))
    add_kwargs_to_reset(env)
    return env, low, high




def build_easy_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=False,
        debug_args=None, #TODO(yue)
        render_params=None):
    import easy_sim

    render_params = dict(viz_dir=os.path.join(exp_dir, 'imitate/viz'), is_render=args.validator_render,
                         n_veh=args.n_envs, env_H=args.env_H, primesteps=args.env_primesteps)
    for key in debug_args:
        if key != "viz_dir":
            render_params[key] = debug_args[key]

    if args.no_estimate_statistics:
        alpha = 0

    alpha = alpha * args.n_envs
    normalize_wrapper = vectorized_normalized_env

    env = easy_sim.EasySim(render_params)
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=True, obs_alpha=alpha))
    add_kwargs_to_reset(env)
    return env, low, high


def build_critic(args, data, env, writer=None):
    if args.use_critic_replay_memory:
        critic_replay_memory = hgail.misc.utils.KeyValueReplayMemory(maxsize=3 * args.batch_size)
    else:
        critic_replay_memory = None

    critic_dataset = CriticDataset(
        data, 
        replay_memory=critic_replay_memory,
        batch_size=args.critic_batch_size,
        flat_recurrent=args.policy_recurrent
    )

    critic_network = ObservationActionMLP(
        name='critic', 
        hidden_layer_dims=args.critic_hidden_layer_dims,
        dropout_keep_prob=args.critic_dropout_keep_prob
    )
    critic = WassersteinCritic(
        obs_dim=env.observation_space.flat_dim,
        act_dim=env.action_space.flat_dim,
        dataset=critic_dataset, 
        network=critic_network,
        gradient_penalty=args.gradient_penalty,
        optimizer=tf.train.RMSPropOptimizer(args.critic_learning_rate),
        n_train_epochs=args.n_critic_train_epochs,
        summary_writer=writer,
        grad_norm_rescale=args.critic_grad_rescale,
        verbose=2,
        debug_nan=True
    )
    return critic


def build_policy(args, env, latent_sampler=None, as_reference=False, is_first=None):

    suffix="_ref" if as_reference else ""
    if is_first is not None:
        if is_first==True:
            suffix="_first"
        else:
            suffix="_second"

    if as_reference == False and args.use_my_policy:
        return utils_cbf.MyPolicy(env, args)
    if args.use_infogail:
        if latent_sampler is None:
            latent_sampler = UniformlyRandomLatentSampler(
                scheduler=ConstantIntervalScheduler(k=args.scheduler_k),
                name='latent_sampler',
                dim=args.latent_dim
            )
        if args.policy_recurrent:
            policy = GaussianLatentVarGRUPolicy(
                name="policy"+suffix,
                latent_sampler=latent_sampler,
                env_spec=env.spec,
                hidden_dim=args.recurrent_hidden_dim,
            )
        else:
            policy = GaussianLatentVarMLPPolicy(
                name="policy"+suffix,
                latent_sampler=latent_sampler,
                env_spec=env.spec,
                hidden_sizes=args.policy_mean_hidden_layer_dims,
                std_hidden_sizes=args.policy_std_hidden_layer_dims
            )
    else:
        if args.policy_recurrent:
            policy = GaussianGRUPolicy(
                name="policy"+suffix,
                env_spec=env.spec,
                hidden_dim=args.recurrent_hidden_dim,
                output_nonlinearity=None,
                learn_std=True,
                args=args,
            )
        else:
            policy = GaussianMLPPolicy(
                name="policy"+suffix,
                env_spec=env.spec,
                hidden_sizes=args.policy_mean_hidden_layer_dims,
                std_hidden_sizes=args.policy_std_hidden_layer_dims,
                adaptive_std=True,
                output_nonlinearity=None,
                learn_std=True,
                args=args,
            )
    return policy

# TODO(yue)
def build_high_level_policy(args, env):
    high_level_policy = utils_cbf.MyPolicyHighLevel(env, args)
    return high_level_policy


def build_recognition_model(args, env, writer=None):
    if args.use_infogail:
        recognition_dataset = RecognitionDataset(
            args.batch_size,
            flat_recurrent=args.policy_recurrent
        )
        recognition_network = ObservationActionMLP(
            name='recog', 
            hidden_layer_dims=args.recognition_hidden_layer_dims,
            output_dim=args.latent_dim
        )
        recognition_model = RecognitionModel(
            obs_dim=env.observation_space.flat_dim,
            act_dim=env.action_space.flat_dim,
            dataset=recognition_dataset, 
            network=recognition_network,
            variable_type='categorical',
            latent_dim=args.latent_dim,
            optimizer=tf.train.AdamOptimizer(args.recognition_learning_rate),
            n_train_epochs=args.n_recognition_train_epochs,
            summary_writer=writer,
            verbose=2
        )
    else:
        recognition_model = None
    return recognition_model

def build_baseline(args, env):
    return GaussianMLPBaseline(env_spec=env.spec)

def build_reward_handler(args, writer=None):
    reward_handler = hgail.misc.utils.RewardHandler(
        use_env_rewards=False,
        max_epochs=args.reward_handler_max_epochs, # epoch at which final scales are used
        critic_final_scale=1.,
        recognition_initial_scale=0.,
        recognition_final_scale=args.reward_handler_recognition_final_scale,
        summary_writer=writer,
        normalize_rewards=True,
        critic_clip_low=-100,
        critic_clip_high=100,
    )
    return reward_handler

def build_hierarchy(args, env, writer=None):
    levels = []

    latent_sampler = UniformlyRandomLatentSampler(
        name='base_latent_sampler',
        dim=args.latent_dim,
        scheduler=ConstantIntervalScheduler(k=args.env_H)
    )
    for level_idx in [1,0]:
        # wrap env in different spec depending on level
        if level_idx == 0:
            level_env = env
        else:
            level_env = SpecWrapperEnv(
                env,
                action_space=Discrete(args.latent_dim),
                observation_space=env.observation_space
            )
            
        with tf.variable_scope('level_{}'.format(level_idx)):
            # recognition_model = build_recognition_model(args, level_env, writer)
            recognition_model = None
            if level_idx == 0:
                policy = build_policy(args, env, latent_sampler=latent_sampler)
            else:
                scheduler = ConstantIntervalScheduler(k=args.scheduler_k)
                policy = latent_sampler = CategoricalLatentSampler(
                    scheduler=scheduler,
                    name='latent_sampler',
                    policy_name='latent_sampler_policy',
                    dim=args.latent_dim,
                    env_spec=level_env.spec,
                    latent_sampler=latent_sampler,
                    max_n_envs=args.n_envs
                )
            baseline = build_baseline(args, level_env)
            if args.vectorize:
                force_batch_sampler = False
                if level_idx == 0:
                    sampler_args = dict(n_envs=args.n_envs)
                else:
                    sampler_args = None
            else:
                force_batch_sampler = True
                sampler_args = None

            sampler_cls = None if level_idx == 0 else HierarchySampler
            algo = TRPO(
                env=level_env,
                policy=policy,
                baseline=baseline,
                batch_size=args.batch_size,
                max_path_length=args.max_path_length,
                n_itr=args.n_itr,
                discount=args.discount,
                step_size=args.trpo_step_size,
                sampler_cls=sampler_cls,
                force_batch_sampler=force_batch_sampler,
                sampler_args=sampler_args,
                optimizer_args=dict(
                    max_backtracks=50,
                    debug_nan=True
                )
            )
            reward_handler = build_reward_handler(args, writer)
            level = Level(
                depth=level_idx,
                algo=algo,
                reward_handler=reward_handler,
                recognition_model=recognition_model,
                start_itr=0,
                end_itr=0 if level_idx == 0 else np.inf
            )
            levels.append(level)

    # by convention the order of the levels should be increasing
    # but they must be built in the reverse order 
    # so reverse the list before returning it
    return list(reversed(levels))

'''
setup
'''

def latest_snapshot(exp_dir, phase='train'):
    snapshot_dir = os.path.join(exp_dir, phase, 'log')
    snapshots = glob.glob('{}/itr_*.pkl'.format(snapshot_dir))
    latest = sorted(snapshots, reverse=True)[0]
    return latest

def get_exp_home():
    return "../../../train_exps"


def set_up_experiment(
        exp_name, 
        phase,
        snapshot_gap=5):
    exp_home = get_exp_home()
    maybe_mkdir(exp_home)
    exp_dir = os.path.join(exp_home, exp_name)
    maybe_mkdir(exp_dir)
    phase_dir = os.path.join(exp_dir, phase)
    maybe_mkdir(phase_dir)
    log_dir = os.path.join(phase_dir, 'log')
    maybe_mkdir(log_dir)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode('gap')
    logger.set_snapshot_gap(snapshot_gap)
    log_filepath = os.path.join(log_dir, 'log.txt')
    logger.add_text_output(log_filepath)

    # TODO(yue)
    logger.add_text_output(os.path.join(exp_dir, 'log.txt'))

    return exp_dir

'''
data utilities
'''

def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return np.array(lengths)

def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(x, axis=0, keepdims=True)
    x = x - mean
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)
    x = x / std
    return x, mean, std

def normalize_range(x, low, high):
    low = np.array(low)
    high = np.array(high)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
    return x

def load_x_feature_names(filepath, ngsim_filename):
    f = h5py.File(filepath, 'r')
    xs = []
    traj_id = NGSIM_FILENAME_TO_ID[ngsim_filename]
    # in case this nees to allow for multiple files in the future
    traj_ids = [traj_id]
    for i in traj_ids:
        if str(i) in f.keys():
            xs.append(f[str(i)])
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    x = np.concatenate(xs)
    feature_names = f.attrs['feature_names']
    return x, feature_names

def load_data(
        filepath,
        act_keys=['accel', 'turn_rate_global'],
        ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt',
        debug_size=None,
        min_length=50,
        normalize_data=True,
        shuffle=False,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf,
        debug_args={}):
    
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names(filepath, ngsim_filename)

    # optionally keep it to a reasonable size
    if debug_size is not None:
        x = x[:debug_size]
       
    if shuffle:
        idxs = np.random.permutation(len(x))
        x = x[idxs]

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i,:l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    # obs = x #TODO(debug)
    #if "affordance" in debug_args or "attention" in debug_args or "attractive" in debug_args:
    if "ext_intervals" in debug_args:
        ext_indices = get_ext_indices(debug_args["ext_intervals"])
        obs = x[:, ext_indices]
    else:
        raise NotImplementedError

    print("critic expert data: shape", obs.shape)

    if "lane_control" in debug_args or "multilane_control" in debug_args or "naive_control" in debug_args:  #TODO(debug)
        act_idxs = [i for (i, n) in enumerate(feature_names) if n in ["residual_accel", "residual_omega"]]
    else:
        act_idxs = [i for (i,n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    #TODO(yue) output features used by each component:
    print_feature_usages(debug_args, feature_names)

    if normalize_data:
        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


#TODO(debug)
def get_ext_indices(ext_intervals):
    intervals = [int(xx) for xx in ext_intervals.split(",")]
    ext_indices = []
    # (a,b) means [a-1,b-1], i.e. [a-1,b)
    for i in range(0, len(intervals), 2):
        i_start = intervals[i]
        i_end = intervals[i + 1]
        for j in range(i_start - 1, i_end):
            ext_indices.append(j)
    return ext_indices


def load_x_feature_names_mono(filepath):
    f = h5py.File(filepath, 'r')
    x = f['0']
    feature_names = f['0'].attrs['feature_names']
    return x, feature_names


def load_data_mono(
        filepath,
        act_keys=['accel'],
        ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt',
        debug_size=None,
        min_length=50,
        normalize_data=True,
        shuffle=False,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf,
        debug_args={}):
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names_mono(filepath)

    # TODO(yue) hardcoded for now
    feature_names = ["ego_v", "fore_x", "fore_v", "rear_x", "rear_v", "accel"]

    # switch from (feat, T, N_veh) to (N_veh, T, feat)
    x = np.swapaxes(x, 0, 2)

    # optionally keep it to a reasonable size
    if debug_size is not None:
        x = x[:debug_size]

    if shuffle:
        idxs = np.random.permutation(len(x))
        x = x[idxs]

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i, :l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    # obs = x #TODO(debug)
    # if "affordance" in debug_args or "attention" in debug_args or "attractive" in debug_args:
    if "ext_intervals" in debug_args:
        ext_indices = get_ext_indices(debug_args["ext_intervals"])
        obs = x[:, ext_indices]
    else:
        raise NotImplementedError

    print("critic expert data: shape", obs.shape)

    print("feat_names",feature_names)

    if "lane_control" in debug_args or "multilane_control" in debug_args or "naive_control" in debug_args:  # TODO(debug)
        act_idxs = [i for (i, n) in enumerate(feature_names) if n in ["residual_accel", "residual_omega"]]
    else:
        act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    # TODO(yue) output features used by each component:
    print_feature_usages(debug_args, feature_names)

    if normalize_data:
        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


def print_feature_usages(debug_args, feature_names):
    # TODO(yue) output features used by each component:
    logger.log("Policy/Critic Features:")
    ext_features = []
    rate = 4
    str_cache = ""
    prefix = "%d %16s\t"
    for i, idx in enumerate(get_ext_indices(debug_args["ext_intervals"])):
        str_cache += prefix % (i, feature_names[idx])
        if i % rate == rate - 1:
            logger.log(str_cache)
            str_cache = ""

        ext_features.append(feature_names[idx])
    if str_cache != "":
        logger.log(str_cache)
        str_cache = ""

    if "ctrl_intervals" in debug_args:
        logger.log("Controller Features:")
        for i, idx in enumerate(get_ext_indices(debug_args["ctrl_intervals"])):
            str_cache += prefix % (i, feature_names[idx])
            if i % rate == rate - 1:
                logger.log(str_cache)
                str_cache = ""
        if str_cache != "":
            logger.log(str_cache)
            str_cache = ""

    if "cbf_intervals" in debug_args:
        logger.log("CBF Features:")
        for i, idx in enumerate(get_ext_indices(debug_args["cbf_intervals"])):
            str_cache += prefix % (i, ext_features[idx])
            if i % rate == rate - 1:
                logger.log(str_cache)
                str_cache = ""
        if str_cache != "":
            logger.log(str_cache)
            str_cache = ""

    if "cbf_ctrl_intervals" in debug_args:
        logger.log("CBF Controller Features:")
        for i, idx in enumerate(get_ext_indices(debug_args["cbf_ctrl_intervals"])):
            str_cache += prefix % (i, ext_features[idx])
            if i % rate == rate - 1:
                logger.log(str_cache)
                str_cache = ""
        if str_cache != "":
            logger.log(str_cache)

def load_x_feature_names_ped(filepath, traj_idx_list):
    f = h5py.File(filepath, 'r')
    xs = []
    for i in traj_idx_list.split(","):
        feature_names = f[i].attrs['feature_names']
        if i in f.keys():
            xs.append(f[i])
            print(xs[-1].shape)
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    x = np.concatenate(xs, axis=-1)

    return x, feature_names

def load_x_feature_names_pedcyc(filepath, traj_idx_list):
    f = h5py.File(filepath, 'r')
    xs = []
    for i in traj_idx_list.split(","):
        feature_names = f[i].attrs['feature_names']
        if i in f.keys():
            xs.append(f[i])
            print(xs[-1].shape)
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    x = np.concatenate(xs, axis=-1)

    return x, feature_names

def load_x_feature_names_round(filepath, traj_idx_list):
    f = h5py.File(filepath, 'r')
    xs = []
    for i in traj_idx_list.split(","):
        feature_names = f[i].attrs['feature_names']
        if i in f.keys():
            xs.append(f[i])
            print(xs[-1].shape)
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    x = np.concatenate(xs, axis=-1)

    return x, feature_names


def load_x_feature_names_high(filepath, traj_idx_list):
    f = h5py.File(filepath, 'r')
    xs = []
    for i in traj_idx_list.split(","):
        feature_names = f[i].attrs['feature_names']
        if i in f.keys():
            xs.append(f[i])
            print(xs[-1].shape)
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    x = np.concatenate(xs, axis=-1)

    return x, feature_names

def load_data_ped(
        filepath,
        act_keys=['gt_ax', "gt_ay"],  # TODO this changed
        min_length=10,  # TODO this changed
        normalize_data=True,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf,
        debug_args={}):
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names_ped(filepath, debug_args["traj_idx_list"])

    # switch from (feat, T, N_veh) to (N_veh, T, feat)
    x = np.swapaxes(x, 0, 2)

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i, :l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    if "ext_intervals" in debug_args:
        ext_indices = get_ext_indices(debug_args["ext_intervals"])
        obs = x[:, ext_indices]
    else:
        raise NotImplementedError

    print("critic expert data: shape", obs.shape)

    act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    print_feature_usages(debug_args, feature_names)

    if normalize_data:
        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )




def load_data_pedcyc(
        filepath,
        act_keys=['gt_ax', "gt_ay"],  # TODO this changed
        min_length=10,  # TODO this changed
        normalize_data=True,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf,
        debug_args={}):
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names_pedcyc(filepath, debug_args["traj_idx_list"])

    # switch from (feat, T, N_veh) to (N_veh, T, feat)
    x = np.swapaxes(x, 0, 2)

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i, :l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    if "ext_intervals" in debug_args:
        ext_indices = get_ext_indices(debug_args["ext_intervals"])
        obs = x[:, ext_indices]
    else:
        raise NotImplementedError

    print("critic expert data: shape", obs.shape)

    # TODO(diff)
    if debug_args["control_mode"]=="ped_only":
        act_keys=["gt_ax", "gt_ay"]
    elif debug_args["control_mode"]=="cyc_only":
        act_keys=["gt_a", "gt_omega"]
    elif debug_args["control_mode"] =="ped_cyc":
        act_keys=["gt_ax", "gt_ay"]
    else:
        raise NotImplementedError

    act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    print_feature_usages(debug_args, feature_names)

    if normalize_data:
        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


def load_data_round(
        filepath,
        act_keys=['gt_a', "gt_omega"],  # TODO this changed
        min_length=10,  # TODO this changed
        normalize_data=True,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf,
        debug_args={}):
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names_round(filepath, debug_args["traj_idx_list"])

    # switch from (feat, T, N_veh) to (N_veh, T, feat)
    x = np.swapaxes(x, 0, 2)

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i, :l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    if "ext_intervals" in debug_args:
        ext_indices = get_ext_indices(debug_args["ext_intervals"])
        obs = x[:, ext_indices]
    else:
        raise NotImplementedError

    print("critic expert data: shape", obs.shape)

    act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    print_feature_usages(debug_args, feature_names)

    if normalize_data:
        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


def load_data_high(
        filepath,
        act_keys=['gt_a', "gt_omega"],  # TODO this changed
        min_length=10,  # TODO this changed
        normalize_data=True,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf,
        debug_args={}):
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names_high(filepath, debug_args["traj_idx_list"])

    # switch from (feat, T, N_veh) to (N_veh, T, feat)
    x = np.swapaxes(x, 0, 2)

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i, :l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    if "ext_intervals" in debug_args:
        ext_indices = get_ext_indices(debug_args["ext_intervals"])
        obs = x[:, ext_indices]
    else:
        raise NotImplementedError

    print("critic expert data: shape", obs.shape)

    act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    print_feature_usages(debug_args, feature_names)

    if normalize_data:
        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
