import pickle

import tensorflow as tf
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools

# TODO(yue)
import sandbox.rocky.tf.algos.utils_cbf as utils_cbf
import casadi

class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

        # TODO(yue) load necessary data for ref controllers. e.g. k_table
        if self.algo.args.residual_u:
            if self.algo.args.reference_control:
                if self.algo.args.use_pedcyc and self.algo.args.control_mode in ["cyc_only","ped_cyc"]:
                    self.k_table = np.load("cycped_sim/K_lqr_v_t0.1000.npz")["table"].item()
                elif self.algo.args.use_round:
                    self.k_table = np.load("round_sim/K_lqr_v_t0.040.npz")["table"].item()
                elif self.algo.args.use_high:
                    self.k_table = np.load("round_sim/K_lqr_v_t0.040.npz")["table"].item()

                if self.algo.args.record_time:
                    self.time_d={"mpc":[], "policy":[], "ref":[]}

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr):
        if self.algo.args.quiet == False:
            logger.log("Obtaining samples for iteration %d..." % itr)

        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        # TODO(yue) extract mean/var for each simulation timestep
        next_obs_mean = np.array(self.vec_env.vec_env._obs_mean)
        next_obs_var = np.array(self.vec_env.vec_env._obs_var)

        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        if self.algo.args.quiet==False:
            pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        # TODO(yue) RESET initial variables
        first_reset=True
        t_idx=0
        if self.algo.args.use_pedcyc:
            if self.algo.args.control_mode in ["cyc_only", "ped_cyc"]:
                cached = [{"pe": 0.0, "pth_e": 0.0} for _ in range(self.algo.args.n_envs2)]
            ctrl_trajs1, ctrl_trajs2 = precompute_trajs_SDD(self.algo.args,
                    self.vec_env.vec_env._wrapped_env.start_state1, self.vec_env.vec_env._wrapped_env.start_state2,
                    self.vec_env.vec_env._wrapped_env.end_state1, self.vec_env.vec_env._wrapped_env.end_state2)
        elif self.algo.args.use_round:
            cached = [{"pe": 0.0, "pth_e": 0.0} for _ in range(self.algo.args.n_envs)]
            ctrl_trajs = self.vec_env.vec_env._wrapped_env.plan_trajs
        elif self.algo.args.use_high:
            direction = self.vec_env.vec_env._wrapped_env.get_direction()
            cached = {"pe": np.zeros((self.algo.args.n_envs, 1)), "pth_e": np.zeros((self.algo.args.n_envs, 1))}
            cached["before_lc"] = np.zeros((self.algo.args.n_envs, 1))
            cached["before_lc"][0, 0] = 1
            cached["in_lc"] = np.zeros((self.algo.args.n_envs, 1))
            cached["direction"] = direction

            prev_state = None



        policy = self.algo.policy
        import time
        while n_samples < self.algo.batch_size:
            t = time.time()
            if self.algo.args.use_my_policy:  # TODO(yue)
                gt_obses = obses * np.sqrt(next_obs_var) + next_obs_mean
                actions, agent_infos = policy.get_actions(gt_obses)
            else:
                policy.reset(dones)
                actions, agent_infos = policy.get_actions(obses)

            # TODO(yue)
            if self.algo.args.residual_u:
                if self.algo.policy_as_ref is not None:
                    self.algo.policy_as_ref.reset(dones)
                    ref_actions, ref_agent_infos = self.algo.policy_as_ref.get_actions(obses)
                    actions = actions + ref_actions
                    agent_infos['ref_actions'] = ref_actions
                    for k in ref_agent_infos:
                        agent_infos["ref_"+k] = ref_agent_infos[k]
                elif self.algo.args.reference_control:
                    if self.algo.args.record_time:
                        ref_t_start=time.time()
                    if self.algo.args.use_ped:  # TODO only uses dest now
                        ref_actions = dest_controller_VCI(self.algo.args,
                              self.vec_env.vec_env._wrapped_env.curr_state, None,
                              self.vec_env.vec_env._wrapped_env.end_state, t_idx)
                        ref_actions = ref_actions / np.array([[4.0, 4.0]])

                    elif self.algo.args.use_pedcyc:  # TODO only uses even for now
                        assert self.algo.args.fps==10
                        if self.algo.args.control_mode == "ped_only":  # TODO tag=1
                            ref_actions = dest_controller_SDD(self.algo.args,
                                                              self.vec_env.vec_env._wrapped_env.curr_state1, ctrl_trajs1,
                                                              self.vec_env.vec_env._wrapped_env.end_state1, t_idx, tag="1",
                                                                     cached=None, k_table=None)
                            ref_actions = ref_actions / np.array([[4.0, 4.0]])

                        # tag=2
                        elif self.algo.args.control_mode == "cyc_only":  # TODO tag=2
                            ref_actions = dest_controller_SDD(self.algo.args,
                                                         self.vec_env.vec_env._wrapped_env.curr_state2, ctrl_trajs2,
                                                         self.vec_env.vec_env._wrapped_env.end_state2, t_idx, tag="2",
                                                             cached=cached, k_table=self.k_table)
                            ref_actions = ref_actions / np.array([[4.0, 0.15]])
                        elif self.algo.args.control_mode == "ped_cyc": # TODO
                            ref_actions1 = dest_controller_SDD(self.algo.args,
                                                              self.vec_env.vec_env._wrapped_env.curr_state1,
                                                              ctrl_trajs1,
                                                              self.vec_env.vec_env._wrapped_env.end_state1, t_idx,
                                                              tag="1",
                                                              cached=None, k_table=None)
                            ref_actions1 = ref_actions1 / np.array([[4.0, 4.0]])

                            ref_actions2 = dest_controller_SDD(self.algo.args,
                                                              self.vec_env.vec_env._wrapped_env.curr_state2,
                                                              ctrl_trajs2,
                                                              self.vec_env.vec_env._wrapped_env.end_state2, t_idx,
                                                              tag="2",
                                                              cached=cached, k_table=self.k_table)
                            ref_actions2 = ref_actions2 / np.array([[4.0, 4.0]])

                            ref_actions = np.concatenate((ref_actions1, ref_actions2), axis=0)
                        else:
                            raise NotImplementedError

                    elif self.algo.args.use_round:
                        assert self.algo.args.fps==25
                        ref_actions = dest_controller_ROUND(self.algo.args,
                                                     self.vec_env.vec_env._wrapped_env.curr_state, ctrl_trajs,
                                                     self.vec_env.vec_env._wrapped_env.end_state, t_idx, cached,
                                                        k_table=self.k_table)
                        ref_actions = ref_actions / np.array([[4.0, 4.0]])

                    elif self.algo.args.use_high:
                        assert self.algo.args.fps == 25
                        assert self.algo.args.cbf_intervals == self.algo.args.ext_intervals =="1,56"
                        state = self.vec_env.vec_env._wrapped_env.obs_tmp[:, 0:56]

                        if t_idx > 30 and cached["before_lc"][0, 0] == 1:  # TODO state: wait->lane-changing
                            cached["before_lc"] = np.zeros((self.algo.args.n_envs, 1))
                            cached["in_lc"] = np.ones((self.algo.args.n_envs, 1))

                        lld_jump_threshold = 3.0

                        if prev_state is not None:
                            lane_changing = np.abs(state[:, -8:-7] - prev_state[:, -8:-7]) > lld_jump_threshold
                        else:
                            lane_changing = np.zeros((state.shape[0], 1))
                        for ni in range(self.algo.args.n_envs):
                            if lane_changing[ni, 0] > 0.5:
                                cached["in_lc"][ni, 0] = 0

                        # reference
                        ref_actions = dest_controller_HIGH(self.algo.args, state, cached, k_table=self.k_table)
                        # print("ref_actions:", ref_actions[0])
                        ref_actions = ref_actions / np.array([[4.0, 0.15]])
                        prev_state = np.array(state)

                    else:
                        raise NotImplementedError

                    if self.algo.args.record_time:
                        ref_t_end=time.time()
                        self.time_d["ref"].append(ref_t_end-ref_t_start)

                    if self.algo.args.mpc:
                        if self.algo.args.zero_mpc:
                            print(self.vec_env.vec_env._wrapped_env.epid, self.vec_env.vec_env._wrapped_env.t, "zero mpc~")
                            mpc_a = 0
                        else:
                            print(self.vec_env.vec_env._wrapped_env.epid, self.vec_env.vec_env._wrapped_env.t, "mpc~")
                            args=self.algo.args
                            mpc_a = mpc_solver(ref_actions, self.vec_env.vec_env._wrapped_env, args)
                            if self.algo.args.use_ped:
                                mpc_a = mpc_a / np.array([[4.0, 4.0]])
                            elif self.algo.args.use_round:
                                mpc_a = mpc_a / np.array([[4.0, 0.15]])
                            elif self.algo.args.use_high:
                                mpc_a = mpc_a / np.array([[4.0, 0.15]])
                            elif self.algo.args.use_pedcyc:
                                mpc_a[:self.algo.args.n_envs1] = mpc_a[:self.algo.args.n_envs1] / np.array([[4.0, 4.0]])
                                mpc_a[self.algo.args.n_envs1:] = mpc_a[self.algo.args.n_envs1:] / np.array([[4.0, 4.0]])
                            else:
                                raise  NotImplementedError
                        ref_actions += mpc_a


                    if self.algo.args.record_time:
                        mpc_t_end=time.time()
                        self.time_d["mpc"].append(mpc_t_end-ref_t_end)

                    if self.algo.args.zero_policy:
                        actions = actions * 0.0 + ref_actions
                    else:
                        if self.algo.args.clip_policy_a is not None:
                            actions = np.clip(actions, -self.algo.args.clip_policy_a, self.algo.args.clip_policy_a) + ref_actions
                        else:
                            actions = actions + ref_actions
                    agent_infos['ref_actions'] = ref_actions


                elif self.algo.args.high_level:
                    # TODO only for NGSIM
                    assert not any([self.algo.args.use_ped, self.algo.args.use_easy, self.algo.args.use_mono,
                                    self.algo.args.use_pedcyc, self.algo.args.use_round, self.algo.args.use_high])
                    assert self.algo.args.use_my_policy
                    choice, choice_infos = self.algo.high_level_policy.get_choice(gt_obses)
                    lk_actions = utils_cbf.lane_keeping_controller(gt_obses, self.algo.args)
                    if self.algo.args.no_action_scaling == False:
                        lk_actions = lk_actions / np.array([[4.0, 0.15]])

                    # debug_n = 5
                    # print("ACT action-a", actions[:debug_n, 0])
                    # print("ACT action-w", actions[:debug_n, 1])

                    # TODO check the shapes of those controlls
                    actions = actions + lk_actions * choice[:, 0:1]   # choice last-dim1=1 means changing, which doesn't need lane-keep
                    # print("ACT action-a", actions[:debug_n, 0])
                    # print("ACT action-w", actions[:debug_n, 1])

                    agent_infos['lk_actions'] = lk_actions
                    agent_infos['tau'] = np.ones_like(lk_actions) * choice_infos['tau']
                    agent_infos['logits'] = choice_infos['logits']
                    agent_infos['choice'] = choice_infos['choice']

                else:
                    # TODO only for use_ped/use_easy
                    assert any([self.algo.args.use_ped, self.algo.args.use_easy, self.algo.args.use_pedcyc,
                                self.algo.args.use_round, self.algo.args.use_high])
                    if first_reset:  # TODO we don't have env_info due to first reset
                        ref_actions = actions * 0.0
                        first_reset = False
                    else:
                        if self.algo.args.no_action_scaling == False:
                            ref_actions = raw_env_infos["controls"] / 4.0

                    # # TODO debug
                    # print("state", obses, "actions", actions, "ref_actions", ref_actions)

                    actions = actions + ref_actions
                    agent_infos['ref_actions'] = ref_actions


            t_idx += 1


            policy_time += time.time() - t
            if all([self.algo.args.reference_control, self.algo.args.mpc, hasattr(self.algo.args, "record_time"), self.algo.args.record_time]):
                self.time_d["policy"].append(time.time() - t)
                logger.log(
                    "profiling: %d | policy:%.4f  ref:%.4f  mpc:%.4f | policy:%.4f  ref:%.4f  mpc:%.4f" % (
                        len(self.time_d["policy"]),
                        np.mean(self.time_d["policy"]), np.mean(self.time_d["ref"]),
                        np.mean(self.time_d["mpc"]),
                        np.sum(self.time_d["policy"]), np.sum(self.time_d["ref"]),
                        np.sum(self.time_d["mpc"]),
                    ), with_prefix=False, with_timestamp=False)

            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            if self.algo.args.debug_render and itr % self.algo.args.debug_render_freq==0 and not any(dones):
                self.vec_env.vec_env._wrapped_env.render()  # TODO(yue)
            raw_env_infos = env_infos  # TODO(yue)

            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                #TODO(yue)
                env_info["obs_mean"] = next_obs_mean
                env_info["obs_var"] = next_obs_var

                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None

                    #TODO(yue) RESET
                    if idx==0:
                        t_idx=0
                        if self.algo.args.use_pedcyc:
                            if self.algo.args.control_mode in ["cyc_only", "ped_cyc"]:
                                cached = [{"pe": 0.0, "pth_e": 0.0} for _ in range(self.algo.args.n_envs2)]
                            ctrl_trajs1, ctrl_trajs2 = precompute_trajs_SDD(self.algo.args,
                                                                        self.vec_env.vec_env._wrapped_env.start_state1,
                                                                        self.vec_env.vec_env._wrapped_env.start_state2,
                                                                        self.vec_env.vec_env._wrapped_env.end_state1,
                                                                        self.vec_env.vec_env._wrapped_env.end_state2)
                        elif self.algo.args.use_round:
                            cached = [{"pe": 0.0, "pth_e": 0.0} for _ in range(self.algo.args.n_envs)]
                            ctrl_trajs = self.vec_env.vec_env._wrapped_env.plan_trajs

                        elif self.algo.args.use_high:
                            direction = self.vec_env.vec_env._wrapped_env.get_direction()
                            cached = {"pe": np.zeros((self.algo.args.n_envs, 1)),
                                      "pth_e": np.zeros((self.algo.args.n_envs, 1))}
                            cached["before_lc"] = np.zeros((self.algo.args.n_envs, 1))
                            cached["before_lc"][0, 0] = 1
                            cached["in_lc"] = np.zeros((self.algo.args.n_envs, 1))
                            cached["direction"] = direction

                            prev_state = None


            process_time += time.time() - t
            if self.algo.args.quiet == False:
                pbar.inc(len(obses))
            obses = next_obses

            # TODO(yue) extract mean/var for each simulation timestep
            next_obs_mean = np.array(self.vec_env.vec_env._obs_mean)
            next_obs_var = np.array(self.vec_env.vec_env._obs_var)

        if self.algo.args.quiet==False:
            pbar.stop()
            logger.record_tabular("PolicyExecTime", policy_time)
            logger.record_tabular("EnvExecTime", env_time)
            logger.record_tabular("ProcessExecTime", process_time)

        return paths



def mpc_solver(uref, env, args):
    import time
    t1=time.time()

    # N = args.n_envs
    T = args.planning_horizon  # planning horizon
    quiet = True
    if args.use_ped:
        # dynamics
        # ax, ay -> vx, vy,
        dt = 1 / 24.0
        obs_r = 0.15
    elif args.use_pedcyc:
        dt = 1 / 10.0
        obs_r = 0.15

    elif args.use_round:
        dt = 1 / 25.0
    elif args.use_high:
        dt = 1 / 25.0
    else:
        raise NotImplementedError
    if args.use_pedcyc:
        mpc_a = np.zeros((args.n_envs1+args.n_envs2, 2))
        uref1 = uref[:args.n_envs1, :]
        uref2 = uref[args.n_envs1:, :]
    else:
        mpc_a=np.zeros((args.n_envs, 2))



    if args.use_ped:
        nei_state = []
        nei_ids = []
        ego_state = []
        ego_ids = env.egoids
        # initial nei list from current scene
        for id in env.traj_data.ped_snapshots[env.t - 1]:
            if id not in env.egoids:
                nei_ids.append(id)
                nei = env.traj_data.ped_snapshots[env.t - 1][id]
                nei_state.append([nei.x, nei.y, nei.vx, nei.vy])
        nei_state=np.array(nei_state)
        nei_state_list = [nei_state]
        for _ in range(T):
            new_nei_state=np.array(nei_state_list[-1])
            new_nei_state[:, 0] += new_nei_state[:, 2] * dt
            new_nei_state[:, 1] += new_nei_state[:, 3] * dt
            nei_state_list.append(new_nei_state)

        # initial ego list from current scene
        for id in ego_ids:
            ego = env.traj_data.ped_snapshots[env.t - 1][id]
            ego_state.append([ego.x, ego.y, ego.vx, ego.vy])
        ego_state = np.array(ego_state)
        ego_state_list = [ego_state]
        for _ in range(T):
            new_ego_state = np.array(ego_state_list[-1])
            new_ego_state[:, 0] += new_ego_state[:, 2] * dt
            new_ego_state[:, 1] += new_ego_state[:, 3] * dt
            if args.consider_uref_init:
                new_ego_state[:, 2] += uref[:, 0] * dt
                new_ego_state[:, 3] += uref[:, 1] * dt
            ego_state_list.append(new_ego_state)
    elif args.use_round:
        nei_state = []
        nei_ids = []
        ego_state = []
        ego_ids = env.egoids
        # initial nei list from current scene
        for id in env.traj_data.snapshots[env.t - 1]:
            if id not in env.egoids:
                nei_ids.append(id)
                nei = env.traj_data.snapshots[env.t - 1][id]
                nei_state.append([nei.x, nei.y, nei.heading, nei.v_lon])
        nei_state = np.array(nei_state)
        nei_state_list = [nei_state]
        for _ in range(T):
            new_nei_state = np.array(nei_state_list[-1])
            new_nei_state[:, 0] += new_nei_state[:, 3] * np.cos(new_nei_state[:, 2]) * dt
            new_nei_state[:, 1] += new_nei_state[:, 3] * np.sin(new_nei_state[:, 2]) * dt
            nei_state_list.append(new_nei_state)

        # initial ego list from current scene
        for i,id in enumerate(ego_ids):
            ego = env.ego_vehs[i]
            ego_state.append([ego.x, ego.y, ego.heading, ego.v_lon])
        ego_state = np.array(ego_state)
        ego_state_list = [ego_state]
        for _ in range(T):
            new_ego_state = np.array(ego_state_list[-1])
            new_ego_state[:, 0] += new_ego_state[:, 3] * np.cos(new_ego_state[:, 2]) * dt
            new_ego_state[:, 1] += new_ego_state[:, 3] * np.sin(new_ego_state[:, 2]) * dt
            if args.consider_uref_init:
                new_ego_state[:, 2] += uref[:, 0] * dt
                new_ego_state[:, 3] += uref[:, 1] * dt
            ego_state_list.append(new_ego_state)

    elif args.use_high:
        nei_state = []
        nei_ids = []
        ego_state = []
        ego_ids = env.egoids
        # initial nei list from current scene
        for id in env.traj_data.snapshots_upper[env.t - 1]:
            if id not in env.egoids:
                nei_ids.append(id)
                nei = env.traj_data.snapshots_upper[env.t - 1][id]
                nei_state.append([nei.x, nei.y, nei.heading, nei.v_lon])
        nei_state = np.array(nei_state)
        nei_state_list = [nei_state]
        for _ in range(T):
            new_nei_state = np.array(nei_state_list[-1])
            new_nei_state[:, 0] += new_nei_state[:, 3] * np.cos(new_nei_state[:, 2]) * dt
            new_nei_state[:, 1] += new_nei_state[:, 3] * np.sin(new_nei_state[:, 2]) * dt
            nei_state_list.append(new_nei_state)

        # initial ego list from current scene
        for i, id in enumerate(ego_ids):
            ego = env.ego_vehs[i]
            ego_state.append([ego.x, ego.y, ego.heading, ego.v_lon])
        ego_state = np.array(ego_state)
        ego_state_list = [ego_state]
        for _ in range(T):
            new_ego_state = np.array(ego_state_list[-1])
            new_ego_state[:, 0] += new_ego_state[:, 3] * np.cos(new_ego_state[:, 2]) * dt
            new_ego_state[:, 1] += new_ego_state[:, 3] * np.sin(new_ego_state[:, 2]) * dt
            if args.consider_uref_init:
                new_ego_state[:, 2] += uref[:, 0] * dt
                new_ego_state[:, 3] += uref[:, 1] * dt
            ego_state_list.append(new_ego_state)

    elif args.use_pedcyc:
        nei_state1 = []
        nei_ids1 = []
        ego_state1 = []
        ego_ids1 = env.egoids1

        nei_state2 = []
        nei_ids2 = []
        ego_state2 = []
        ego_ids2 = env.egoids2

        # initial nei list from current scene
        for id in env.traj_data["Pedestrian"]["snapshots"][env.t - 1]:
            if id not in env.egoids1:
                nei_ids1.append(id)
                nei = env.traj_data["Pedestrian"]["snapshots"][env.t - 1][id]
                nei_state1.append(nei[:4])  # x, y, vx, vy
        nei_state1=np.array(nei_state1)
        nei_state_list1 = [nei_state1]
        for _ in range(T):
            new_nei_state1=np.array(nei_state_list1[-1])
            new_nei_state1[:, 0] += new_nei_state1[:, 2] * dt
            new_nei_state1[:, 1] += new_nei_state1[:, 3] * dt
            nei_state_list1.append(new_nei_state1)

        for id in env.traj_data["Biker"]["snapshots"][env.t - 1]:
            if id not in env.egoids2:
                nei_ids2.append(id)
                nei = env.traj_data["Biker"]["snapshots"][env.t - 1][id]
                nei_state2.append(nei[:4])  # x, y, th, v_lon
        nei_state2 = np.array(nei_state2)
        nei_state_list2 = [nei_state2]
        for _ in range(T):
            new_nei_state2=np.array(nei_state_list2[-1])
            new_nei_state2[:, 0] += new_nei_state2[:, 2] * dt
            new_nei_state2[:, 1] += new_nei_state2[:, 3] * dt
            nei_state_list2.append(new_nei_state2)

        # initial ego list from current scene
        for ego in env.ego_peds1:
            ego_state1.append(ego[0:4])
        ego_state1 = np.array(ego_state1)
        ego_state_list1 = [ego_state1]
        for _ in range(T):
            new_ego_state1 = np.array(ego_state_list1[-1])
            new_ego_state1[:, 0] += new_ego_state1[:, 2] * dt
            new_ego_state1[:, 1] += new_ego_state1[:, 3] * dt
            if args.consider_uref_init:
                new_ego_state1[:, 2] += uref1[:, 0] * dt
                new_ego_state1[:, 3] += uref1[:, 1] * dt
            ego_state_list1.append(new_ego_state1)

        # initial ego list from current scene
        for ego in env.ego_peds2:
            ego_state2.append(ego[0:4])
        ego_state2 = np.array(ego_state2)
        ego_state_list2 = [ego_state2]
        for _ in range(T):
            new_ego_state2 = np.array(ego_state_list2[-1])
            new_ego_state2[:, 0] += new_ego_state2[:, 3] * np.cos(new_ego_state2[:, 2]) * dt
            new_ego_state2[:, 1] += new_ego_state2[:, 3] * np.sin(new_ego_state2[:, 2]) * dt
            if args.consider_uref_init:
                new_ego_state2[:, 2] += uref2[:, 0] * dt
                new_ego_state2[:, 3] += uref2[:, 1] * dt
            ego_state_list2.append(new_ego_state2)
    else:
        raise NotImplementedError



    if args.use_pedcyc:
        all_states1 = np.concatenate((np.array(ego_state_list1), np.array(nei_state_list1)), axis=1)
        all_states2 = np.concatenate((np.array(ego_state_list2), np.array(nei_state_list2)), axis=1)
        rev_d1={}  # revsere look up
        for i,id in enumerate(list(ego_ids1)+list(nei_ids1)):
            rev_d1[id]=i

        rev_d2={}  # revsere look up
        for i,id in enumerate(list(ego_ids2)+list(nei_ids2)):
            rev_d2[id]=i
        n_envs = args.n_envs1 + args.n_envs2
    else:
        all_states = np.concatenate((np.array(ego_state_list), np.array(nei_state_list)), axis=1)
        rev_d = {}  # revsere look up
        for i, id in enumerate(list(ego_ids) + list(nei_ids)):
            rev_d[id] = i
        n_envs = args.n_envs


    t2 = time.time()
    sum_t3=0
    sum_t4=0

    sel_idx=-1

    if args.use_pedcyc:
        for agi in range(args.n_envs1): # pedestrian case
            t3 = time.time()

            x_ref1 = ego_state_list1[-1][agi, 0]
            y_ref1 = ego_state_list1[-1][agi, 1]

            NM = 10  # 8 neighbors for pedcyc
            if agi==sel_idx:
                print("id=%d x:%.4f y:%.4f vx:%.4f vy:%.4f   -> ref_x:%.4f ref_y:%.4f  ref_ax:%.4f ref_ay:%.4f numrx:%.4f numry:%.4f" % (
                    agi, ego_state_list1[0][agi, 0], ego_state_list1[0][agi, 1], ego_state_list1[0][agi, 2], ego_state_list1[0][agi, 3],
                    x_ref1, y_ref1, uref1[agi, 0], uref1[agi, 1], new_ego_state1[agi, 0], new_ego_state1[agi, 1]))

            opti = casadi.Opti()
            x = opti.variable(T + 1, 4)  # state   (x,y,vx,vy) or (x,y, th, v)
            u = opti.variable(T, 2)  # control (ax, ay)    or (accel, omega)
            gamma = opti.variable(T, NM - 1)

            opti.minimize(
                casadi.sumsqr(x[T, 0] - x_ref1) + casadi.sumsqr(x[T, 1] - y_ref1) + 100 * casadi.sumsqr(gamma)
            )

            # initial condition
            opti.subject_to(x[0, 0] == ego_state_list1[0][agi, 0])
            opti.subject_to(x[0, 1] == ego_state_list1[0][agi, 1])
            opti.subject_to(x[0, 2] == ego_state_list1[0][agi, 2])
            opti.subject_to(x[0, 3] == ego_state_list1[0][agi, 3])

            # boxing constraints
            opti.subject_to(u[:, 0] <= 4)  # ax
            opti.subject_to(u[:, 0] >= -4)  # ax
            opti.subject_to(u[:, 1] <= 4)  # ay
            opti.subject_to(u[:, 1] >= -4)  # ay

            # dynamics
            for k in range(T):  # timesteps:
                opti.subject_to(x[k + 1, 0] == x[k, 0] + x[k, 2] * dt)  # x+=vx*dt
                opti.subject_to(x[k + 1, 1] == x[k, 1] + x[k, 3] * dt)  # y+=vy*dt
                opti.subject_to(x[k + 1, 2] == x[k, 2] + (u[k, 0] + uref1[agi, 0]) * dt)  # vx+=ax*dt
                opti.subject_to(x[k + 1, 3] == x[k, 3] + (u[k, 1] + uref1[agi, 1]) * dt)  # vy+=ay*dt


            # collision avoidance constraints
            for k in range(T):  # timesteps
                idx = 0
                if env.infos is None:
                    continue
                for ii, i in enumerate(env.infos["neighbors"][agi]):
                    if env.infos["neighbors_modes"][agi][ii] == 'ped':
                        opti.subject_to(
                            ((x[k + 1, 0] - all_states1[k + 1][rev_d1[i], 0]) / (2 * obs_r)) ** 2 +
                            ((x[k + 1, 1] - all_states1[k + 1][rev_d1[i], 1]) / (2 * obs_r)) ** 2 + gamma[k, idx] >= 1)
                    elif env.infos["neighbors_modes"][agi][ii] == 'cyc':
                        opti.subject_to(
                            ((x[k + 1, 0] - all_states2[k + 1][rev_d2[i], 0]) / (2 * obs_r)) ** 2 +
                            ((x[k + 1, 1] - all_states2[k + 1][rev_d2[i], 1]) / (2 * obs_r)) ** 2 + gamma[k, idx] >= 1)
                    else:
                        raise NotImplementedError
                    idx += 1


            t4 = time.time()
            # optimizer setting
            p_opts = {"expand": True}
            s_opts = {"max_iter": 1000, "tol":1e-6}
            if quiet:
                p_opts["print_time"] = 0
                s_opts["print_level"] = 0
                s_opts["sb"] = "yes"
            opti.solver("ipopt", p_opts, s_opts)
            sol1 = opti.solve()
            # opti.solve()
            # sol1 = opti.debug
            mpc_a[agi, :] = sol1.value(u)[0, :]
            t5 = time.time()
            sum_t3 += t4 - t3
            sum_t4 += t5 - t4
            if agi==sel_idx:
                print(agi, env.egoids1[agi], mpc_a[agi, :])
                print(sol1.value(x))

        for agi in range(args.n_envs2):  # pedestrian case
            t3 = time.time()

            x_ref2 = ego_state_list2[-1][agi, 0]
            y_ref2 = ego_state_list2[-1][agi, 1]

            NM = 10  # 8 neighbors for pedcyc

            opti = casadi.Opti()
            x = opti.variable(T + 1, 4)  # state   (x,y,vx,vy) or (x,y, th, v)
            u = opti.variable(T, 2)  # control (ax, ay)    or (accel, omega)
            gamma = opti.variable(T, NM - 1)

            opti.minimize(
                casadi.sumsqr(x[T, 0] - x_ref2) + casadi.sumsqr(x[T, 1] - y_ref2) + 100 * casadi.sumsqr(gamma)
            )

            # initial condition
            opti.subject_to(x[0, 0] == ego_state_list2[0][agi, 0])
            opti.subject_to(x[0, 1] == ego_state_list2[0][agi, 1])
            opti.subject_to(x[0, 2] == ego_state_list2[0][agi, 2])
            opti.subject_to(x[0, 3] == ego_state_list2[0][agi, 3])

            # boxing constraints
            opti.subject_to(u[:, 0] <= 4)    # accel
            opti.subject_to(u[:, 0] >= -4)   # accel
            opti.subject_to(u[:, 1] <= .15)  # omega
            opti.subject_to(u[:, 1] >= -.15) # omega

            # dynamics
            for k in range(T):  # timesteps:
                opti.subject_to(x[k + 1, 0] == x[k, 0] + x[k, 3] * np.cos(x[k, 2]) * dt)  # x+=vx*dt
                opti.subject_to(x[k + 1, 1] == x[k, 1] + x[k, 3] * np.sin(x[k, 2]) * dt)  # y+=vy*dt
                opti.subject_to(x[k + 1, 2] == x[k, 2] + (u[k, 0] + uref2[agi, 0]) * dt)  # vx+=ax*dt
                opti.subject_to(x[k + 1, 3] == x[k, 3] + (u[k, 1] + uref2[agi, 1]) * dt)  # vy+=ay*dt

            # collision avoidance constraints
            for k in range(T):  # timesteps
                idx = 0
                if env.infos is None:
                    continue
                for ii, i in enumerate(env.infos["neighbors"][agi+args.n_envs1]):
                    if env.infos["neighbors_modes"][agi+args.n_envs1][ii] == 'ped':
                        opti.subject_to(
                            ((x[k + 1, 0] - all_states1[k + 1][rev_d1[i], 0]) / (2 * obs_r)) ** 2 +
                            ((x[k + 1, 1] - all_states1[k + 1][rev_d1[i], 1]) / (2 * obs_r)) ** 2 + gamma[
                                k, idx] >= 1)
                    elif env.infos["neighbors_modes"][agi+args.n_envs1][ii] == 'cyc':
                        opti.subject_to(
                            ((x[k + 1, 0] - all_states2[k + 1][rev_d2[i], 0]) / (2 * obs_r)) ** 2 +
                            ((x[k + 1, 1] - all_states2[k + 1][rev_d2[i], 1]) / (2 * obs_r)) ** 2 + gamma[
                                k, idx] >= 1)
                    else:
                        raise NotImplementedError
                    idx += 1

            t4 = time.time()
            # optimizer setting
            p_opts = {"expand": True}
            s_opts = {"max_iter": args.mpc_max_iters, "tol":1e-6}
            if quiet:
                p_opts["print_time"] = 0
                s_opts["print_level"] = 0
                s_opts["sb"] = "yes"
            opti.solver("ipopt", p_opts, s_opts)
            sol1 = opti.solve()
            # opti.solve()
            # sol1 = opti.debug
            mpc_a[agi+args.n_envs1, :] = sol1.value(u)[0, :]
            t5 = time.time()
            sum_t3 += t4 - t3
            sum_t4 += t5 - t4
    else:
        for agi in range(n_envs):
            t3=time.time()

            x_ref = ego_state_list[-1][agi, 0]
            y_ref = ego_state_list[-1][agi, 1]

            # print("id=%d x:%.4f y:%.4f vx:%.4f vy:%.4f   -> ref_x:%.4f ref_y:%.4f  ref_ax:%.4f ref_ay:%.4f numrx:%.4f numry:%.4f" % (
            #     agi, x, y, vx, vy, x_ref, y_ref, uref[agi, 0], uref[agi, 1], new_ego_state[agi, 0], new_ego_state[agi, 1]))

            NM = all_states.shape[1]

            opti = casadi.Opti()
            x = opti.variable(T+1, 4)  # state   (x,y,vx,vy) or (x,y, th, v)
            u = opti.variable(T, 2)    # control (ax, ay)    or (accel, omega)
            gamma = opti.variable(T, NM-1)

            opti.minimize(
                casadi.sumsqr(x[T, 0] - x_ref) + casadi.sumsqr(x[T, 1] - y_ref) + 100 * casadi.sumsqr(gamma)
            )

            # initial condition
            opti.subject_to(x[0, 0] == ego_state_list[0][agi, 0])
            opti.subject_to(x[0, 1] == ego_state_list[0][agi, 1])
            opti.subject_to(x[0, 2] == ego_state_list[0][agi, 2])
            opti.subject_to(x[0, 3] == ego_state_list[0][agi, 3])

            # boxing constraints
            if args.use_ped or args.use_pedcyc:
                opti.subject_to(u[:, 0] <= 4)  # ax
                opti.subject_to(u[:, 0] >= -4)  # ax
                opti.subject_to(u[:, 1] <= 4)  # ay
                opti.subject_to(u[:, 1] >= -4)  # ay
            else:
                opti.subject_to(u[:, 0] <= 4)  # accel
                opti.subject_to(u[:, 0] >= -4)  # accel
                opti.subject_to(u[:, 1] <= .15)  # omega
                opti.subject_to(u[:, 1] >= -.15)  # omega

            # dynamics
            for k in range(T):  # timesteps
                if args.use_ped or args.use_pedcyc:
                    opti.subject_to(x[k + 1, 0] == x[k, 0] + x[k, 2] * dt)  # x+=vx*dt
                    opti.subject_to(x[k + 1, 1] == x[k, 1] + x[k, 3] * dt)  # y+=vy*dt
                    opti.subject_to(x[k + 1, 2] == x[k, 2] + (u[k, 0]+uref[agi,0]) * dt)  # vx+=ax*dt
                    opti.subject_to(x[k + 1, 3] == x[k, 3] + (u[k, 1]+uref[agi,1]) * dt)  # vy+=ay*dt
                else:
                    opti.subject_to(x[k + 1, 0] == x[k, 0] + x[k, 3] * casadi.cos(x[k, 2]) * dt)  # x+=v*cos(theta)*dt
                    opti.subject_to(x[k + 1, 1] == x[k, 1] + x[k, 3] * casadi.sin(x[k, 2]) * dt)  # y+=v*sin(theta)*dt
                    opti.subject_to(x[k + 1, 2] == x[k, 2] + u[k, 1] * dt)  # theta+=omega*dt
                    opti.subject_to(x[k + 1, 3] == x[k, 3] + u[k, 0] * dt)  # v+=a*dt

            # collision avoidance constraints
            if args.use_ped:
                for k in range(T):  # timesteps
                    idx=0
                    if env.infos is None:
                        continue
                    for i in env.infos["neighbors"][agi]:
                        opti.subject_to(
                            ((x[k + 1, 0] - all_states[k + 1][rev_d[i], 0]) / (2 * obs_r)) ** 2 +
                            ((x[k + 1, 1] - all_states[k + 1][rev_d[i], 1]) / (2 * obs_r)) ** 2 + gamma[k, idx] >= 1)
                        idx+=1
            elif args.use_round:
                for k in range(T):  # timesteps
                    idx=0
                    if env.infos is None:
                        continue
                    for i in env.infos["neighbors"][agi]:
                        r1 = ((env.ego_vehs[agi].length/2)**2 + (env.ego_vehs[agi].width/2)**2)**0.5
                        r2 = ((env.traj_data.snapshots[env.t-1][i].length/2)**2 + (env.traj_data.snapshots[env.t-1][i].width/2)**2)**0.5
                        opti.subject_to(
                            ((x[k + 1, 0] - all_states[k + 1][rev_d[i], 0]) / (r1 + r2)) ** 2 +
                            ((x[k + 1, 1] - all_states[k + 1][rev_d[i], 1]) / (r1 + r2)) ** 2 + gamma[k, idx] >= 1)
                        idx+=1
            elif args.use_high:
                for k in range(T):  # timesteps
                    idx=0
                    if env.infos is None:
                        continue
                    for i in env.infos["neighbors"][agi]:
                        if i==-1:
                            continue
                        l1 = env.ego_vehs[agi].length/2
                        w1 = env.ego_vehs[agi].width/2
                        l2 = env.traj_data.snapshots_upper[env.t - 1][i].length / 2
                        w2 = env.traj_data.snapshots_upper[env.t - 1][i].width / 2
                        opti.subject_to(
                            ((x[k + 1, 0] - all_states[k + 1][rev_d[i], 0]) / (l1 + l2)) ** 2 +
                            ((x[k + 1, 1] - all_states[k + 1][rev_d[i], 1]) / (w1 + w2)) ** 2 + gamma[k, idx] >= 1)
                        idx+=1
            else:
                raise NotImplementedError
            t4 = time.time()
            # optimizer setting
            p_opts = {"expand": True}
            s_opts = {"max_iter": 1000}
            if quiet:
                p_opts["print_time"] = 0
                s_opts["print_level"] = 0
                s_opts["sb"] = "yes"
            opti.solver("ipopt", p_opts, s_opts)
            sol1 = opti.solve()
            mpc_a[agi, :] = sol1.value(u)[0, :]
            t5 = time.time()
            sum_t3 += t4-t3
            sum_t4 += t5-t4
        # print("control: ax: %.4f  ay:%.4f"%(mpc_a[agi, 0], mpc_a[agi, 1]))
        # print("points %s" % (sol1.value(x)))

    # print("%.6f %.6f %.6f"%(t2-t1, sum_t3, sum_t4))

    return mpc_a




def dest_controller_VCI(args, s, trajs, dest, t):
    # s    (n_envs, 4) x,y,vx,vy
    # trajs(n_envs, T, 4) x,y,vx,vy in each timestep
    # dest (n_envs, 4) x,y,vx,vy
    # t    current time, also should consider total time left
    # uref (n_envs, 2) ax, ay

    dt = 1.0 / 24

    xt=s[:, 0]
    yt=s[:, 1]
    vxt=s[:, 2]
    vyt=s[:, 3]

    xn=dest[:, 0]
    yn=dest[:, 1]

    left_t = (args.env_H - t) * dt

    if args.dest_controller_type=="dest":
        if left_t==0:
            uref = np.zeros((args.n_envs, 2))
        else:
            ax = 2 * (xn - xt - vxt * left_t) / left_t / left_t
            ay = 2 * (yn - yt - vyt * left_t) / left_t / left_t
            uref = np.stack((ax,ay), axis=-1)

        # idx=3
        # print("t=%d %d left%.4f x:%.4f y:%.4f xn:%.4f yn:%.4f vx:%.4f vy:%.4f ax:%.4f ay:%.4f"%(
        #     t, idx, left_t, xt[idx], yt[idx], xn[idx], yn[idx], vxt[idx], vyt[idx], ax[idx], ay[idx]
        # ))
    else:
        raise NotImplementedError

    return uref


def precompute_trajs_SDD(args, s1, s2, dest1, dest2):
    # s    (n_envs, 4) x,y,vx,vy
    # dest (n_envs, 4) x,y,vx,vy
    # trajs(n_envs, T, 4) x,y,vx,vy in each timestep

    dt=1.0/args.fps
    T = args.env_H * dt

    trajs1 = np.zeros((args.n_envs1, args.env_H, 4))
    vx1=(dest1[:,0]-s1[:,0])/T
    vy1=(dest1[:,1]-s1[:,1])/T

    if args.n_envs1>0:
        for t in range(args.env_H):
            trajs1[:, t, 0] = s1[:, 0] + vx1 * (t+1) * dt
            trajs1[:, t, 1] = s1[:, 1] + vy1 * (t+1) * dt
            trajs1[:, t, 2] = vx1
            trajs1[:, t, 3] = vy1

    trajs2 = np.zeros((args.n_envs2, args.env_H, 4))

    if args.n_envs2>0:
        curr_s = np.array(s2)
        for t in range(0, args.env_H):
            l2 = ((dest2[:, 0] - curr_s[:, 0])**2 + (dest2[:, 1] - curr_s[:, 1])**2)**0.5
            v2 = l2 / (args.env_H-t)/dt

            new_sx = curr_s[:, 0] + curr_s[:, 3] * np.cos(curr_s[:, 2]) * dt
            new_sy = curr_s[:, 1] + curr_s[:, 3] * np.sin(curr_s[:, 2]) * dt
            new_th = np.arctan2(dest2[:,1] - curr_s[:,1], dest2[:, 0] - curr_s[:, 0])
            new_v = v2

            curr_s = np.stack((new_sx, new_sy, new_th, new_v), axis=-1)

            trajs2[:, t, 0] = curr_s[:, 0]
            trajs2[:, t, 1] = curr_s[:, 1]
            trajs2[:, t, 2] = curr_s[:, 2]
            trajs2[:, t, 3] = curr_s[:, 3]
            # print(t,curr_s[:, 0], curr_s[:, 1],curr_s[:, 2], curr_s[:, 3])

    return trajs1, trajs2



def pi_2_pi_SDD(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
def calc_nearest_index_SDD(state, cx, cy, cyaw, i):
    dx = [state[0] - icx for icx in cx]
    dy = [state[1] - icy for icy in cy]

    # if i==0:
    #     print("debug current state: %.4f %.4f traj:"%(state[0], state[1]))
    #     for j in range(len(cx)):
    #         print(j,cx[j],cy[j],cyaw[j])

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = np.sqrt(mind)

    dxl = cx[ind] - state[0]
    dyl = cy[ind] - state[1]

    angle = pi_2_pi_SDD(cyaw[ind] - np.arctan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def lqr_1d_controller_SDD(d, v0, v1, dt):
    a_max = 2
    a_min = -2

    # # LQR 1d
    # A = np.array([[1.0, dt], [0, 1.0]])
    # B = np.array([[0.0], [dt]])
    # Q = np.array([[1.0, 0.0], [0.0, 0.1]])
    # R = np.array([0.0001])
    # K = dlqr(A, B, Q, R)

    if dt==0.1:
        K = np.array([[-24.83505377, -11.86672548]])
    else:
        raise NotImplementedError

    # print(K.shape)

    a = K @ np.array([[-d], [v0 - v1]])
    return max(min(a, a_max), a_min)

def dest_controller_SDD(args, s, trajs, dest, t, tag, cached, k_table):  # TODO
    # s    (n_envs, 4) x,y,vx,vy
    # trajs(n_envs, T, 4) x,y,vx,vy in each timestep
    # dest (n_envs, 4) x,y,vx,vy
    # t    current time, also should consider total time left
    # uref (n_envs, 2) ax, ay

    dt = 1.0 / args.fps
    if tag=="1":
        xt=s[:, 0]
        yt=s[:, 1]
        vxt=s[:, 2]
        vyt=s[:, 3]

        xn=dest[:, 0]
        yn=dest[:, 1]

        left_t = (args.env_H - t) * dt

        if args.dest_controller_type=="dest":
            raise NotImplementedError

        elif args.dest_controller_type=="even":
            uref = np.zeros((args.n_envs1, 2))
            for i in range(args.n_envs1):
                s1 = trajs[i, min(t , args.env_H-1), :]
                ctrl_ax = float(lqr_1d_controller_SDD(s1[0] - xt[i], vxt[i], s1[2], dt))
                ctrl_ay = float(lqr_1d_controller_SDD(s1[1] - yt[i], vyt[i], s1[3], dt))
                uref[i, 0] = ctrl_ax
                uref[i, 1] = ctrl_ay
                # if i==0:
                #     print(s1, ctrl_ax, ctrl_ay)
        else:
            raise NotImplementedError
    elif tag=="2":
        xt=s[:,0]
        yt=s[:,1]
        tht=s[:,2]
        vt=s[:,3]

        xn=dest[:,0]
        yn=dest[:,1]
        thn=dest[:,2]

        left_t = (args.env_H - t) * dt

        if args.dest_controller_type=="dest":
            if left_t==0:
                uref = np.zeros((args.n_envs2, 2))
            else:
                l=((xn-xt)**2+(yn-yt)**2)**0.5
                dth=thn-tht
                accel=2*(l-vt*left_t)/left_t/left_t
                omega=dth/left_t
                uref=np.stack((accel, omega), axis=-1)
        elif args.dest_controller_type=="even":
            uref = np.zeros((args.n_envs2, 2))

            for i in range(args.n_envs2):
                ind, e = calc_nearest_index_SDD(s[i, 0:2], trajs[i, :, 0], trajs[i, :, 1], trajs[i, :, 2], i)
                th_e = pi_2_pi_SDD(s[i, 2] - trajs[i, ind, 2])
                v = s[i, 3]

                pe = cached[i]["pe"]
                pth_e= cached[i]["pth_e"]

                x = np.zeros((5, 1))
                x[0, 0] = e
                x[1, 0] = (e - pe) / dt
                x[2, 0] = th_e
                x[3, 0] = (th_e - pth_e) / dt
                x[4, 0] = v - trajs[i, ind, 3]

                # if i==0:
                #     print("x=",x.flatten())

                A = np.zeros((5, 5))
                A[0, 0] = 1.0
                A[0, 1] = dt
                A[1, 2] = v
                A[2, 2] = 1.0
                A[2, 3] = dt
                A[4, 4] = 1.0

                B = np.zeros((5, 2))
                B[3, 1] = 1
                B[4, 0] = dt

                lqr_Q = np.eye(5)
                lqr_R = np.eye(2)

                # print(A, B)
                # K, _, _ = dlqr_steer(A, B, lqr_Q, lqr_R)


                assert args.fps==10
                v_str="%.3f"%(v)
                if v_str in k_table:
                    K = k_table[v_str]
                else:
                    if v<-25:
                        print("v too small: %.4f"%v)
                        K=k_table["-25.000"]
                    else:
                        print("v too big: %.4f" % v)
                        K=k_table["24.999"]

                ustar = -K @ x
                uref[i, 0] = ustar[0, 0]
                uref[i, 1] = ustar[1, 0]

                cached[i]["pe"]=e
                cached[i]["pth_e"]=th_e

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # if tag=="2":
    #     print("ref-ctrl",sum_dt1, sum_dt2)

    return uref

def pi_2_pi_ROUND(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calc_nearest_index_ROUND(state_xy, cxy, cyaw, i):
    d=np.linalg.norm(cxy-state_xy, axis=1)
    ind=np.argmin(d)
    mind=d[ind]
    dxyl=cxy[ind]-state_xy

    angle = pi_2_pi_ROUND(cyaw[ind] - np.arctan2(dxyl[1], dxyl[0]))
    if angle < 0:
        mind *= -1

    return ind, mind

def dest_controller_ROUND(args, s, trajs, dest, t, cached, k_table):
    # s    (n_envs, 9) x,y,w,l,heading,vx,vy,v_lat,v_lon
    # trajs(n_envs, T, 4) x,y,vx,vy in each timestep
    # dest (n_envs, 4) x,y,vx,vy
    # t    current time, also should consider total time left
    # uref (n_envs, 2) accel, omega
    '''
    state.append(feat.x)
    state.append(feat.y)
    state.append(feat.width)
    state.append(feat.length)
    state.append(feat.heading)
    state.append(feat.vx)
    state.append(feat.vy)
    state.append(feat.v_lat)
    state.append(feat.v_lon)
    '''
    dt = 1.0 / args.fps

    # xt=s[:, 0]
    # yt=s[:, 1]
    xyt=s[:, 0:2]
    heading=s[:, 4]
    vt = s[:, 8]

    # TODO trajectory with LQR
    trajs_xy = [traj_item[:, 0:2] for traj_item in trajs]
    trajs_th = [traj_item[:, 2] for traj_item in trajs]  #trajs[:, :, 2]
    trajs_v = [traj_item[:, 3] for traj_item in trajs] #trajs[:, :, 3]


    uref = np.zeros((args.n_envs, 2))

    for i in range(args.n_envs):
        ind, e = calc_nearest_index_ROUND(xyt[i], trajs_xy[i], trajs_th[i], i)
        th_e = pi_2_pi_ROUND(heading[i] - trajs_th[i][ind])

        v = vt[i]

        pe = cached[i]["pe"]
        pth_e = cached[i]["pth_e"]

        x = np.zeros((5, 1))
        x[0, 0] = e
        x[1, 0] = (e - pe) / dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - pth_e) / dt
        x[4, 0] = v - trajs_v[i][ind]

        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt
        A[4, 4] = 1.0

        B = np.zeros((5, 2))
        B[3, 1] = 1
        B[4, 0] = dt

        lqr_Q = np.eye(5)
        lqr_R = np.eye(2)

        # print(A, B)
        # ttt3 = time.time()
        # K, _, _ = dlqr_steer(A, B, lqr_Q, lqr_R)

        # TODO(k table)
        assert args.fps == 25
        v_str = "%.3f" % (v)
        if v_str in k_table:
            K = k_table[v_str]
        else:
            if v < -25:
                print("v too small: %.4f" % v)
                K = k_table["-25.000"]
            else:
                print("v too big: %.4f" % v)
                K = k_table["24.999"]

        ustar = -K @ x
        uref[i, 0] = ustar[0, 0]
        uref[i, 1] = ustar[1, 0]

        # ttt4 = time.time()
        cached[i]["pe"] = e
        cached[i]["pth_e"] = th_e

    return uref


def dest_controller_HIGH(args, s, cached, k_table):
    # uref (n_envs, 2) accel, omega
    '''
    state cbf input: 6*8 + 8
    '''
    split = 6 * 8
    ego_s = s[:, split:]
    dt = 1.0 / 25

    uref = np.zeros((args.n_envs, 2))

    for i in range(args.n_envs):
        lane_width = 3.8963  # TODO prior
        # if before lane changing
        if cached["before_lc"][i, 0] > 0.5:
            e = (ego_s[i, 1] - ego_s[i, 0]) / 2
        elif cached["in_lc"][i, 0] > 0.5:  # TODO ramp should consider more
            e = (ego_s[i, 1] - ego_s[i, 0]) / 2 - cached["direction"][i, 0] * lane_width
        else:  # after lc
            e = (ego_s[i, 1] - ego_s[i, 0]) / 2
        th_e = ego_s[i, 4] % (2*np.pi) - np.pi  # for direction=1 only
        v = ego_s[i, 5]

        # print(i, e, th_e, v)

        pe = cached["pe"][i, 0]
        pth_e = cached["pth_e"][i, 0]

        x = np.zeros((5, 1))
        x[0, 0] = e
        x[1, 0] = (e - pe) / dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - pth_e) / dt
        x[4, 0] = 0.0

        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt
        A[4, 4] = 1.0

        B = np.zeros((5, 2))
        B[3, 1] = 1
        B[4, 0] = dt

        lqr_Q = np.eye(5)
        # lqr_Q = np.diag([0.01, 0.01, 1, 1, 1])
        # lqr_Q = np.eye(5) * 0.1

        lqr_R = np.eye(2)

        # K, _, _ = dlqr_steer(A, B, lqr_Q, lqr_R)

        v_str = "%.3f" % (v)
        if v_str in k_table:
            K = k_table[v_str]
        elif v_str == "-0.000":
            K = k_table["0.000"]
        else:
            if v < -5:
                print("v too small: %.4f" % v)
                K = k_table["-5.000"]
            else:
                print("v too big: %.4f" % v)
                K = k_table["24.999"]

        ustar = -K @ x
        uref[i, 0] = ustar[0, 0]
        uref[i, 1] = ustar[1, 0]

        uref[i, 0] = np.clip(uref[i, 0], -4, 4)
        uref[i, 1] = np.clip(uref[i, 1], -.15, .15)

        cached["pe"][i, 0] = e
        cached["pth_e"][i, 0] = th_e

    return uref