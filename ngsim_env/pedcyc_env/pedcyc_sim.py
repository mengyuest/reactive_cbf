import os
from os.path import join as ospj
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import rllab
from rllab.envs.base import Env
import ped_trajdata
import scipy

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

class PedCycSim(Env):
    def __init__(self, tmp_params):
        self.feat_dim = tmp_params["num_neighbors"] * 4 + 2
        self.ext_indices = get_ext_indices(tmp_params["ext_intervals"])

        self._observation_space = rllab.spaces.Box(
            low=np.array([-50, ] * len(self.ext_indices)),
            high=np.array([50, ] * len(self.ext_indices)),
        )

        self.epid = 0
        self.t = 0
        self.h = 0

        self.primesteps = tmp_params["primesteps"]
        self.fps = tmp_params["fps"]   # TODO 10 fps
        self.delta_t = 1.0 / self.fps

        self.H = tmp_params["env_H"]

        self.n_veh1 = tmp_params["n_veh1"]
        self.n_veh2 = tmp_params["n_veh2"]
        self.is_render = tmp_params['is_render']

        # TODO control mode
        # ped_only; cyc_only; ped_cyc
        assert tmp_params["control_mode"] in ["ped_only", "cyc_only", "ped_cyc"]
        self.mode = tmp_params["control_mode"]

        if self.mode=="cyc_only":
            self._action_space = rllab.spaces.Box(low=np.array([-4., -.15]), high=np.array([4., .15]))
        else:
            self._action_space = rllab.spaces.Box(low=np.array([-4., -4.]), high=np.array([4., 4.]))

        if self.mode=="ped_only":
            assert self.n_veh1>0
            assert self.n_veh2==0
        elif self.mode=="cyc_only":
            assert self.n_veh1==0
            assert self.n_veh2>0
        else:
            assert self.n_veh1>0
            assert self.n_veh2>0

        cate_splits={
            0: ("deathCircle", "video3"),
            1: ("deathCircle", "video1")
        }
        self.traj_data_dict = {}
        for i in [int(x) for x in tmp_params["traj_idx_list"].split(",")]:
            f_dir = os.path.dirname(os.path.realpath(__file__))
            motion_path = ospj(f_dir, "motion_dict_fps%d.npz" % (self.fps))
            traj_data = np.load(motion_path)["motion_dict"].item()
            self.traj_data_dict[i] = traj_data[cate_splits[i][0]][cate_splits[i][1]]

        self.viz_dir = tmp_params["viz_dir"]
        self.candidates_dict = self.validate_agent_span()
        # print("candidates",self.candidates_dict)

        self.tmp_params = tmp_params
        self.first_traj_idx = None
        self.first_egoids = None
        self.first_ts = None
        self.first_te = None

        self.infos = None

    def validate_agent_span(self):
        candidates_dict = {}
        for key, traj_data in self.traj_data_dict.items():
            candidates_dict[key] = {"Pedestrian":[], "Biker":[]}

            # TODO remove those timestamps because the camera is shaking and coords are wrong
            if key == 0:  # video3
                banned_times = list(range(120 * self.fps - self.H, 180 * self.fps + self.H))
                doable_times = list(range(30 * self.fps - self.H, 70 * self.fps + self.H))
            elif key == 1:  # video1
                banned_times = list(range(0 * self.fps - self.H, 0 * self.fps + self.H))
                doable_times = list(range(335 * self.fps - self.H, 360 * self.fps + self.H)) + \
                               list(range(380 * self.fps - self.H, 410 * self.fps + self.H))
            else:
                raise NotImplementedError


            for label in candidates_dict[key]:
                for pid, (ts, te) in traj_data[label]["ped_span"].items():
                    # not in the banned time spans
                    if ts in banned_times or te in banned_times:
                        continue
                    valid_set=set(range(ts,te+1))
                    # within the doable time span
                    # if valid_set & set(doable_times) == valid_set:
                    if ts in doable_times:
                        if ts + self.H + self.primesteps <= te:
                            candidates_dict[key][label].append(pid)
        return candidates_dict

    def sample_multiple_trajdata(self, max_resamples=100, rseed=None):
        if "fixed_trajectory" in self.tmp_params and self.tmp_params["fixed_trajectory"]:
            if self.first_traj_idx is not None:
                return self.first_traj_idx, self.first_egoids1, self.first_egoids2, self.first_ts, self.first_te

        if rseed is not None:
            np.random.seed(rseed)

        traj_idx = np.random.choice(list(self.candidates_dict.keys()))

        if self.mode == "ped_cyc" or self.mode == "ped_only":
            egoid = np.random.choice(list(self.candidates_dict[traj_idx]["Pedestrian"]))
            traj_data = self.traj_data_dict[traj_idx]["Pedestrian"]
            first_n_veh = self.n_veh1
            if self.mode=="ped_cyc":
                second_n_veh = self.n_veh2
        else:
            egoid = np.random.choice(list(self.candidates_dict[traj_idx]["Biker"]))
            traj_data = self.traj_data_dict[traj_idx]["Biker"]
            first_n_veh = self.n_veh2

        leftmost_ts = traj_data["ped_span"][egoid][0]
        rightmost_te = traj_data["ped_span"][egoid][1]

        offset = self.H + self.primesteps

        ts = np.random.randint(leftmost_ts, rightmost_te - offset + 1)
        te = ts + offset

        egoids = set()

        for pid in traj_data["ped_t_states"]:
            oth_ts = traj_data["ped_span"][pid][0]
            oth_te = traj_data["ped_span"][pid][1]
            if oth_ts <= ts and te <= oth_te:
                egoids.add(pid)

        if len(egoids) < first_n_veh:
            if max_resamples<10:
                print("WARNING: insuffcient sampling ids for first type in sample multiple trajdata," +
                      " resamples remaining: (%d)" % (max_resamples))

            if max_resamples == 0:
                exit("ERROR: reached maximum resamples in sample multiple trajdata")

            return self.sample_multiple_trajdata(max_resamples=max_resamples - 1, rseed=rseed)

        egoids = np.random.choice(list(egoids), size=first_n_veh, replace=False)

        # TODO sample for the second type
        if self.mode == "ped_cyc":
            egoids_alt=set()
            traj_data_alt = self.traj_data_dict[traj_idx]["Biker"]
            for pid in traj_data_alt["ped_t_states"]:
                oth_ts = traj_data_alt["ped_span"][pid][0]
                oth_te = traj_data_alt["ped_span"][pid][1]
                if oth_ts <= ts and te <= oth_te:
                    egoids_alt.add(pid)

            if len(egoids_alt) < second_n_veh:
                if max_resamples < 10:
                    print("WARNING: insuffcient sampling ids for second type in sample multiple trajdata," +
                          " resamples remaining: (%d)" % (max_resamples))

                if max_resamples == 0:
                    exit("ERROR: reached maximum resamples in sample multiple trajdata")

                return self.sample_multiple_trajdata(max_resamples=max_resamples - 1, rseed=rseed)


            egoids_alt = np.random.choice(list(egoids_alt), size=second_n_veh, replace=False)

        else:
            egoids_alt = []

        self.first_traj_idx = traj_idx
        self.first_ts = ts
        self.first_te = te

        if self.mode == "ped_cyc":
            self.first_egoids1 = egoids
            self.first_egoids2 = egoids_alt
            return traj_idx, egoids, egoids_alt, ts, te

        elif self.mode == "ped_only":
            self.first_egoids1 = egoids
            self.first_egoids2 = egoids_alt
            return traj_idx, egoids, egoids_alt, ts, te

        else:
            self.first_egoids1 = egoids_alt
            self.first_egoids2 = egoids
            return traj_idx, egoids_alt, egoids, ts, te

    def get_scene_deepcopy(self, t):
        ped_scene = copy.deepcopy(self.traj_data["Pedestrian"]["snapshots"][t])
        cyc_scene = copy.deepcopy(self.traj_data["Biker"]["snapshots"][t])

        return ped_scene, cyc_scene

    def reset(self, dones=None, rseed=None, **kwargs):
        if dones is not None and dones[0]==False:
            return

        self.epid += 1

        # ttt1=time.time()

        if "video_mode" in self.tmp_params and self.tmp_params["video_mode"]:  # TODO(video)
            traj_idx = self.tmp_params["video_traj_idx"]
            self.egoids1 = [int(xx) for xx in self.tmp_params["video_egoids1"].split(",")]
            self.egoids2 = [int(xx) for xx in self.tmp_params["video_egoids2"].split(",")]
            self.t = self.tmp_params["video_t"]
            self.h = self.tmp_params["video_h"]
        else:
            traj_idx, self.egoids1, self.egoids2, self.t, self.h = \
                self.sample_multiple_trajdata(rseed=rseed)

        # ttt2 = time.time()

        # TODO (for debug reset)
        # traj_idx=0
        # self.egoids1 = [775, 252, 731, 277, 251]
        # self.t=592
        # self.h=643
        #
        # if self.n_veh2!=0:
        #     self.egoids2 =  [269, 759, 284, 242, 260]

        # 1 [ 21  90  39  22 310] [] 4093 4144
        # traj_idx=1
        # self.egoids1=[21, 90, 39, 22, 310]
        # self.egoids2=[]
        # self.t=4093
        # self.h=4144

        # print(traj_idx, self.egoids1, self.egoids2, self.t, self.h)

        # TODO (for debug reset)
        # INIT: traj_idx:1 egoids1:[944  33 968 197  22] egoids2:[1124  243  345  973  759] t:4084
        # 001 00:00:06 out:0.0000(0.0000) col:0.1740(0.1010) 0.70(0.55) e:0.70(0.70)



        # # TODO debug
        # traj_idx=1
        # self.egoids1=[356, 945, 310, 197, 943]
        # self.egoids2=[1151, 1146,  306,   31,   91]
        # self.t= 4028
        # self.h = self.t + self.H + self.primesteps

        print("INIT: traj_idx:%d egoids1:%s egoids2:%s t:%d h:%d" % (traj_idx, self.egoids1, self.egoids2, self.t, self.h))

        # # TODO debug only setup
        # if dones is None:
        #     print("INIT: traj_idx:%d egoids1:%s egoids2:%s t:%d" % (
        #         traj_idx, self.egoids1, self.egoids2, self.t))

        # self.traj_data = copy.deepcopy(self.traj_data_dict[traj_idx])
        self.traj_data = self.traj_data_dict[traj_idx]

        # ttt3 = time.time()

        self.prev_scene1 = None
        self.scene1 = None

        self.prev_scene2 = None
        self.scene2 = None

        for t in range(self.t, self.t + self.primesteps + 1):
            self.prev_scene1 = self.scene1
            self.prev_scene2 = self.scene2
            self.scene1, self.scene2 = self.get_scene_deepcopy(t)

        self.ego_peds1=[None for _ in range(self.n_veh1)]
        self.ego_peds2=[None for _ in range(self.n_veh2)]
        self.backtrace={}

        # ttt4 = time.time()

        for i, ego_id in enumerate(self.egoids1):
            self.ego_peds1[i] = copy.deepcopy(self.scene1[ego_id])
            assert ego_id not in self.backtrace
            self.backtrace[ego_id] = i

        for i, ego_id in enumerate(self.egoids2):
            self.ego_peds2[i] = copy.deepcopy(self.scene2[ego_id])
            assert ego_id not in self.backtrace
            self.backtrace[ego_id] = i + self.n_veh1

        # ttt5 = time.time()

        self.t += self.primesteps

        obs, obs_infos = self.get_observations(
            self.scene1, self.prev_scene1, self.egoids1,
            self.scene2, self.prev_scene2, self.egoids2
        )

        # ttt6 = time.time()

        self.starting_t = self.t
        self.ending_t = self.h

        self.start_state1 = self.get_the_state(self.starting_t, first=True)
        self.end_state1 = self.get_the_state(self.ending_t, first=True)
        self.curr_state1 = self.get_the_state(self.t, first=True)
        self.start_state2 = self.get_the_state(self.starting_t, first=False)
        self.end_state2 = self.get_the_state(self.ending_t, first=False)
        self.curr_state2 = self.get_the_state(self.t, first=False)

        self.t += 1
        self.h = min(self.h, self.t + self.H)
        self.prev_scene1 = self.scene1
        self.prev_scene2 = self.scene2
        self.infos = None

        # obs = np.concatenate((obs1, obs2), axis=0)  # TODO merge two observations together

        # ttt7 = time.time()
        # print("reset %.6f %.6f %.6f %.6f %.6f %.6f" % (ttt2 - ttt1, ttt3 - ttt2, ttt4 - ttt3, ttt5 - ttt4, ttt6 - ttt5, ttt7-ttt6))
        return obs[:, self.ext_indices]

    def get_the_state(self, t, first=True):
        _scene1, _scene2 = self.get_scene_deepcopy(t)
        if first:
            _scene = _scene1
            N = self.n_veh1
            egoids = self.egoids1
        else:
            _scene = _scene2
            N = self.n_veh2
            egoids = self.egoids2
        the_state = np.zeros((N, 4))
        for i, ego_id in enumerate(egoids):
            for feat_i in range(4):
                the_state[i, feat_i] = _scene[ego_id][feat_i]
        return the_state

    def get_current_state(self, first=True):
        if first:
            N = self.n_veh1
            ego_peds = self.ego_peds1
        else:
            N = self.n_veh2
            ego_peds = self.ego_peds2
        curr_state = np.zeros((N, 4))
        for i in range(N):
            for feat_i in range(4):
                curr_state[i, feat_i] = ego_peds[i][feat_i]
        return curr_state

    def step(self, action):
        # tt1=time.time()
        # print(action)

        step_infos = self.step_x(action)

        # tt2=time.time()

        obs, obs_infos = self.get_observations(
            self.scene1, self.prev_scene1, self.egoids1,
            self.scene2, self.prev_scene2, self.egoids2
        )

        # tt3=time.time()

        rewards, rewards_infos = self.get_rewards(obs_infos)

        # tt4=time.time()
        # print("after reward", self.ego_peds2[0])
        infos = {}
        infos.update(step_infos)
        infos.update(obs_infos)
        infos.update(rewards_infos)

        self.infos = infos

        self.curr_state1 = self.get_current_state(first=True)
        self.curr_state2 = self.get_current_state(first=False)


        # tt5=time.time()

        self.t += 1
        terminal = [self.t > self.h for _ in range(self.n_veh1+self.n_veh2)]
        self.reset(dones=terminal)
        self.prev_scene1 = self.scene1
        self.prev_scene2 = self.scene2

        # tt6=time.time()

        # print("%.6f %.6f %.6f %.6f %.6f"%(tt2-tt1, tt3-tt2,tt4-tt3, tt5-tt4, tt6-tt5))

        # obs = np.concatenate((obs1, obs2), axis=0)  # TODO merge two observations together
        # print("after step", self.ego_peds2[0], self.curr_state2[0])
        return obs[:, self.ext_indices], rewards, terminal, infos


    def step_x(self, action):
        assert action.shape[0] == self.n_veh1 + self.n_veh2

        action1 = action[0:self.n_veh1]
        action2 = action[self.n_veh1:self.n_veh1+self.n_veh2]
        if "use_gt_trajs" in self.tmp_params and self.tmp_params["use_gt_trajs"]:
            gt_scene = self.traj_data["Pedestrian"]["snapshots"][self.t]
            for i in range(self.n_veh1):
                self.ego_peds1[i][0] = gt_scene[self.egoids1[i]][0]
                self.ego_peds1[i][1] = gt_scene[self.egoids1[i]][1]
                self.ego_peds1[i][2] = gt_scene[self.egoids1[i]][2]
                self.ego_peds1[i][3] = gt_scene[self.egoids1[i]][3]

            gt_scene = self.traj_data["Biker"]["snapshots"][self.t]
            for i in range(self.n_veh2):
                self.ego_peds2[i][0] = gt_scene[self.egoids2[i]][0]
                self.ego_peds2[i][1] = gt_scene[self.egoids2[i]][1]
                self.ego_peds2[i][2] = gt_scene[self.egoids2[i]][2]
                self.ego_peds2[i][3] = gt_scene[self.egoids2[i]][3]

        else:
            for i in range(self.n_veh1):
                self.ego_peds1[i][0] += self.ego_peds1[i][2] * self.delta_t
                self.ego_peds1[i][1] += self.ego_peds1[i][3] * self.delta_t
                self.ego_peds1[i][2] += action1[i, 0] * self.delta_t
                self.ego_peds1[i][3] += action1[i, 1] * self.delta_t
                if "speed_constraint" in self.tmp_params and self.tmp_params["speed_constraint"] is not None:
                    norm = (self.ego_peds1[i][2]**2+self.ego_peds1[i][3]**2)**0.5
                    ratio=norm/np.clip(norm, 0, self.tmp_params["speed_constraint"])
                    self.ego_peds1[i][2] = self.ego_peds1[i][2] / ratio
                    self.ego_peds1[i][3] = self.ego_peds1[i][3] / ratio

            for i in range(self.n_veh2):
                self.ego_peds2[i][0] += self.ego_peds2[i][3] * np.cos(self.ego_peds2[i][2]) * self.delta_t
                self.ego_peds2[i][1] += self.ego_peds2[i][3] * np.sin(self.ego_peds2[i][2]) * self.delta_t
                self.ego_peds2[i][2] += action2[i, 1] * self.delta_t
                self.ego_peds2[i][3] += action2[i, 0] * self.delta_t

                if "speed_constraint" in self.tmp_params and self.tmp_params["speed_constraint"] is not None:
                    self.ego_peds2[i][3]=np.clip(self.ego_peds2[i][3], 0, self.tmp_params["speed_constraint"])

        # print("after x_step", self.ego_peds2[0])

        self.scene1, self.scene2 = self.get_scene_deepcopy(self.t)
        backup_states = [None for _ in range(self.n_veh1+self.n_veh2)]

        for i, vid in enumerate(self.egoids1):
            state = self.scene1[vid]
            backup_states[i] = copy.deepcopy(state)
            state[0] = self.ego_peds1[i][0]
            state[1] = self.ego_peds1[i][1]
            state[2] = self.ego_peds1[i][2]
            state[3] = self.ego_peds1[i][3]

        for i, vid in enumerate(self.egoids2):
            state = self.scene2[vid]
            backup_states[i+self.n_veh1] = copy.deepcopy(state)
            state[0] = self.ego_peds2[i][0]
            state[1] = self.ego_peds2[i][1]
            state[2] = self.ego_peds2[i][2]
            state[3] = self.ego_peds2[i][3]

        step_infos = {"rmse_pos":[], "rmse_vel":[], "rmse_t":[], "x":[], "y":[], "s":[], "phi":[]}

        def _sqrt(x,y):
            return (x**2+y**2)**0.5

        for i,vid in enumerate(self.egoids1):
            v0 = backup_states[i]
            v1 = self.ego_peds1[i]
            step_infos["rmse_pos"].append(_sqrt(v1[0] - v0[0], v1[1] - v0[1]))
            step_infos["rmse_vel"].append(_sqrt(v1[2] - v0[2], v1[3] - v0[3]))
            step_infos["rmse_t"].append(0.0)
            step_infos["x"].append(0.0)
            step_infos["y"].append(0.0)
            step_infos["s"].append(0.0)
            step_infos["phi"].append(0.0)

        for i,vid in enumerate(self.egoids2):
            v0 = backup_states[i+self.n_veh1]
            v1 = self.ego_peds2[i]
            step_infos["rmse_pos"].append(_sqrt(v1[0] - v0[0], v1[1] - v0[1]))
            step_infos["rmse_vel"].append(abs(v1[3] - v0[3]))
            step_infos["rmse_t"].append(0.0)
            step_infos["x"].append(0.0)
            step_infos["y"].append(0.0)
            step_infos["s"].append(0.0)
            step_infos["phi"].append(0.0)

        # print("after x_step return", self.ego_peds2[0])
        return step_infos

    def get_observations(self, scene1, prev_scene1, egoids1, scene2, prev_scene2, egoids2):
        feature_list = []
        addtional_info = {}
        for i, ego_id in enumerate(egoids1):
            feature_vec, feature_info = PedCycSim.get_pedcyc_observation(
                self.traj_data, self.t, self.delta_t, scene1, scene2, scene1[ego_id], ego_id, self.feat_dim, self.tmp_params,
            first=True)
            feature_list.append(feature_vec)
            for key in feature_info:
                if key not in addtional_info:
                    addtional_info[key] = []
                addtional_info[key].append(feature_info[key])

        for i, ego_id in enumerate(egoids2):
            feature_vec, feature_info = PedCycSim.get_pedcyc_observation(
                self.traj_data, self.t, self.delta_t, scene1, scene2, scene2[ego_id], ego_id, self.feat_dim, self.tmp_params,
            first=False)
            feature_list.append(feature_vec)
            for key in feature_info:
                if key not in addtional_info:
                    addtional_info[key] = []
                addtional_info[key].append(feature_info[key])

        feature_array = np.stack(feature_list)
        for key in addtional_info:
            addtional_info[key] = np.array(addtional_info[key])
        return feature_array, addtional_info

    def in_region(self, x, y, x_min, x_max, y_min, y_max):
        return x_min<x<x_max and y_min <y<y_max

    def render(self):

        plt.figure(figsize=(8, 8))
        plt.axis('off')
        ax = plt.gca()
        ax.cla()
        c_x_min = 0
        c_x_max = 60
        c_y_min = 0
        c_y_max = 60

        plt.fill([c_x_min, c_x_max, c_x_max, c_x_min],
                 [c_y_min, c_y_min, c_y_max, c_y_max],
                 color="black", linewidth=0.0)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        for id, ped in self.scene1.items():
            if id in self.egoids1 and "print_gt" in self.tmp_params and self.tmp_params['print_gt']:
                id, gt_x, gt_y = id, self.traj_data["Pedestrian"]["snapshots"][self.t-1][id][0], \
                                     self.traj_data["Pedestrian"]["snapshots"][self.t-1][id][1]
                gt_circ = plt.Circle((gt_x, gt_y), self.tmp_params["ped_radius"], color="cyan")

                if self.in_region(gt_x, gt_y, c_x_min+3, c_x_max-3, c_y_min+3, c_y_max-3):
                    ax.add_artist(gt_circ)
                    plt.text(gt_x+0.1, gt_y+0.2, "%d"%id, color="white", fontsize=8)
            id, x, y = id, ped[0], ped[1]

            if id in self.egoids1 and self.infos is not None and self.infos["is_colliding"][self.backtrace[id]]==1.0:
                ego_color = "pink"
            else:
                ego_color = "red" if id in self.egoids1 else "green"
            if self.in_region(x, y, c_x_min+3, c_x_max-3, c_y_min+3, c_y_max-3):
                circle1 = plt.Circle((x, y), self.tmp_params["ped_radius"], color=ego_color)
                ax.add_artist(circle1)
                plt.text(x+0.1, y+0.2, "%d" % id, color="white", fontsize=8)


        cyc_ratio=1.0
        # TODO for cyclist we might want to visualize in another way
        for id, cyc in self.scene2.items():
            if id in self.egoids2 and "print_gt" in self.tmp_params and self.tmp_params['print_gt']:
                id, gt_x, gt_y, gt_th = id, self.traj_data["Biker"]["snapshots"][self.t-1][id][0], \
                                 self.traj_data["Biker"]["snapshots"][self.t-1][id][1], \
                                 self.traj_data["Biker"]["snapshots"][self.t - 1][id][2]
                if self.in_region(gt_x, gt_y, c_x_min+3, c_x_max-3, c_y_min+3, c_y_max-3):
                    # gt_circ = plt.Circle((gt_x, gt_y), self.tmp_params["ped_radius"], color="gray")
                    # ax.add_artist(gt_circ)
                    gt_x0 = gt_x - self.tmp_params["ped_radius"] * np.cos(gt_th) * cyc_ratio
                    gt_y0 = gt_y - self.tmp_params["ped_radius"] * np.sin(gt_th) * cyc_ratio
                    gt_xd = 2 * self.tmp_params["ped_radius"] * np.cos(gt_th) * cyc_ratio
                    gt_yd = 2 * self.tmp_params["ped_radius"] * np.sin(gt_th) * cyc_ratio
                    plt.arrow(gt_x0, gt_y0, gt_xd, gt_yd, color="cyan", head_width=0.5, width=0.05)
                    plt.text(gt_x+0.1, gt_y+0.2, "%d"%id, color="white", fontsize=8)

                # # TODO (debug for ref controller)
                # ii = self.backtrace[id] - self.n_veh1
                # ref_s = self.ref_traj[ii, self.t-self.starting_t-1]
                # if self.in_region(ref_s[0], ref_s[1], c_x_min + 3, c_x_max - 3, c_y_min + 3, c_y_max - 3):
                #     ref_x0 = ref_s[0] - self.tmp_params["ped_radius"] * np.cos(ref_s[2]) * cyc_ratio
                #     ref_y0 = ref_s[1] - self.tmp_params["ped_radius"] * np.sin(ref_s[2]) * cyc_ratio
                #     ref_xd = 2 * self.tmp_params["ped_radius"] * np.cos(ref_s[2]) * cyc_ratio
                #     ref_yd = 2 * self.tmp_params["ped_radius"] * np.sin(ref_s[2]) * cyc_ratio
                #     plt.arrow(ref_x0, ref_y0, ref_xd, ref_yd, color="green", head_width=0.5, width=0.05)
                #     plt.text(ref_x0 + 0.1, ref_y0 + 0.2, "%d" % id, color="white", fontsize=8)



            id, x, y, th = id, cyc[0], cyc[1], cyc[2]

            if id in self.egoids2 and self.infos is not None and self.infos["is_colliding"][self.backtrace[id]]==1.0:
                ego_color = "pink"
                # print("print")
            else:
                # ego_color = "orange" if id in self.egoids2 else "brown"
                ego_color = "red" if id in self.egoids2 else "green"
            if self.in_region(x, y, c_x_min+3, c_x_max-3, c_y_min+3, c_y_max-3):
                # circle1 = plt.Circle((x, y), self.tmp_params["ped_radius"], color=ego_color)
                # ax.add_artist(circle1)

                x0 = x - self.tmp_params["ped_radius"] * np.cos(th) * cyc_ratio
                y0 = y - self.tmp_params["ped_radius"] * np.sin(th) * cyc_ratio
                xd = 2 * self.tmp_params["ped_radius"] * np.cos(th) * cyc_ratio
                yd = 2 * self.tmp_params["ped_radius"] * np.sin(th) * cyc_ratio

                plt.arrow(x0, y0, xd, yd, color=ego_color, head_width=0.5, width=0.05)

                plt.text(x+0.1, y+0.2, "%d" % id, color="white", fontsize=8)

        plt.tight_layout()
        plt.xlim(c_x_min, c_x_max)
        plt.ylim(c_y_min, c_y_max)
        ax.set_xlim((c_x_min, c_x_max))
        ax.set_ylim((c_y_min, c_y_max))
        os.makedirs(ospj(self.viz_dir, "%s"%self.epid), exist_ok=True)
        # print("save to %s/%s/%05d.png" % (self.viz_dir, self.epid, self.t-1))
        plt.savefig("%s/%s/%05d.png" % (self.viz_dir, self.epid, self.t-1), bbox_inches='tight', pad_inches=0)
        plt.close()


    @staticmethod
    def get_pedcyc_observation(traj_data, t, dt, scene1, scene2, ego_veh, ego_id, feat_dim, tmp_params, first=True):
    # def get_ped_observation(traj_data, t, dt, scene, veh_scene, ego_veh, feat_dim, tmp_params):
        feat_infos={}
        # nearest N neighbors (x, y, vx, vy)
        dists=[]
        for nei_id, nei in scene1.items():
            if nei_id != ego_id:
                d = ((nei[0] - ego_veh[0])**2+(nei[1]-ego_veh[1])**2)**0.5
                dists.append((nei_id, "ped", nei[0], nei[1], d))

        for nei_id, nei in scene2.items():
            if nei_id != ego_id:
                d = ((nei[0] - ego_veh[0]) ** 2 + (nei[1] - ego_veh[1]) ** 2) ** 0.5
                dists.append((nei_id, "cyc", nei[0], nei[1], d))

        topk = tmp_params["num_neighbors"]
        indices_modes_xys = [(z[0], z[1], z[2], z[3]) for z in sorted(dists, key=lambda x: x[4])[:topk]]
        indices = [x[0] for x in indices_modes_xys]
        ind_modes = [x[1] for x in indices_modes_xys]
        ind_xys = [x[2:4] for x in indices_modes_xys]
        feat_infos["neighbors"]=indices
        feat_infos["neighbors_modes"] = ind_modes
        feat_infos["neighbors_xys"] = ind_xys

        feature_vec = np.zeros(feat_dim,)

        if first:  # TODO * -> ped
            for i, idx in enumerate(indices):
                if ind_modes[i] == "ped":  # TODO ped -> ped
                    feature_vec[i*4+0] = scene1[idx][0] - ego_veh[0]
                    feature_vec[i*4+1] = scene1[idx][1] - ego_veh[1]
                    feature_vec[i*4+2] = scene1[idx][2] - ego_veh[2]
                    feature_vec[i*4+3] = scene1[idx][3] - ego_veh[3]
                else:  # TODO cyc -> ped
                    feature_vec[i * 4 + 0] = scene2[idx][0] - ego_veh[0]
                    feature_vec[i * 4 + 1] = scene2[idx][1] - ego_veh[1]
                    feature_vec[i * 4 + 2] = scene2[idx][3] * np.cos(scene2[idx][2]) - ego_veh[2]
                    feature_vec[i * 4 + 3] = scene2[idx][3] * np.sin(scene2[idx][2]) - ego_veh[3]

        else:  # TODO * -> cyc
            for i, idx in enumerate(indices):
                if ind_modes[i] == "ped":  # TODO ped -> cyc
                    feature_vec[i * 4 + 0] = scene1[idx][0] - ego_veh[0]
                    feature_vec[i * 4 + 1] = scene1[idx][1] - ego_veh[1]
                    feature_vec[i * 4 + 2] = scene1[idx][2] - ego_veh[3] * np.cos(ego_veh[2])
                    feature_vec[i * 4 + 3] = scene1[idx][3] - ego_veh[3] * np.sin(ego_veh[2])
                else:  # TODO cyc -> cyc
                    feature_vec[i * 4 + 0] = scene2[idx][0] - ego_veh[0]
                    feature_vec[i * 4 + 1] = scene2[idx][1] - ego_veh[1]
                    feature_vec[i * 4 + 2] = scene2[idx][3] * np.cos(scene2[idx][2]) - ego_veh[3] * np.cos(ego_veh[2])
                    feature_vec[i * 4 + 3] = scene2[idx][3] * np.sin(scene2[idx][2]) - ego_veh[3] * np.sin(ego_veh[2])


        # clipping
        x_max, x_min = 30, -30
        y_max, y_min = 30, -30
        for i, idx in enumerate(indices):
            feature_vec[i * 4 + 0] = max(min(feature_vec[i * 4 + 0], x_max), x_min)
            feature_vec[i * 4 + 1] = max(min(feature_vec[i * 4 + 1], y_max), y_min)

        if "record_vxvy" in tmp_params and tmp_params["record_vxvy"]:
            feature_vec[feat_dim - 2] = ego_veh[2]  # for first, this is vx,vy; for not first, this is th,v
            feature_vec[feat_dim - 1] = ego_veh[3]
            ctrl_ax = 0.0
            ctrl_ay = 0.0
        else:
            raise NotImplementedError

        feat_infos["controls"] = np.array([ctrl_ax, ctrl_ay])
        if first:  # TODO ped - ax,ay
            feat_infos["gt_accels"] = np.array([
                traj_data["Pedestrian"]["snapshots"][t][ego_id][4],
                traj_data["Pedestrian"]["snapshots"][t][ego_id][5],
            ])
        else:   # TODO cyc - a,w
            feat_infos["gt_accels"] = np.array([
                traj_data["Biker"]["snapshots"][t][ego_id][4],
                traj_data["Biker"]["snapshots"][t][ego_id][5],
            ])

        return feature_vec, feat_infos

    def get_rewards(self, obs_infos):
        rewards = np.array([0 for _ in range(self.n_veh1 + self.n_veh2)])
        infos = {"collision": [], "is_colliding": [], "out_of_lane": [0.0]* (self.n_veh1+self.n_veh2),}
        for i, ego_veh1 in enumerate(self.ego_peds1):
            is_collide = 0.0
            for j, other_id in enumerate(obs_infos["neighbors"][i]):
                if other_id == -1:
                    continue
                if other_id in self.scene1:
                    other = self.scene1[other_id]
                elif other_id in self.scene2:
                    other = self.scene2[other_id]
                else:
                    raise NotImplementedError

                if ((ego_veh1[0] - other[0])**2+(ego_veh1[1] - other[1])**2)**0.5 < self.tmp_params["ped_radius"] * 2:
                    print("t:%d,id:%d,id:%d"%(self.t, self.egoids1[i], other_id))
                    is_collide=1.0
                    break

            infos["collision"].append(is_collide)
            infos["is_colliding"].append(is_collide)

        for i, ego_veh2 in enumerate(self.ego_peds2):
            is_collide = 0.0
            for j, other_id in enumerate(obs_infos["neighbors"][i+self.n_veh1]):
                if other_id == -1:
                    continue
                if other_id in self.scene1:
                    other = self.scene1[other_id]
                elif other_id in self.scene2:
                    other = self.scene2[other_id]
                else:
                    raise NotImplementedError
                if ((ego_veh2[0] - other[0]) ** 2 + (ego_veh2[1] - other[1]) ** 2) ** 0.5 < self.tmp_params["ped_radius"] * 2:
                    print("t:%d,id:%d,id:%d" % (self.t, self.egoids2[i], other_id))
                    is_collide = 1.0
                    break

            infos["collision"].append(is_collide)
            infos["is_colliding"].append(is_collide)

        return rewards, infos

    def obs_names(self):
        return [str(i) for i in range(self.feat_dim)]

    def vec_env_executor(self, *args, **kwargs):
        return self

    @property
    def num_envs(self):
        return self.n_veh1 + self.n_veh2

    @property
    def vectorized(self):
        return True

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


if __name__=="__main__":
    tmp_params = {
        "rseed": 1007,

        "traj_idx_list": "0,1",
        "ext_intervals": "1,34",
        "env_H": 100,
        "primesteps": 1,
        "control_mode": "ped_cyc",
        "n_veh1": 5,
        "n_veh2": 5,
        "num_neighbors": 8,
        "ped_radius": 0.15,

        "is_render": True,
        "viz_dir": "./vis_veh/",   # "./"
        "print_gt": True,


        "record_vxvy": True,

        "fps": 10,
    }

    np.random.seed(tmp_params['rseed'])
    env = PedCycSim(tmp_params=tmp_params)
    obs = env.reset()
    if env.is_render:
        env.render()

    num_reset = 3
    reset_cnt = 0
    action = np.zeros((env.n_veh1+env.n_veh2, 2))
    t=0
    lag=10
    # lag=0
    while reset_cnt < num_reset:
        obs, _, done, infos = env.step(action)
        # if t<lag:
        #     action = action * 0.0
        # else:
        #     action = infos["controls"]  #obs[:, -2:]
        #     # action = infos["gt_accels"]
        if done[0] == True:
            reset_cnt += 1
            t=0
        if env.is_render and (not (done[0] and reset_cnt==num_reset)):
            env.render()
        t+=1