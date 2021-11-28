import tensorflow as tf
import numpy as np

def get_safe_mask_easy(args, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
    # get the safe mask from the states
    # input: (batchsize, feat_dim)
    # return: (batchsize, 1)
    nei_d = np.linalg.norm(s[:, :2], axis=-1)
    outbound_mask = env_out_of_lane.reshape((nei_d.shape[0],))
    collide = (nei_d < dist_threshold + args.radius + args.obs_radius)
    dang_mask = np.logical_or(collide >= 0.5, outbound_mask > 0.5)

    if check_safe:  # no collision for each row, means that ego vehicle is safe
        return np.logical_not(dang_mask)
    else:  # one collision in rows means that ego vehicle is dang
        return dang_mask

def dynamics_easy(args, state, control, primal_control):
    dT = 0.1

    ax = control[:, 0] + primal_control[:, 0]
    ay = control[:, 1] + primal_control[:, 1]

    new_x = state[:, 0] + state[:, 2] * dT
    new_y = state[:, 1] + state[:, 3] * dT
    new_vx = state[:, 2] + ax * dT
    new_vy = state[:, 3] + ay * dT

    return tf.stack([new_x, new_y, new_vx, new_vy], axis=-1)