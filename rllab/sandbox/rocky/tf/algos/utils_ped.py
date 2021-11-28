import tensorflow as tf
import numpy as np

def get_safe_mask_ped(args, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
    # get the safe mask from the states
    # input: (batchsize, feat_dim)
    # return: (batchsize, 1)
    n_nei_feat = 4

    nei_s = s.reshape((s.shape[0], args.num_neighbors, n_nei_feat))
    nei_d = np.linalg.norm(nei_s[:, :, :2], axis=-1)
    collide = (nei_d < dist_threshold + 2 * args.ped_radius)

    collide = np.sum(collide, axis=1)

    dang_mask = collide >= 0.5

    if check_safe:  # no collision for each row, means that ego vehicle is safe
        return np.logical_not(dang_mask)
    else:  # one collision in rows means that ego vehicle is dang
        return dang_mask

def dynamics_ped(args, state, control, primal_control):
    n_nei_feat = 4

    next_state = state * 1.0
    dT = 1.0 / 24

    N = args.num_neighbors

    ax = control[:, 0] + primal_control[:, 0]
    ay = control[:, 1] + primal_control[:, 1]

    # return tf.stack([ax, ax, ax, ax, ax, ax, ax, ax, ax, ax, ax, ax, ax, ax, ax, ax,
    #                  ay, ay, ay, ay,ay, ay, ay, ay,ay, ay, ay, ay,ay, ay, ay, ay], axis=-1)


    ax_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ax, [N]), [N, -1])), [-1])
    ay_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ay, [N]), [N, -1])), [-1])

    nei_feat = tf.reshape(next_state, [-1, n_nei_feat])
    # [1,2,3]->[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]

    # x = x + vx * dt
    # y = y + vy * dt
    # vx = vx + ax * dt
    # vy = vy + ay * dt
    new_x = nei_feat[:, 0] + nei_feat[:, 2] * dT
    new_y = nei_feat[:, 1] + nei_feat[:, 3] * dT
    new_vx = nei_feat[:, 2] - ax_cp * dT
    new_vy = nei_feat[:, 3] - ay_cp * dT

    # clipping
    x_max, x_min = 30.0, -30.0
    y_max, y_min = 30.0, -30.0

    new_x = tf.clip_by_value(new_x, x_min, x_max)
    new_y = tf.clip_by_value(new_y, y_min, y_max)

    new_nei = tf.stack([new_x, new_y, new_vx, new_vy], axis=-1)

    return tf.reshape(new_nei, [-1, N * n_nei_feat])