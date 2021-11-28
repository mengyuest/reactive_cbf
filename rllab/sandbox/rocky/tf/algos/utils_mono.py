import tensorflow as tf
import numpy as np

def get_safe_mask_mono(args, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
    # get the safe mask from the states
    # input: (batchsize, feat_dim)
    # return: (batchsize, 1)

    a_max = 4.0

    ego_v = s[:, 0]
    fore_x = s[:, 1]
    fore_v = s[:, 2]
    rear_x = s[:, 3]
    rear_v = s[:, 4]
    if args.simple_safe_checking:
        fore_collide = fore_x - 5 < dist_threshold
        rear_collide = -rear_x - 5 < dist_threshold
    else:
        fore_collide = (fore_x < (ego_v ** 2 - fore_v ** 2) / (2 * a_max) + dist_threshold)
        rear_collide = (-rear_x < (rear_v ** 2 - ego_v ** 2) / (2 * a_max) + dist_threshold)

    dang_mask = np.logical_or(fore_collide >= 0.5, rear_collide > 0.5)

    if check_safe:  # no collision for each row, means that ego vehicle is safe
        return np.logical_not(dang_mask)
    else:  # one collision in rows means that ego vehicle is dang
        return dang_mask

def dynamics_mono(args, state, control, primal_control):
    next_state = state * 1.0
    dT = 0.1
    discrete_num = 1
    dt = dT / discrete_num

    for tt in range(discrete_num):
        # symbol table
        ego_v = next_state[:, 0]
        head_x = next_state[:, 1]
        head_v = next_state[:, 2]
        rear_x = next_state[:, 3]
        rear_v = next_state[:, 4]

        accel = control[:, 0] + primal_control[:, 0]

        # updates
        head_x = head_x - ego_v * dt + head_v * dt
        rear_x = rear_x - ego_v * dt + rear_v * dt
        ego_v = ego_v + accel * dt

        next_state = tf.stack([ego_v, head_x, head_v, rear_x, rear_v], axis=-1)

    return next_state