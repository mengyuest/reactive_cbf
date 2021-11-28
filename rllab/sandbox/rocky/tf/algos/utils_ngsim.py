import tensorflow as tf
import numpy as np
import geo_check

def get_safe_mask(args, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
    # get the safe mask from the states
    # input: (batchsize, feat_dim)
    # return: (batchsize, 1)

    if args.new_affordance:
        if args.include_action:
            n_ego_feat = 11
            n_nei_feat = 9
        else:
            n_ego_feat = 9
            n_nei_feat = 7
    elif args.attractive:
        n_ego_feat = 11
        n_nei_feat = 9
    else:
        raise NotImplementedError

    bs = s.shape[0]
    neighbor_s = s[:, n_ego_feat:].reshape((bs, 6, n_nei_feat))
    nei_ind = neighbor_s[:, :, 0]

    outbound_mask = env_out_of_lane.reshape((bs,))

    if args.new_affordance:
        if args.include_action:
            ego_l = s[:, 9]
            ego_w = s[:, 10]
            nei_l = neighbor_s[:, :, 7]
            nei_w = neighbor_s[:, :, 8]
        else:
            ego_l = s[:, 7]
            ego_w = s[:, 8]
            nei_l = neighbor_s[:, :, 5]
            nei_w = neighbor_s[:, :, 6]
    elif args.attractive:
        ego_l = s[:, 9]
        ego_w = s[:, 10]
        nei_l = neighbor_s[:, :, 7]
        nei_w = neighbor_s[:, :, 8]

    ego_r = (((ego_l ** 2 + ego_w ** 2) ** 0.5) / 2)
    nei_r = ((nei_l ** 2 + nei_w ** 2) ** 0.5) / 2
    dx = neighbor_s[:, :, 1]
    dy = neighbor_s[:, :, 2]

    if args.bbx_collision_check:
        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
        poss_collide = (np.sum(nei_collide, axis=1) > 0.5)
        for v_i in range(nei_collide.shape[0]):
            if not poss_collide[v_i]:
                continue
            for nei_i in range(nei_collide.shape[1]):  # neighbors,
                if nei_collide[v_i, nei_i]:  # fine-grained check to flip possible dang->safe
                    if args.new_affordance:
                        phi0 = s[v_i, 5] + np.pi / 2  # because lane as the coordinate directions
                    elif args.attractive:
                        phi0 = np.pi / 2  # because ego veh as coordinates directions

                    hl0 = ego_l[v_i] / 2 + dist_threshold / 2
                    hw0 = ego_w[v_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2

                    x1 = dx[v_i, nei_i]
                    y1 = dy[v_i, nei_i]
                    phi1 = neighbor_s[v_i, nei_i, 3] + np.pi / 2
                    hl1 = nei_l[v_i, nei_i] / 2 + dist_threshold / 2
                    hw1 = nei_w[v_i, nei_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2

                    nei_collide[v_i, nei_i] = geo_check.check_rot_rect_collision(0, 0, phi0, hl0, hw0, x1, y1,
                                                                                 phi1, hl1, hw1)

        coll_mask = (np.sum(nei_collide, axis=1) > 0.5)
    elif args.ellipse_collision_check:
        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
        poss_collide = (np.sum(nei_collide, axis=1) > 0.5)
        for v_i in range(nei_collide.shape[0]):
            if not poss_collide[v_i]:
                continue
            for nei_i in range(nei_collide.shape[1]):  # neighbors,
                if nei_collide[v_i, nei_i]:  # fine-grained check to flip possible dang->safe
                    hl0 = ego_l[v_i] / 2 + dist_threshold / 2
                    hw0 = ego_w[v_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2
                    if args.new_affordance:
                        phi0 = s[v_i, 5] + np.pi / 2  # because lane as the coordinate directions
                    elif args.attractive:
                        phi0 = np.pi / 2  # because ego veh as coordinates directions

                    x1 = dx[v_i, nei_i]
                    y1 = dy[v_i, nei_i]
                    phi1 = neighbor_s[v_i, nei_i, 3] + np.pi / 2
                    hl1 = nei_l[v_i, nei_i] / 2 + dist_threshold / 2
                    hw1 = nei_w[v_i, nei_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2

                    nei_collide[v_i, nei_i] = geo_check.check_rot_rect_collision_by_ellipse(
                        0, 0, phi0, hl0, hw0, x1, y1, phi1, hl1, hw1, args.ellipse_factor)
        coll_mask = (np.sum(nei_collide, axis=1) > 0.5)
    else:
        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
        coll_mask = (np.sum(nei_collide, axis=1) > 0.5)

    dang_mask = np.logical_or(coll_mask, outbound_mask)

    if check_safe:  # no collision for each row, means that ego vehicle is safe
        return np.logical_not(dang_mask)
    else:  # one collision in rows means that ego vehicle is dang
        return dang_mask

def get_safe_mask_agent(args, s, env_out_of_lane, dist_threshold, dist_threshold_side, check_safe):
    # get the safe mask from the states
    # input: (batchsize, feat_dim)
    # return: (batchsize, 6)

    if args.attractive:
        n_ego_feat = 11
        n_nei_feat = 9
    else:
        raise NotImplementedError

    bs = s.shape[0]
    neighbor_s = s[:, n_ego_feat:].reshape((bs, 6, n_nei_feat))
    nei_ind = neighbor_s[:, :, 0]

    outbound_mask = env_out_of_lane.reshape((bs, 1))

    ego_l = s[:, 9]
    ego_w = s[:, 10]
    nei_l = neighbor_s[:, :, 7]
    nei_w = neighbor_s[:, :, 8]

    ego_r = (((ego_l ** 2 + ego_w ** 2) ** 0.5) / 2)
    nei_r = ((nei_l ** 2 + nei_w ** 2) ** 0.5) / 2
    dx = neighbor_s[:, :, 1]
    dy = neighbor_s[:, :, 2]

    if args.bbx_collision_check:
        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
        poss_collide = (np.sum(nei_collide, axis=1) > 0.5)
        for v_i in range(nei_collide.shape[0]):
            if not poss_collide[v_i]:
                continue
            for nei_i in range(nei_collide.shape[1]):  # neighbors,
                if nei_collide[v_i, nei_i]:  # fine-grained check to flip possible dang->safe
                    if args.new_affordance:
                        phi0 = s[v_i, 5] + np.pi / 2  # because lane as the coordinate directions
                    elif args.attractive:
                        phi0 = np.pi / 2  # because ego veh as coordinates directions

                    hl0 = ego_l[v_i] / 2 + dist_threshold / 2
                    hw0 = ego_w[v_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2

                    x1 = dx[v_i, nei_i]
                    y1 = dy[v_i, nei_i]
                    phi1 = neighbor_s[v_i, nei_i, 3] + np.pi / 2
                    hl1 = nei_l[v_i, nei_i] / 2 + dist_threshold / 2
                    hw1 = nei_w[v_i, nei_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2

                    nei_collide[v_i, nei_i] = geo_check.check_rot_rect_collision(0, 0, phi0, hl0, hw0, x1, y1,
                                                                                 phi1, hl1, hw1)

        coll_mask = (nei_collide > 0.5)
    elif args.ellipse_collision_check:
        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
        poss_collide = (np.sum(nei_collide, axis=1) > 0.5)
        for v_i in range(nei_collide.shape[0]):
            if not poss_collide[v_i]:
                continue
            for nei_i in range(nei_collide.shape[1]):  # neighbors,
                if nei_collide[v_i, nei_i]:  # fine-grained check to flip possible dang->safe
                    hl0 = ego_l[v_i] / 2 + dist_threshold / 2
                    hw0 = ego_w[v_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2
                    if args.new_affordance:
                        phi0 = s[v_i, 5] + np.pi / 2  # because lane as the coordinate directions
                    elif args.attractive:
                        phi0 = np.pi / 2  # because ego veh as coordinates directions

                    x1 = dx[v_i, nei_i]
                    y1 = dy[v_i, nei_i]
                    phi1 = neighbor_s[v_i, nei_i, 3] + np.pi / 2
                    hl1 = nei_l[v_i, nei_i] / 2 + dist_threshold / 2
                    hw1 = nei_w[v_i, nei_i] / 2 + dist_threshold_side / 2  # dist_threshold / 2

                    nei_collide[v_i, nei_i] = geo_check.check_rot_rect_collision_by_ellipse(
                        0, 0, phi0, hl0, hw0, x1, y1, phi1, hl1, hw1, args.ellipse_factor)
        coll_mask = (nei_collide > 0.5)
    else:
        nei_dist = (dx ** 2 + dy ** 2) ** 0.5
        nei_thres = ego_r[:, None] + nei_r
        nei_collide = np.logical_and(nei_ind >= 0.5, nei_dist < nei_thres + dist_threshold)
        coll_mask = (nei_collide > 0.5)

    dang_mask = np.logical_or(coll_mask, outbound_mask)

    if check_safe:  # no collision for each row, means that ego vehicle is safe
        return np.logical_not(dang_mask)
    else:  # one collision in rows means that ego vehicle is dang
        return dang_mask



# check feature definition at julia_pkgs/AutoRisk/src/extraction/feature_extractors.jl
# function AutomotiveDrivingModels.pull_features!(
#         ext::AttractiveExtractor,...
def dynamics_attr(args, state, control, primal_control):
    if args.new_affordance:
        return dynamics_new_affordance(args, state, control, primal_control)
    elif args.attractive:
        return dynamics_attractive(args, state, control, primal_control)
    else:
        raise NotImplementedError

def dynamics_attractive(args, state, control, primal_control):
    n_ego_feat = 11
    n_nei_feat = 9

    feat_x_def = tf.constant([0.0, -15.0, 15.0, 0.0, -15.0, 15.0])
    feat_y_def = tf.constant([30.0, 30.0, 30.0, -30.0, -30.0, -30.0])

    next_state = state * 1.0
    # perform precised updates (numerically)
    dT = 0.1
    discrete_num = args.cbf_discrete_num
    dt = dT / discrete_num

    for tt in range(discrete_num):
        # symbol table
        curve = next_state[:, 0]
        lld = next_state[:, 1]
        rld = next_state[:, 2]
        lrd = next_state[:, 3]
        rrd = next_state[:, 4]
        theta = next_state[:, 2+3]
        v = next_state[:, 3+3]
        length = next_state[:, 6+3]
        width = next_state[:, 7+3]
        accel = control[:, 0] + primal_control[:, 0]
        omega = control[:, 1] + primal_control[:, 1]
        if args.debug_accel_only:
            omega = control[:, 1] * 0.0 + primal_control[:, 1]

        # 1-order dynamic
        dx = v * tf.sin(theta) * dt
        ds = v * dt

        # 2-order dynamic
        dv = accel * dt
        dtheta = omega * dt

        # updates
        new_curve = curve
        new_lld = lld - dx
        new_rld = rld + dx
        new_lrd = lrd - dx
        new_rrd = rrd + dx
        new_theta = theta + dtheta
        new_v = v + dv

        new_accel = accel
        new_omega = omega
        new_length = length
        new_width = width

        # neighbors
        nei_feat = tf.reshape(next_state[:, n_ego_feat:], [-1, n_nei_feat])
        nei_ind = nei_feat[:, 0]
        nei_x = nei_feat[:, 1]
        nei_y = nei_feat[:, 2]
        nei_theta=nei_feat[:, 3]
        nei_v=nei_feat[:, 4]

        nei_accel=nei_feat[:, 5]
        nei_omega=nei_feat[:, 6]

        nei_length=nei_feat[:, 7]
        nei_width=nei_feat[:, 8]

        # [1,2,3]->[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
        ds_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(ds, [6]), [6, -1])), [-1])
        dtheta_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(dtheta, [6]), [6, -1])), [-1])
        new_v_cp = tf.reshape(tf.transpose(tf.reshape(tf.tile(new_v, [6]), [6, -1])), [-1])

        # 1-order dynamic
        nei_dx = - nei_v * tf.sin(nei_theta) * dt
        nei_dy = nei_v * tf.cos(nei_theta) * dt

        # 2-order dynamic
        nei_dtheta = nei_omega * dt
        nei_dv = nei_accel * dt

        # updates
        new_nei_x = nei_x + nei_dx - 0
        new_nei_y = nei_y + nei_dy - ds_cp
        # (TODO) consider transformation caused by ego rotation
        new_nei_x, new_nei_y = \
            new_nei_x * tf.cos(dtheta_cp) + new_nei_y * tf.sin(dtheta_cp), \
            - new_nei_x * tf.sin(dtheta_cp) + new_nei_y * tf.cos(dtheta_cp)
        new_nei_theta = nei_theta + nei_dtheta - dtheta_cp
        new_nei_v = nei_v + nei_dv
        new_nei_length = nei_length
        new_nei_width = nei_width

        # TODO clip
        new_nei_x = tf.clip_by_value(new_nei_x, -15.0, 15.0)
        new_nei_y = tf.clip_by_value(new_nei_y, -30.0, 30.0)

        # indicator and bounding values
        # using current calculated value, or the default value, depending on the indicator
        new_nei_x = nei_ind * new_nei_x + (1-nei_ind)*(tf.tile(feat_x_def, [tf.shape(nei_ind)[0]//6])) #TODO
        new_nei_y = nei_ind * new_nei_y + (1-nei_ind)*(tf.tile(feat_y_def, [tf.shape(nei_ind)[0]//6])) #TODO
        new_nei_theta = nei_ind * new_nei_theta
        new_nei_v = nei_ind * new_nei_v + (1-nei_ind) * new_v_cp
        new_nei_accel = nei_accel
        new_nei_omega = nei_omega
        new_nei_length = nei_ind * new_nei_length
        new_nei_width = nei_ind * new_nei_width

        # merge them to a single tensor
        new_ego = tf.stack([new_curve, new_lld, new_rld, new_lrd, new_rrd, new_theta, new_v, new_accel, new_omega, new_length, new_width], axis=-1)
        new_nei = tf.stack(
            [nei_ind, new_nei_x, new_nei_y, new_nei_theta, new_nei_v, new_nei_accel, new_nei_omega, new_nei_length, new_nei_width], axis=-1)

        new_nei = tf.reshape(new_nei, [-1, 6 * n_nei_feat])
        next_state = tf.concat([new_ego, new_nei], axis=-1)

    return next_state

def dynamics_new_affordance(args, state, control, primal_control):
    next_state = state * 1.0
    dt = 0.1 / args.cbf_discrete_num
    for tt in range(args.cbf_discrete_num):
        if args.include_action:
            dsdt = compute_derivative_aw(next_state, control, primal_control)
        else:
            dsdt = compute_derivative(next_state, control, primal_control)
        next_state = next_state + dsdt * dt
    return next_state

def compute_derivative_aw(s, u, u_ref):
    ego_dim = 11
    nei_dim = 9
    num_nei = 6
    curve, lld, rld, lrd, rrd, theta, v, _, _, length, width = tf.split(s, ego_dim, axis=1)
    accel, omega = tf.split(u + u_ref, 2, axis=1)

    nei_s = tf.reshape(s[:, ego_dim:], [tf.shape(s)[0], num_nei, nei_dim])
    nei_ind, nei_x, nei_y, nei_theta, nei_v, nei_accel, nei_omega, nei_length, nei_width = tf.split(nei_s, nei_dim, axis=2)

    # ego features
    d_curve = tf.zeros_like(curve)
    d_lld = - v * tf.sin(theta)
    d_rld = v * tf.sin(theta)
    d_lrd = - v * tf.sin(theta)
    d_rrd = v * tf.sin(theta)
    d_theta = omega
    d_v = accel
    d_accel = tf.zeros_like(accel)
    d_omega = tf.zeros_like(omega)
    d_length = tf.zeros_like(length)
    d_width = tf.zeros_like(width)

    # neighbor features
    d_nei_ind = tf.zeros_like(nei_ind)
    d_nei_x = tf.stack([(v * tf.sin(theta) - nei_v[:, i] * tf.sin(nei_theta[:, i])) * nei_ind[:, i]  for i in range(num_nei)], axis=1)
    d_nei_y = tf.stack([(-v * tf.cos(theta) + nei_v[:, i] * tf.cos(nei_theta[:, i])) * nei_ind[:, i] for i in range(num_nei)], axis=1)
    d_nei_theta = nei_omega * nei_ind
    d_nei_v = nei_accel * nei_ind
    d_nei_accel = tf.zeros_like(nei_accel)
    d_nei_omega = tf.zeros_like(nei_omega)
    d_nei_length = tf.zeros_like(nei_length)
    d_nei_width = tf.zeros_like(nei_width)

    # merge them to a single tensor
    d_ego = tf.stack(
        [d_curve, d_lld, d_rld, d_lrd, d_rrd, d_theta, d_v, d_accel, d_omega, d_length, d_width],
        axis=-1)
    d_nei = tf.stack(
        [d_nei_ind, d_nei_x, d_nei_y, d_nei_theta, d_nei_v, d_nei_accel, d_nei_omega, d_nei_length, d_nei_width], axis=-1)

    d_nei = tf.reshape(d_nei, [-1, num_nei * nei_dim])
    dsdt = tf.concat([d_ego, d_nei], axis=-1)

    return dsdt


def compute_derivative(s, u, u_ref):
    ego_dim = 9
    nei_dim = 7
    num_nei = 6
    curve, lld, rld, lrd, rrd, theta, v, length, width = tf.split(s, ego_dim, axis=1)
    accel, omega = tf.split(u + u_ref, 2, axis=1)

    nei_s = tf.reshape(s[:, ego_dim:], [tf.shape(s)[0], num_nei, nei_dim])
    nei_ind, nei_x, nei_y, nei_theta, nei_v, nei_length, nei_width = tf.split(nei_s, nei_dim, axis=2)

    # ego features
    d_curve = tf.zeros_like(curve)
    d_lld = - v * tf.sin(theta)
    d_rld = v * tf.sin(theta)
    d_lrd = - v * tf.sin(theta)
    d_rrd = v * tf.sin(theta)
    d_theta = omega
    d_v = accel
    d_length = tf.zeros_like(length)
    d_width = tf.zeros_like(width)

    # neighbor features
    d_nei_ind = tf.zeros_like(nei_ind)
    d_nei_x = tf.stack(
        [(v * tf.sin(theta) - nei_v[:, i] * tf.sin(nei_theta[:, i])) * nei_ind[:, i] for i in range(num_nei)], axis=1)
    d_nei_y = tf.stack(
        [(-v * tf.cos(theta) + nei_v[:, i] * tf.cos(nei_theta[:, i])) * nei_ind[:, i] for i in range(num_nei)], axis=1)
    d_nei_theta = tf.zeros_like(nei_theta)
    d_nei_v = tf.zeros_like(nei_v)
    d_nei_length = tf.zeros_like(nei_length)
    d_nei_width = tf.zeros_like(nei_width)

    # merge them to a single tensor
    d_ego = tf.stack([d_curve, d_lld, d_rld, d_lrd, d_rrd, d_theta, d_v, d_length, d_width], axis=-1)
    d_nei = tf.stack([d_nei_ind, d_nei_x, d_nei_y, d_nei_theta, d_nei_v,  d_nei_length, d_nei_width], axis=-1)

    d_nei = tf.reshape(d_nei, [-1, num_nei * nei_dim])
    dsdt = tf.concat([d_ego, d_nei], axis=-1)

    return dsdt