import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh


def gen_bbox(c, rot, hl, hw):
    box = np.array([
        [hl, -hl, -hl, hl],
        [hw, hw, -hw, -hw]
    ])

    return rot @ box + c


def check_rot_rect_collision(x0, y0, phi0, hl0, hw0, x1, y1, phi1, hl1, hw1):
    # c0, c1 2*1 centers of matrices
    # r0,r1 2*4 matrices
    # phi0, phi1 angles of x-axis to global coords

    c0 = np.array([[x0], [y0]])
    c1 = np.array([[x1], [y1]])

    rot0= np.array([[np.cos(phi0), -np.sin(phi0)],
                    [np.sin(phi0), np.cos(phi0)]])

    rot1 = np.array([[np.cos(phi1), -np.sin(phi1)],
                     [np.sin(phi1), np.cos(phi1)]])

    r0 = gen_bbox(c0, rot0, hl0, hw0)
    r1 = gen_bbox(c1, rot1, hl1, hw1)

    # print(r0, r1)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 8))
    # plt.axis('equal')
    #
    # x1 = r0[0]
    # y1 = r0[1]
    # x2 = r1[0]
    # y2 = r1[1]
    # plt.fill(x1, y1)
    # plt.fill(x2, y2)
    # plt.show()

    return not proj_to_exists_split_axis(c0, rot0, hl0, hw0, r1) and not proj_to_exists_split_axis(c1, rot1, hl1, hw1, r0)

def proj_to_exists_split_axis(c0, rot0, hl0, hw0, r1):
    A = rot0.T @ (r1-c0)
    Amin, Amax = np.min(A, axis=1), np.max(A, axis=1)
    return Amin[0] > hl0 or Amax[0] < -hl0 or Amin[1] > hw0 or Amax[1] < -hw0



def K(x, dd, v):
    return 1. - np.sum(v * ((dd * x * (1. - x)) / (x + dd * (1. - x))) * v)

def f(x, A, B, a, b):
    return 1-(b-a).T @ np.linalg.inv(1/(1-x) * np.linalg.inv(A) + 1/x * np.linalg.inv(B)) @ (b-a)

def ellipsoids_intersect(A, B, a, b):
    dd, Phi = eigh(A, B, eigvals_only=False)
    v = np.dot(Phi.T, a - b)
    # res = minimize_scalar(K, bounds=(0., 1.), args=(dd, v), method='bounded')
    res = minimize_scalar(f, bounds=(0., 1.), args=(A,B,a,b), method='bounded')
    # print("optim fun=",res.fun)

    return (res.fun >= 0)

# return A, x0
# (x-x0)^T A (x-x0) <= 1
def bbox2ellipse(x, y, h, w, theta, factor=1.0):

    # factor=1 ~ assuming minimum area
    assert (factor**2)>0.5
    a = np.sqrt(1.0 / 2.0) * h * factor
    # b = 1 / (np.sqrt(4 - 2/(factor*factor))) * 2 * w
    b = factor / np.sqrt(2 * factor * factor -1) * np.sqrt(1.0 / 2.0) * w

    Q = np.array([[1/a/a, 0.0],[0.0, 1/b/b]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    A = R @ Q @ R.T

    return A, np.array([[x],[y]])

def get_bbox(h, w, theta, x, y):
    bbox = np.array(
        [[h / 2, h / 2, -h / 2, -h / 2],
         [w / 2, -w / 2, -w / 2, w / 2]]
    )

    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )

    trans_mat = np.array([[x], [y]])

    rot_bbox = rot_mat @ bbox
    trans_bbox = rot_bbox + trans_mat

    return trans_bbox

def get_ellipse_value(xx, yy, A, dx):
    return (xx - dx[0]) ** 2 * A[0, 0] + (yy - dx[1]) ** 2 * A[1, 1] + 2 * (xx - dx[0]) * (yy - dx[1]) * A[0, 1]

def check_rot_rect_collision_by_ellipse(x0, y0, phi0, h0, w0, x1, y1, phi1, h1, w1, factor):
    A0, dx0 = bbox2ellipse(x0, y0, h0, w0, phi0, factor)
    A1, dx1 = bbox2ellipse(x1, y1, h1, w1, phi1, factor)
    return ellipsoids_intersect(A0, A1, dx0, dx1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(10071)

    factor = 1.2

    x0 = float(np.random.uniform() * 5)
    y0 = float(np.random.uniform() * 5)
    h0 = 5.0
    w0 = 2.0
    theta0 = float(np.random.uniform() * 1.57)
    print(x0, y0, h0, w0, theta0)
    A0, dx0 = bbox2ellipse(x0, y0, h0, w0, theta0, factor)


    x1 = float(np.random.uniform() * 5)
    y1 = float(np.random.uniform() * 5) - 1.2
    h1 = 6.0
    w1 = 1.6
    theta1 = float(np.random.uniform() * 1.57)
    print(x1, y1, h1, w1, theta1)
    A1, dx1 = bbox2ellipse(x1, y1, h1, w1, theta1, factor)

    # plot box
    bbox0 = get_bbox(h0, w0, theta0, x0, y0)
    plt.plot(bbox0[0, [0, 1, 2, 3, 0]], bbox0[1, [0, 1, 2, 3, 0]])

    bbox1 = get_bbox(h1, w1, theta1, x1, y1)
    plt.plot(bbox1[0, [0, 1, 2, 3, 0]], bbox1[1, [0, 1, 2, 3, 0]])

    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10

    # plot ellipse
    X = np.arange(xmin, xmax, 0.1)
    Y = np.arange(ymin, ymax, 0.1)
    xx, yy = np.meshgrid(X, Y, sparse=True)

    Z0 = get_ellipse_value(xx, yy, A0, dx0)
    Z1 = get_ellipse_value(xx, yy, A1, dx1)

    check = ellipsoids_intersect(A0, A1, dx0, dx1)
    print(check)
    print(check_rot_rect_collision_by_ellipse(x0, y0, theta0, h0, w0, x1, y1, theta1, h1, w1, factor))

    plt.contour(X, Y, Z0, levels=[1.0])
    plt.contour(X, Y, Z1, levels=[1.0])
    plt.axis('scaled')
    plt.show()