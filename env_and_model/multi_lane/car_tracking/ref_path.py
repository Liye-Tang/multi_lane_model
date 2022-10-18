import bezier
import numpy as np
from math import pi, cos, sin, tan
import matplotlib.pyplot as plt

from env_and_model.multi_lane.car_tracking.utils import *


class ReferencePath(object):
    def __init__(self, goal_point, start_point=(0, 0, 90)):
        self.start_point = start_point
        self.goal_point = goal_point
        self.exp_v = Para.EXPECTED_V
        self.control_points = self.get_bezier_control_points()
        self.ref_path = self.construct_ref_path()
        self.whole_path = self.construct_whole_path()
        self.random_n_points = None

    def get_bezier_control_points(self):
        x1, y1, phi1 = self.start_point[0], self.start_point[1], self.start_point[2] * pi / 180
        x4, y4, phi4 = self.goal_point[0], self.goal_point[1], self.goal_point[2] * pi / 180
        weight = 7/10
        x2 = x1 * ((cos(phi1) ** 2) * (1-weight) + sin(phi1) ** 2) + y1 * (-sin(phi1) * cos(phi1) * weight) + x4 * ((cos(phi1) ** 2) * weight) + y4 * (sin(phi1) * cos(phi1) * weight)
        y2 = x1 * (-sin(phi1) * cos(phi1) * weight) + y1 * (cos(phi1) ** 2 + (sin(phi1) ** 2) * (1-weight)) + x4 * (sin(phi1) * cos(phi1) * weight) + y4 * ((sin(phi1) ** 2) * weight)
        x3 = x1 * (cos(phi4) ** 2) * weight + y1 * (sin(phi4) * cos(phi4) * weight) + x4 * ((cos(phi4) ** 2) * (1-weight) + sin(phi4) ** 2) + y4 * (-sin(phi4) * cos(phi4) * weight)
        y3 = x1 * (sin(phi4) * cos(phi4) * weight) + y1 * ((sin(phi4) ** 2) * weight) + x4 * (-sin(phi4) * cos(phi4) * weight) + y4 * (cos(phi4) ** 2 + (sin(phi4) ** 2) * (1-weight))
        control_point1 = x1, y1
        control_point2 = x2, y2
        control_point3 = x3, y3
        control_point4 = x4, y4
        # print([control_point1, control_point2, control_point3, control_point4])

        return [control_point1, control_point2, control_point3, control_point4]

    def construct_ref_path(self):
        node = np.asfortranarray([[i[0] for i in self.control_points],
                                  [i[1] for i in self.control_points]], dtype=np.float32)
        curve = bezier.Curve(node, degree=3)
        # print(curve.length)
        s_vals = np.linspace(0, 1.0, int(curve.length) * Para.METER_POINT_NUM)
        trj_data = curve.evaluate_multi(s_vals)
        trj_data = trj_data.astype(np.float32)
        x, y = trj_data[0], trj_data[1]
        xs_1, ys_1 = trj_data[0][:-1], trj_data[1][:-1]
        xs_2, ys_2 = trj_data[0][1:], trj_data[1][1:]
        phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
        phi = np.append(90, phis_1)
        v = np.ones_like(phi) * self.exp_v
        return x, y, phi, v

    def construct_whole_path(self):
        # start ref path
        start_point_num = int(Para.START_LENGTH * Para.METER_POINT_NUM)
        start_xs = np.zeros(start_point_num)
        start_ys = np.linspace(
            -Para.START_LENGTH,
            0,
            start_point_num
        )
        start_phis = np.zeros(start_point_num)
        start_vs = np.ones(start_point_num) * self.exp_v

        # medium ref path
        medium_xs = self.ref_path[0]
        medium_ys = self.ref_path[1]
        medium_phis = self.ref_path[2]
        medium_vs = self.ref_path[3]

        # # end ref path
        # end_point_num = int(Para.END_LENGTH * Para.METER_POINT_NUM)
        # end_xs = np.linspace(
        #     self.goal_point[0],
        #     self.goal_point[0] + cos(self.goal_point[2] * pi / 180) * Para.END_LENGTH,
        #     end_point_num
        # )
        # end_ys = np.linspace(
        #     self.goal_point[1],
        #     self.goal_point[1] + sin(self.goal_point[2] * pi / 180) * Para.END_LENGTH,
        #     end_point_num
        # )
        # end_phis = np.ones(end_point_num) * self.goal_point[2]
        # end_vs = np.ones(end_point_num) * self.exp_v

        # xs = np.concatenate((start_xs, medium_xs, end_xs), axis=0)
        # ys = np.concatenate((start_ys, medium_ys, end_ys), axis=0)
        # phis = np.concatenate((start_phis, medium_phis, end_phis), axis=0)
        # vs = np.concatenate((start_vs, medium_vs, end_vs), axis=0)

        xs = np.concatenate((start_xs, medium_xs,), axis=0)
        ys = np.concatenate((start_ys, medium_ys,), axis=0)
        phis = np.concatenate((start_phis, medium_phis,), axis=0)
        vs = np.concatenate((start_vs, medium_vs,), axis=0)
        return xs, ys, phis, vs

    def get_random_n_points(self, n):
        start_index = np.random.randint(20, Para.INIT_UP_INDEX)
        untransformed_points = self.get_n_future_point(self.idx2point(start_index), start_index, n)
        lane_shape = 0 if (self.idx2point(start_index - 10)[2] - self.idx2point(start_index)[2]) * \
                          (self.idx2point(start_index + 10)[2] - self.idx2point(start_index)[2]) > 0 else 1
        self.random_n_points = ref_points_transform(untransformed_points)
        return self.random_n_points, lane_shape

    def idx2point(self, index):
        return self.ref_path[0][index], self.ref_path[1][index], self.ref_path[2][index], self.exp_v

    def idx2whole(self, index):
        return self.whole_path[0][index], self.whole_path[1][index], self.whole_path[2][index], self.exp_v

    def plot_path(self, ax):
        ax.plot(self.ref_path[0], self.ref_path[1], color='red', alpha=1.)

    def find_closest_point(self, ego_pos, ratio=1):
        ego_x, ego_y, ego_phi = ego_pos[0], ego_pos[1], ego_pos[2]

        # area 0
        area_index = 0
        if ego_y < 0:
            closest_point = 0, ego_y, 90, self.exp_v
            min_dist = np.square(ego_x)
        else:
            closest_point = 0, 0, 90, self.exp_v
            min_dist = np.square(ego_x) + np.square(ego_y)

        # area 1
        path_len = len(self.random_n_points[0])
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_path_x, reduced_path_y = self.random_n_points[0][reduced_idx], self.random_n_points[1][reduced_idx]
        dists = np.square(ego_x - reduced_path_x) + np.square(ego_y - reduced_path_y)
        min_reduced_index = np.argmin(dists)
        index = min_reduced_index

        if dists[min_reduced_index] < min_dist:
            area_index = 1
            closest_point = self.random_n_points[:, index]
            min_dist = dists[min_reduced_index]

        # area 2
        k = tan(self.random_n_points[2][-1] * np.pi / 180)
        b = reduced_path_y[-1] - k * reduced_path_x[-1]
        if ego_y > - 1 / k * (ego_x - reduced_path_x[-1]) + reduced_path_y[-1]:
            x = (k * ego_y + ego_x - k * b) / (k ** 2 + 1)
            y = (k ** 2 * ego_y + k * ego_x + b) / (k ** 2 + 1)
            dist = np.square(ego_x - x) + np.square(ego_y - y)
        else:
            x = reduced_path_x[-1]
            y = reduced_path_y[-1]
            dist = np.square(ego_x - x) + np.square(ego_y - y)

        if dist < min_dist:
            area_index = 2
            closest_point = x, y, self.goal_point[2], self.exp_v
            min_dist = dist

        k = np.tan(closest_point[2] * np.pi / 180)
        b = closest_point[1] - k * closest_point[0]
        x = (k * ego_y + ego_x - k * b) / (k ** 2 + 1)
        y = (k ** 2 * ego_y + k * ego_x + b) / (k ** 2 + 1)

        return [x, y, closest_point[2]], area_index, k, b

    def get_n_future_point(self, closest_point, index, n, ratio=Para.REF_RATIO, dt=0.1):
        future_n_x, future_n_y, future_n_phi, future_n_v = [], [], [], []
        cl_x, cl_y, cl_phi, cl_v = closest_point[0], closest_point[1], closest_point[2], closest_point[3]
        ds = self.exp_v * dt

        if index == -2:  # area 2
            future_n_x = [cl_x + ds * i * cos(self.goal_point[2] * pi / 180) for i in range(n)]
            future_n_y = [cl_y + ds * i * sin(self.goal_point[2] * pi / 180) for i in range(n)]
            future_n_phi = [cl_phi] * n
        elif index == -1:  # area 0
            future_n_x = [future_n_x] * n
            future_n_y = [cl_y + ds * i for i in range(n)]
            future_n_phi = [cl_phi] * n
        else:  # area 1
            x, y = cl_x, cl_y
            for _ in range(n):
                s = 0
                while s < ratio * ds:
                    if index + 1 >= len(self.ref_path[0]):
                        break
                    next_x, next_y, _, _ = self.idx2point(index + 1)
                    s += np.sqrt(np.square(next_x - x) + np.square(next_y - y))
                    x, y = next_x, next_y
                    index += 1
                x, y, phi, v = self.idx2point(index)
                future_n_x.append(x)
                future_n_y.append(y)
                future_n_phi.append(phi)

        future_n_point = np.stack([np.array(future_n_x, dtype=np.float32), np.array(future_n_y, dtype=np.float32),
                                   np.array(future_n_phi, dtype=np.float32)],
                                  axis=0)
        return future_n_point

    def compute_reward(self, ego_state):
        pass

    def reset_ref(self):
        pass


def test_ref():
    start_point = 0, 0, 90
    goal_point = 50, 10, 50
    ego_point = 10, 40, 50
    ax = plt.axes([0, 0, 1, 1])
    ax.axis('equal')
    ref_path = ReferencePath(start_point=start_point, goal_point=goal_point)
    closest_point, area_index, index = ref_path.find_closest_point(ego_point)
    n_future_points = ref_path.get_n_future_point(closest_point, index, 10)
    print(n_future_points, n_future_points.size)
    ref_path.plot_path(ax)
    ax.scatter(n_future_points[0], n_future_points[1], color='black')
    ax.scatter(closest_point[0], closest_point[1], color='blue')
    ax.scatter(ego_point[0], ego_point[1], color='green')
    plt.show()
    # print(closest_point)
    # print(ref_path.judge_area_index(20, 11))
    # ref_path.construct_ref_path()


if __name__ == '__main__':
    test_ref()
