import torch


def get_min_dists(ego_poses, control_points):
    x0, x1, x2, x3, y0, y1, y2, y3 = coord_split(control_points)
    # m, n = ego_poses[:, 0], ego_poses[:, 1]
    m, n = torch.split(ego_poses, 1, dim=1)
    d0 = torch.square(x0 - m) + torch.square(y0 - n)
    d1 = torch.mul(x1 - x0, x0 - m) + torch.mul(y1 - y0, y0 - n)
    d2 = \
        torch.mul(x0 - 2 * x1 + x2, x0 - m) + torch.mul(y0 - 2 * y1 + y2, y0 - n) + \
        2 * torch.square(x1 - x0) + 2 * torch.square(y1 - y0)
    d3 = torch.mul(x0 - 2 * x1 + x2, x1 - x0) + torch.mul(y0 - 2 * y1 + y2, y1 - y0)
    d4 = torch.square(x0 - 2 * x1 + x2) + torch.square(y0 - 2 * y1 + y2)

    # get the minimum of f(t) = d4 * t^4 + 4 * d3 * t^3 + 2 * d2 * t^2 + 4 * d1 * t + d0
    # get the differentiation of f(t): f^'(t) = 4 * (d4 * t^3 + 3 * d3 * t^2 + d2 * t + d1 = 0)
    # get the root of the cubic equation: x^3 + p*x^2 + qx + r = 0
    p, q, r = torch.div(3 * d3, d4), torch.div(d2, d4), torch.div(d1, d4)

    # get the 3 roots t of the cubic bezier, and filter in the range [0, 1], padding 0 if not in
    roots = get_cubic_root(p, q, r)
    zero_paddings = torch.zeros_like(roots)
    filtered_roots = torch.where(torch.logical_and(roots > 0, roots < 1), roots, zero_paddings)

    # add the 0 and 1 boundaries
    poss_min_ts = torch.cat((filtered_roots,
                             torch.zeros_like(p),
                             torch.ones_like(p)),
                            dim=1)

    # get the minimum dist
    min_dists, _ = index2dist(poss_min_ts, d4, d3, d2, d1, d0)
    return min_dists, _


# get the root of a cubic polynomials x^3 + p*x^2 + qx + r = 0
def get_cubic_root(p, q, r):
    a = torch.add(3 * q, -torch.square(q)) / 3
    # a = 3 * q - torch.square(q) / 3
    b = (2 * torch.pow(q, 3) - 9 * torch.mul(p, q) + 27 * r) / 27

    ind = torch.square(b) / 4 + torch.pow(a, 3) / 27
    # A = torch.pow(-b / 2 + torch.sqrt(ind), 1 / 3)
    # B = torch.pow(-b / 2 - torch.sqrt(ind), 1 / 3)
    A = (-b / 2 + torch.sqrt(ind)).sign() * (-b / 2 + torch.sqrt(ind)).abs().pow(1 / 3)
    B = (-b / 2 - torch.sqrt(ind)).sign() * (-b / 2 - torch.sqrt(ind)).abs().pow(1 / 3)

    if ind > 0:
        y1 = A + B
        y2 = torch.zeros_like(A)
        y3 = torch.zeros_like(A)
    elif ind < 0:
        phi = torch.arccos(torch.where(b > 0, -torch.sqrt(torch.div(torch.square(b) / 4, -torch.square(a) / 27)),
                                       torch.sqrt(torch.div(torch.square(b) / 4, -torch.square(a) / 27))))
        y1 = 2 * torch.mul(torch.sqrt(-a / 3), torch.cos(phi / 3 + torch.pi / 3))
        y2 = 2 * torch.mul(torch.sqrt(-a / 3), torch.cos(phi * 2 / 3 + torch.pi / 3))
        y3 = 2 * torch.mul(torch.sqrt(-a / 3), torch.cos(phi * 2 / 3 + torch.pi / 3))
    else:
        assert ind == 0, "wrong indicator"
        if b > 0:
            y1 = -2 * torch.sqrt(-a / 3)
            y2 = torch.sqrt(-a / 3)
            y3 = torch.sqrt(-a / 3)
        elif b < 0:
            y1 = 2 * torch.sqrt(-a / 3)
            y2 = -torch.sqrt(-a / 3)
            y3 = -torch.sqrt(-a / 3)
        else:
            assert b == 0, "wrong b"
            y1 = torch.zeros_like(A)
            y2 = torch.zeros_like(A)
            y3 = torch.zeros_like(A)

    x1, x2, x3 = y1 - p / 3, y2 - p / 3, y3 - p / 3

    return torch.cat((x1, x2, x3), dim=1)


def index2dist(ts, d4, d3, d2, d1, d0):
    # f(t) = d4 * t^4 + 4 * d3 * t^3 + 2 * d2 * t^2 + 4 * d1 * t + d0
    poss_min_dists = torch.mul(d4, torch.pow(ts, 4)) + 4 * torch.mul(d3, torch.pow(ts, 3)) + \
                     2 * torch.mul(d2, torch.square(ts)) + 4 * torch.mul(d1, ts) + d0
    return torch.min(poss_min_dists, dim=1), torch.argmin(poss_min_dists, dim=1)


def coord_split(control_points):
    # x0, y0, x1, y1, x2, y2, x3, y3 = control_points[:, 0], control_points[:, 0], \
    #                                  control_points[:, 0], control_points[:, 0], \
    #                                  control_points[:, 0], control_points[:, 0], \
    #                                  control_points[:, 0], control_points[:, 0]
    x0, y0, x1, y1, x2, y2, x3, y3 = torch.split(control_points, 1, dim=1)
    return x0, x1, x2, x3, y0, y1, y2, y3


def test():
    x0, y0, phi0 = torch.zeros((1, 1)), torch.zeros((1, 1)), torch.zeros((1, 1))
    x3, y3, phi3 = 5 * torch.ones((1, 1)), 5 * torch.ones((1, 1)), torch.pi * 0.5 * torch.ones((1, 1))
    weight = 0.7
    x1 = x0 * ((torch.cos(phi0) ** 2) * (1 - weight) + torch.sin(phi0) ** 2) + y0 * (-torch.sin(phi0) * torch.cos(phi0) * weight) + x3 * (
                (torch.cos(phi0) ** 2) * weight) + y3 * (torch.sin(phi0) * torch.cos(phi0) * weight)
    y1 = x0 * (-torch.sin(phi0) * torch.cos(phi0) * weight) + y0 * (torch.cos(phi0) ** 2 + (torch.sin(phi0) ** 2) * (1 - weight)) + x3 * (
                torch.sin(phi0) * torch.cos(phi0) * weight) + y3 * ((torch.sin(phi0) ** 2) * weight)
    x2 = x0 * (torch.cos(phi3) ** 2) * weight + y0 * (torch.sin(phi3) * torch.cos(phi3) * weight) + x3 * (
                (torch.cos(phi3) ** 2) * (1 - weight) + torch.sin(phi3) ** 2) + y3 * (-torch.sin(phi3) * torch.cos(phi3) * weight)
    y2 = x0 * (torch.sin(phi3) * torch.cos(phi3) * weight) + y0 * ((torch.sin(phi3) ** 2) * weight) + x3 * (
                -torch.sin(phi3) * torch.cos(phi3) * weight) + y3 * (torch.cos(phi3) ** 2 + (torch.sin(phi3) ** 2) * (1 - weight))

    ego_poses = torch.tensor([[5, 5]])
    control_points = torch.cat((x0, y0, x1, y1, x2, y2, x3, y3), dim=1)

    print(get_min_dists(ego_poses, control_points))


if __name__ == '__main__':
    test()