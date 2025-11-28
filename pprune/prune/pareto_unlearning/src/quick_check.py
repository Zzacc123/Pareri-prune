import sys
import numpy as np

sys.path.append('.')

from pareto_core import get_pareto_front, calculate_hypervolume, find_knee_point


def run() -> None:
    f = np.array([0.1, 0.4, 0.3, 0.8, 0.7], dtype=np.float32)
    r = np.array([0.9, 0.5, 0.6, 0.2, 0.3], dtype=np.float32)
    idx = get_pareto_front(f, r)
    print('front indices:', idx.tolist())
    hv = calculate_hypervolume(f[idx], r[idx])
    print('hypervolume:', float(hv))
    kp = find_knee_point(f[idx], r[idx])
    print('knee index in front:', int(kp))


if __name__ == '__main__':
    run()

