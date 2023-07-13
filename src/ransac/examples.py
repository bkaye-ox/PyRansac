import ransac
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEN = 300
RNG = np.random.default_rng()


def gen_data():
    t = np.linspace(0., 3., LEN)

    set_a = list(RNG.choice(tuple(k for k in range(LEN)), 50, replace=False))
    set_b = [k for k in range(LEN) if k not in set_a]
    t1 = t[set_b].reshape(-1,)
    t2 = t[set_a].reshape(-1,)

    y1 = 3*np.sin(t1) + 2*t1 + RNG.normal(size=t1.shape)
    y2 = 6*t2

    y_ag = np.concatenate([y1, y2])
    t_ag = np.concatenate([t1, t2])

    return t_ag, y_ag


def fit(X_tr, y_tr, C=10, eps=1e-2):
    regr = make_pipeline(StandardScaler(), SVR(C=C, epsilon=eps))
    estimator = regr.fit(X_tr, y_tr)
    return estimator


X, y = gen_data()

est = ransac.fit_ransac(X.reshape(-1, 1), y.reshape(-1,), fit,
                        N=200,
                        m=10,
                        inlier_dist=1.,)

X_t, y_t = ransac.grid_predict(est, X, [0.1], return_meta=False)
