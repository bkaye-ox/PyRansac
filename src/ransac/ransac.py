import numpy as np

RNG = np.random.default_rng()


def count_inliers(X, y, estimator, inlier_dist):

    y_hat = estimator.predict(X)
    error = y_hat - y
    inlier_count = sum(1 if abs(e) <= inlier_dist else 0 for e in error)

    return inlier_count


def get_inliers(X, y, est, inlier_dist):
    y_hat = est.predict(X)
    error = y_hat - y
    indices = np.array(tuple(k for k, e in enumerate(error)
                             if abs(e) <= inlier_dist))
    return X[indices, ...], y[indices]


def get_samples(X, y, N, m):
    n_train = len(y)

    permutations = [RNG.choice(tuple(k for k in range(
        n_train)), size=m, replace=False) for _ in range(N)]

    for indices in permutations:
        yield X[indices, ...], y[indices]


def get_train_data(df, X: list[str], y: str):
    train_data = df[X + [y]].dropna().to_numpy(dtype=float)
    Xd, yd = train_data[:, :-1].reshape(-1, len(X)), train_data[:, -1]

    return Xd, yd


def get_convex_hull(X, x_z):
    try:
        from scipy.spatial import ConvexHull
    except: 
        raise Exception('requires scipy!')
    hull = ConvexHull(X)
    edges = X[hull.vertices, ...]
    from scipy.spatial import Delaunay
    hull_d = Delaunay(edges)
    mask = [hull_d.find_simplex(x_z) >= 0]
    return mask


def grid_predict(est, X: np.array, steps: list[float], return_meta=True):
    '''
    return X_T, y_T, optional(list[X ranges], y.shape, nan_mask)
    '''

    mins_ = np.min(X, 0).reshape(-1,)
    maxs_ = np.max(X, 0).reshape(-1,)

    axis_ranges = tuple(np.arange(lower, upper, st)
                        for upper, lower, st in zip(maxs_, mins_, steps))
    u_list = np.meshgrid(*axis_ranges)
    x_list = [u.reshape(-1, 1) for u in u_list]

    # generate triplets of every permutation
    X_test = np.concatenate(x_list, axis=1)

    y_test = est.predict(X_test)

    if return_meta:
        nan_mask = [not m for m in get_convex_hull(X, X_test)[0]]
        return X_test, y_test, (axis_ranges, u_list[0].shape, nan_mask)
    else:
        return X_test, y_test


def fit_ransac(X, y, regressor, N, m, inlier_dist, inlier_count_min):
    samples = get_samples(X, y, N, m)

    inlier_max = 0
    best_estimator = None

    for X_s, y_s in samples:
        est = regressor(X_s, y_s)
        count = count_inliers(X, y, est, inlier_dist)

        if count > inlier_max:
            inlier_max = count
            best_estimator = est

        if inlier_max > inlier_count_min:
            break

    X_in, y_in = get_inliers(X, y, best_estimator, inlier_dist)
    return regressor(X_in, y_in)
