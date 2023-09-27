import numpy as np
import mock

class Model:
    def __init__(self, **kwargs):
        self.n_clusters = kwargs.get("bins", 10)
        self.result = None
        return

    def _fit(self, x):
        minval = np.min(x, axis=0)
        maxval = np.max(x, axis=0)

        labels = np.ones((len(x), 1)).flatten()
        if self.n_clusters == 1:
            centers = [minval + (maxval - minval) / 2]
        else:
            step = (maxval-minval) / self.n_clusters
            centers = []
            lower_bound = minval
            upper_bound = minval + step

            for i in range(self.n_clusters):
                centers.append(lower_bound+step/2)
                if i+1 == self.n_clusters:
                    labels[np.all(x>=lower_bound,axis=1)] = i
                else:
                    labels[np.logical_and(np.all(x>=lower_bound,axis=1), np.all(x<upper_bound,axis=1))] = i
                    lower_bound = lower_bound + step
                    upper_bound = upper_bound + step

        self.result = mock.Mock()
        self.result.cluster_centers_ = np.array(centers)
        self.result.labels_ = labels.astype(int)

        return True

    def _get_centers_all(self):
        return self.result.cluster_centers_[self.result.labels_.tolist(), :]

    def _get_centers(self):
        return self.result.cluster_centers_

    def _get(self):
        centers = self._get_centers_all()
        unique_centers = np.unique(centers)

        y = []
        y_indeces = []
        for i in range(len(unique_centers)):
            indeces = np.argwhere(centers == unique_centers[i])[:,0]
            y.append(self.x[indeces])
            y_indeces.append(indeces)
        return y, y_indeces

    def run(self, x, return_indeces=False):
        values = np.reshape(np.array(x).flatten().astype(np.float), (-1, 1))
        self._fit(values)

        y = []
        for i in range(len(np.unique(self.result.labels_))):
            y.append(values[self.result.labels_ == i])

        y_indeces = self.result.labels_

#        centers = self._get_centers_all()
#        unique_centers = np.unique(centers)

#        y = []
#        y_indeces = np.zeros((len(x),1)).flatten()

#        for i in range(len(unique_centers)):
#            indeces = np.argwhere(centers == unique_centers[i])[:,0]
#            y_indeces[indeces] = i
#            y.append(values[indeces])

        if return_indeces:
            result = (y, y_indeces)
        else:
            result = y
        return result

if __name__ == "__main__":
    values = []
    num = 3

    for i in range(num):
        rv = (i+1) + (np.random.rand(100) - 0.5)
        values = values + rv.tolist()

    model=Model(bins=3)
    result = model.run(values)
    a=0