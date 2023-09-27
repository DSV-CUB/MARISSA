import numpy as np
import mock

class Model:
    def __init__(self, **kwargs):
        self.n_clusters = kwargs.get("bins", 10)
        self.result = None
        return

    def _fit(self, x):
        labels = np.ones((len(x), 1)).flatten()
        if self.n_clusters == 1:
            centers = [np.mean(x)]
        else:
            centers = []
            indexsort = np.argsort(x,axis=0)
            x_len = len(x)

            cluster_size = x_len / self.n_clusters
            size_done = int(0)

            for i in range(self.n_clusters):
                total_size = int(np.round((i+1) * cluster_size))
                labels[indexsort[size_done:total_size]] = i
                centers.append(np.mean(x[indexsort[size_done:total_size]], axis=0).flatten())
                size_done = np.copy(total_size)

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
    num = 10

    for i in range(num):
        rv = (i+1) + (np.random.rand(100) - 0.5)
        values = values + rv.tolist()

    model=Model(bins=num)
    result = model.run(values)
    a = 0