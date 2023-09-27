import numpy as np
import mock
from sklearn.cluster import AgglomerativeClustering

class Model:
    def __init__(self, **kwargs):
        self.model = AgglomerativeClustering()
        self.model.set_params(n_clusters=kwargs.get("bins", 10), linkage="complete")
        self.result = None
        return

    def _fit(self, x):
        fit = self.model.fit(x)

        unique = np.unique(fit.labels_)
        centers = []

        for u in unique:
            centers.append(np.mean(x[fit.labels_==u,:],axis=0).flatten())

        centers = np.array(centers)
        labels = fit.labels_.astype(int)
        _, order = np.unique(centers, return_index=True)
        centers = centers[order,:]

        labels_ordered = -1 * np.ones(np.shape(labels))

        counter = int(0)
        for i in order.astype(int):
            labels_ordered[labels==i] = counter
            counter = counter + 1

        self.result = mock.Mock()
        self.result.labels_ = labels_ordered.astype(int)
        self.result.cluster_centers_ = centers
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