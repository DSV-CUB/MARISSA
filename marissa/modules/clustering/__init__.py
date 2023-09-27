class Model:
    def __init__(self, **kwargs):
        # do setups
        self.bins = kwargs.get("bins", 1)
        return

    def run(self, x, return_indeces=False):
        # calculate clustering for flatten x input
        # sort the cluster centers in ascending order, like with np.sort()

        y = eval("do something with x")
        y_indeces = eval("get the binning number for each value in x")

        if return_indeces:
            result = (y, y_indeces)
        else:
            result = y
        return result