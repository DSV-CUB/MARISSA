import numpy as np
import os
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

class _R():
    def __init__(self, packages=[]):
        self.setup = {}
        rpy2.robjects.numpy2ri.activate()

        for package in packages:
            importr(package)
        return


class Setup_NbClust(_R):
    # in R the packacge NbClust must be installed
    def __init__(self, **kwargs):
        super().__init__(["NbClust"])

        self.setup["distance"] = kwargs.get("setup_distance", "euclidean")
        self.setup["min_nc"] = kwargs.get("setup_min_nc", 2)
        self.setup["max_nc"] = kwargs.get("setup_max_nc", 15)
        self.setup["method"] = kwargs.get("setup_method", "kmeans")
        self.setup["index"] = kwargs.get("setup_index", "all")

        self.index_names = []
        self.result = None
        return

    def run_index_calculator(self, data, **kwargs):
        # https://www.rdocumentation.org/packages/NbClust/versions/3.0.1/topics/NbClust
        self.setup["distance"] = kwargs.get("setup_distance", self.setup["distance"])
        self.setup["min_nc"] = kwargs.get("setup_min_nc", self.setup["min_nc"])
        self.setup["max_nc"] = kwargs.get("setup_max_nc", self.setup["max_nc"])
        self.setup["method"] = kwargs.get("setup_method", self.setup["method"])
        self.setup["index"] = kwargs.get("setup_index", self.setup["index"])

        m = np.array(data)
        if len(np.shape(m)) == 1:
            m = np.expand_dims(m, axis=1)

        if len(np.shape(m)) == 2:
            try:
                # run NbClust
                nrows, ncols = m.shape
                mr = ro.r.matrix(m, nrow=nrows, ncol=ncols)
                ro.r.assign("M", mr)
                ro.r('''res <- NbClust(M, distance="''' + self.setup["distance"] + '''", min.nc=''' + str(self.setup["min_nc"]) + ''', max.nc=''' + str(self.setup["max_nc"]) + ''', method="''' + self.setup["method"] + '''", index="''' + self.setup["index"] + '''")''')
                self.result = dict(ro.r["res"].items())
                # close open R plots
                ro.r('''try(dev.off(dev.list()["RStudioGD"]),silent=TRUE)''')
                ro.r('''try(dev.off(),silent=TRUE)''')

                self.index_names = str(ro.r('''col <- colnames(res$All.index)''')).replace(" ", "").replace("\n", "").replace("\t", "").split("\"")
                self.index_names = [self.index_names[i] for i in range(len(self.index_names)) if not self.index_names[i].startswith("[") and not self.index_names[i] == ""]
            except:
                self.result = None
        else:
            self.result = None
        return self.result

    def extract_optimal_c_index(self):
        result = self.result["Best.nc"][0,:]

        np.nan_to_num(result)
        result[result < self.setup["min_nc"]] = 0
        result[result > self.setup["max_nc"]] = 0
        result[(result - result.astype(int)) != 0] = 0

        #result = np.array([result[0, i] if result[0, i] > 0 else result[1, i] for i in range(np.shape(result)[1])])

        return result.astype(int), self.index_names

    def extract_optimal_c_majority(self):
        return np.argmax(np.bincount(self.extract_optimal_c_index()[0][self.extract_optimal_c_index()[0]>0].astype(int))), "MAJORITY"

    def extract_optimal_c_mean(self):
        return np.round(np.mean(self.extract_optimal_c_index()[0][self.extract_optimal_c_index()[0]>0]), 0).astype(int), "MEAN"

    def extract_optimal_c_median(self):
        return np.round(np.median(self.extract_optimal_c_index()[0][self.extract_optimal_c_index()[0]>0]), 0).astype(int), "MEDIAN"

    def extract_optimal_c_all(self):
        result, info = self.extract_optimal_c_index()

        result = result.tolist()

        r, i = self.extract_optimal_c_majority()
        result.append(r)
        info.append(i)

        r, i = self.extract_optimal_c_mean()
        result.append(r)
        info.append(i)

        r, i = self.extract_optimal_c_median()
        result.append(r)
        info.append(i)

        return result, info

class Setup_FAMD(_R):
    # in R the packacge NbClust must be installed
    def __init__(self):
        super().__init__(["FactoMineR", "factoextra"])

        self.result = None
        return

    def run(self, data, path_out, **kwargs):

        col_names_in = kwargs.get("columnnames", ["C" + str(i) for i in range(np.shape(data)[1])])
        col_names_num = []
        col_names_sym = []

        M_num = []
        M_sym = []
        for i in range(np.shape(data)[1]):
            col_data = data[:,i].flatten()
            try:
                col_data = col_data.astype(float)
                M_num.append(col_data)
                col_names_num.append(col_names_in[i])
            except:
                col_data = col_data.astype(str)
                for j in range(len(col_data)):
                    col_data[j] = "q" + str(len(M_sym) + 1) + "_" + col_data[j]
                M_sym.append(col_data)
                col_names_sym.append(col_names_in[i])
        M_num = np.transpose(M_num)
        M_sym = np.transpose(M_sym)
        col_names = col_names_num + col_names_sym

        # run in R
        if len(M_num) > 0:
            nrows, ncols = M_num.shape
            mnum = ro.r.matrix(M_num, nrow=nrows, ncol=ncols)
            ro.r.assign("Mnum", mnum)
            ro.r('''Mdfnum <- as.data.frame(Mnum)''')

        if len(M_sym) > 0:
            nrows, ncols = M_sym.shape
            msym = ro.r.matrix(M_sym, nrow=nrows, ncol=ncols)
            ro.r.assign("Msym", msym)
            ro.r('''Mdfsym <- as.data.frame(Msym)''')

        if len(M_num) > 0 and len(M_sym) > 0:
            ro.r('''Mdf <- cbind(Mdfnum, Mdfsym)''')
        elif len(M_num) > 0:
            ro.r('''Mdf <- Mdfnum''')
        else:
            ro.r('''Mdf <- Mdfsym''')

        ro.r('''colnames(Mdf) = c("''' + "\", \"".join(col_names) + '''")''')
        ro.r('''res.famd <- FAMD(Mdf, graph=FALSE)''')

        # Relationship Square
        ro.r('''png("''' + os.path.join(path_out, "R_FAMD_relationship_square.png").replace("\\", "\\\\") + '''", width = 10, height = 10, units = 'in', res = 300)''')
        ro.r('''print(fviz_famd_var(res.famd, repel = TRUE))''')
        ro.r('''dev.off()''')

        # Contribution Dim 1
        ro.r('''png("''' + os.path.join(path_out, "R_FAMD_contribution_Dim_1.png").replace("\\", "\\\\") + '''", width = 10, height = 10, units = 'in', res = 300)''')
        ro.r('''print(fviz_contrib(res.famd, "var", axes = 1))''')
        ro.r('''dev.off()''')

        # Contribution Dim 2
        ro.r('''png("''' + os.path.join(path_out, "R_FAMD_contribution_Dim_2.png").replace("\\", "\\\\") + '''", width = 10, height = 10, units = 'in', res = 300)''')
        ro.r('''print(fviz_contrib(res.famd, "var", axes = 2))''')
        ro.r('''dev.off()''')

        if len(M_num) > 0 and np.shape(M_num)[1] > 1:
            # Correlation Circle
            ro.r('''png("''' + os.path.join(path_out, "R_FAMD_correlation_circle.png").replace("\\", "\\\\") + '''", width = 10, height = 10, units = 'in', res = 300)''')
            ro.r('''print(fviz_famd_var(res.famd, "quanti.var", repel = TRUE, col.var = "black"))''')
            ro.r('''dev.off()''')

        if len(M_sym) > 0:
            # Representation of categories
            ro.r('''png("''' + os.path.join(path_out, "R_FAMD_representation_categories.png").replace("\\", "\\\\") + '''", width = 10, height = 10, units = 'in', res = 300)''')
            ro.r('''print(fviz_famd_var(res.famd, "quali.var", repel = TRUE, col.var = "black"))''')
            ro.r('''dev.off()''')

        # Representation of Individuals
        ro.r('''png("''' + os.path.join(path_out, "R_FAMD_representation_individuals.png").replace("\\", "\\\\") + '''", width = 10, height = 10, units = 'in', res = 300)''')
        ro.r('''print(fviz_famd_ind(res.famd, repel = TRUE))''')
        ro.r('''dev.off()''')

        self.result = dict(ro.r["res.famd"].items())

        # close open R plots
        ro.r('''try(dev.off(dev.list()["RStudioGD"]),silent=TRUE)''')
        ro.r('''try(dev.off(),silent=TRUE)''')

        return self.result

if __name__ == "__main__":
    data = np.transpose(np.array([[2,5,3,4,1,6], [4.5,4.5,1,1,1,1], [4,4,2,2,1,2], ["A","C","B","B","A","C"], ["B","B","B","B","A","A"], ["C","C","B","B","A","A"]]))
    famd = Setup_FAMD()
    famd.run(data, r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\FAMD Test")
