import os
import pickle
import numpy as np
import datetime
from marissa.toolbox.tools import tool_general

class Inheritance:
    def __init__(self, SOPinstanceUID, version=None):
        if os.path.isfile(SOPinstanceUID):
            self.SOPinstanceUID = None
            self.mask = None
            self.array = None
            self.parameters = []
            self.parameters_values = []
            self.value_progression = []
            self.scaling = []
            self.creation = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            self.version = version
            self.load(SOPinstanceUID)
        else:
            self.SOPinstanceUID = SOPinstanceUID
            self.mask = None
            self.array = None
            self.parameters = []
            self.parameters_values = []
            self.value_progression = []
            self.scaling = []
            self.creation = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            self.version = version
        return

    def save(self, path_out):
        obj = {}
        for key in vars(self).keys():
            obj[key] = eval("self." + key)

        path_save = os.path.join(path_out, self.SOPinstanceUID + "_" + tool_general.string_stripper(self.creation, []) + ".marissadata")
        file = open(path_save, "wb")
        pickle.dump(obj, file)
        file.close()
        return path_save

    def load(self, path_in):
        if path_in.endswith(".marissadata"):
            file = open(path_in, "rb")
            obj = pickle.load(file)
            file.close()

            for key in obj.keys():
                exec("self." + key + " = obj[\"" + key + "\"]")

            result = True
        else:
            result = False
        return result

    def get_standardized_values(self):
        return None if self.value_progression is None else np.copy(self.value_progression[-1])

    def get_standardized_values_scaled(self):
        sv = self.get_standardized_values()
        if sv is None:
            result = None
        elif len(self.scaling) == 0:
            result = sv
        else:
            result = (sv - self.scaling[0]) / self.scaling[1]
        return result

    def get_standardized_array(self):
        try:
            result = np.copy(self.array)
            result[self.mask.astype(bool)] = self.get_standardized_values_scaled()
        except:
            result = None
        return result