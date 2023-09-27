import pickle
import os
from marissa.toolbox.tools import tool_general


class Inheritance:
    def __init__(self):
        self.path = os.path.realpath(__file__)
        self.name = ""
        return

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and not tool_general.check_class_has_method(self, key) and not key == "path":
                exec("self." + key + " = kwargs.get(\"" + key + "\", None)")
        return True

    def save(self, path=None, timestamp=""):
        if path is None:
            save_to = os.path.join(self.path, self.name + ".pickle")
        else:
            save_to = os.path.join(path, self.name + ".pickle")

        if save_to.endswith(".pickle"):
            with open(save_to, 'wb') as file:
                pickle.dump(self, file)
                file.close()
        return True

    def load(self, path=None):
        if path is None:
            load_from = os.path.join(self.path, self.name + ".pickle")
        else:
            load_from = path

        with open(load_from, 'rb') as file:
            obj = pickle.load(file)
            file.close()

        if type(self) == type(obj):
            for key in obj.__dict__:
                if key in self.__dict__:
                    self.__dict__[key] = obj.__dict__[key]
        else:
            raise TypeError("The loaded object is not a configuration object")

        #self.__dict__.clear()
        #self.__dict__.update(obj.__dict__)

        return True

    def reset(self):
        self.__init__()
        return

