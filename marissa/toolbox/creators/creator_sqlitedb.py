import sqlite3
import numpy as np
import datetime
import os
import io
import zlib
import pickle

class Inheritance:
    def __init__(self, path):
        self.path = path
        self.dbconn = None
        self.dbcur = None
        return

    def _get_timestamp(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _connect(self, path=None):
        if not path is None:
            self.path = path

        if not self.path is None:
            #sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
            sqlite3.register_adapter(np.array, self._adapt_array)
            sqlite3.register_adapter(np.ndarray, self._adapt_array)
            sqlite3.register_converter("ARRAY", self._convert_array)
            sqlite3.register_adapter(list, self._adapt_list)
            sqlite3.register_converter("LIST", self._convert_list)
            self.dbconn = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
            self.dbcur = self.dbconn.cursor()
        return

    def _disconnect(self):
        try:
            self.dbconn.close()
        except:
            pass

        self.dbconn = None
        self.dbcur = None
        return

    @staticmethod
    def _adapt(input):
        try:
            bio = io.BytesIO()
            if type(input) == list:
                pickle.dump(input, bio)
            else: #array
                np.save(bio, np.array(input))
            bio.seek(0)
            bd = bio.read()
            result = zlib.compress(bd) # compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS)
        except:
            result = "NULL"
        return result

    @staticmethod
    def _convert(input, what="list"):
        bio = io.BytesIO(input)
        bio.seek(0)
        bio = io.BytesIO(zlib.decompressobj().decompress(bio.read()))
        try:
            if what == "list":
                bio.seek(0)
                result = pickle.loads(bio.read())
            else: # array
                result = np.load(bio, allow_pickle=True)
        except:
            result = None
        return result

    def _adapt_array(self, input):
        return self._adapt(input)

    def _convert_array(self, input):
        return np.array(self._convert(input, "array"))

    def _adapt_list(self, input):
        return self._adapt(input)

    def _convert_list(self, input):
        return self._convert(input, "list")

    def execute(self, command):
        self._connect()
        if not self.dbcur is None:
            if type(command) == tuple:
                self.dbcur.execute(command[0], command[1])
            else:
                self.dbcur.execute(command)
            self.dbconn.commit()
        self._disconnect()
        return

    def select(self, command, return_description=False):
        self._connect()
        if not self.dbcur is None:
            if type(command) == tuple:
                self.dbcur.execute(command[0], command[1])
            else:
                self.dbcur.execute(command)
            description = [x[0] for x in self.dbcur.description]
            result = self.dbcur.fetchall()
            self._disconnect()
        else:
            result = []
            description = None

        if return_description:
            return result, description
        else:
            return result

    def execute_extern(self, path, command):
        self._connect()
        self.dbcur.execute("ATTACH DATABASE '" + path + "' AS externdb")
        self.dbconn.commit()
        self.dbcur.execute(command)
        self.dbconn.commit()
        self.dbcur.execute("DETACH DATABASE externdb")
        self.dbconn.commit()
        self._disconnect()
        return

    def import_from_extern(self, path, table, overwrite=False):
        counter = int(self.select("SELECT COUNT(*) FROM " + table)[0][0])
        self.execute_extern(path, "INSERT OR " + ("REPLACE" if overwrite else "IGNORE") + " INTO main." + table + " SELECT * FROM externdb." + table)
        counter = int(self.select("SELECT COUNT(*) FROM " + table)[0][0]) - counter
        return counter

    def plot_database_structure(self, path_out, filename="dbstruct"):
        worked = True
        try:
            os.add_dll_directory("C:/Program Files/Graphviz/bin")
        except:
            worked = False

        try:
            from sqlalchemy_schemadisplay import create_schema_graph
            from sqlalchemy import MetaData

            graph = create_schema_graph(metadata=MetaData("sqlite:///"+self.path))
            graph.write_png(os.path.join(path_out, filename + ".png"))

        except:
            worked = False

        return worked

    def vacuum(self):
        self._connect()
        self.dbconn.execute("VACUUM")
        self._disconnect()
        return
