import sqlite3
import numpy as np
import datetime
import os
import pickle
import pydicom
import fnv
import zlib
import traceback
import copy

from scipy.stats import linregress

from marissa.toolbox.tools import tool_general, tool_pydicom, tool_hadler
from marissa.modules import clustering as clustering
from marissa.modules import regression as regression
#from marissa.modules.pesf import pesf
from marissa.toolbox.creators import creator_sqlitedb, creator_marissadata

class Configuration:
    def __init__(self):
        self.path = None
        self.name = "MARISSA"
        self.address_range = [0, 4294967295]
        return


class Module(creator_sqlitedb.Inheritance):
    def __init__(self, path):
        super().__init__(path)
        self.name = "MARISSA"
        self.address_range = [0, 4294967295]

        if not os.path.isfile(self.path):
            self.execute("PRAGMA foreign_keys = 1")

            # tbl_info
            # ID = information
            # parameter = value of the information as string
            self.execute("CREATE TABLE IF NOT EXISTS tbl_info (ID TEXT, parameter TEXT);")

            # tbl_data
            # *SOPinstanceUID = SOP Instance UID of DICOM
            # StudyinstanceUID = Study Instance UID of DICOM
            # seriesnumber = seriesnumber of DICOM (represents referring slices of a 3D volume)
            # instancenumber = instancenumber of DICOM within series (necessary for 3D, as the first instance is used as reference for 3D segmentations)
            # data = DICOM data as List of bytes with length of 3 representing zlib compressed data as json, data file meta as json and data preamble
            self.execute("CREATE TABLE IF NOT EXISTS tbl_data (SOPinstanceUID TEXT UNIQUE PRIMARY KEY NOT NULL, StudyinstanceUID TEXT NOT NULL, seriesnumber INTEGER NOT NULL, instancenumber INTEGER NOT NULL, seriesdescription TEXT, identifier TEXT, age TEXT, gender TEXT, size REAL, weight REAL, description TEXT, acquisitiondatetime DATETIME, data LIST NOT NULL, timestamp DATETIME);")

            # tbl_segmentation
            # *segmentationID = unique identifier
            # SOPinstanceUID = SOP Instance UID of DICOM
            # object = name of the contour (like epicardial, endocardial, myocardial, ...)
            # creator = name of the person that created the contour
            # points = array of points defining the contour
            # mask = array representing the segmented area/volume as binary mask
            # dimension = dimension of the contour (2D refers to exact one slice, 3D refers to all slices with the same StudyInstanceUID and series number)
            self.execute("CREATE TABLE IF NOT EXISTS tbl_segmentation (segmentationID INTEGER UNIQUE PRIMARY KEY NOT NULL, SOPinstanceUID TEXT NOT NULL, description TEXT NOT NULL, creator TEXT NOT NULL, points ARRAY NOT NULL, mask ARRAY NOT NULL, dimension INTEGER NOT NULL, timestamp DATETIME, FOREIGN KEY (SOPinstanceUID) REFERENCES tbl_data(SOPinstanceUID));")

            # tbl_parameter
            # *parameterID = unique parameter ID / DICOM tag address
            # description = description of the parameter
            # VR = value representation
            # VM = value multiplicity
            # formula = formula if it is agglomerated by multiple tags
            self.execute("CREATE TABLE IF NOT EXISTS tbl_parameter (parameterID INTEGER UNIQUE PRIMARY KEY NOT NULL, description TEXT UNIQUE, VR TEXT, VM INTEGER, formula TEXT);")

            # tbl_setup
            # *setupID = unique identifier for a setup (HASH value of bins, clustertype, regressiontype, ytype, ordering)
            # bins = number of bins
            # clustertype = cluster methods (i.e. kmeans)
            # regressiontype = regression method (i.e. linear), only for numeric errorparameters valid
            # ytype = "absolute" or "percentage"
            # mode = indipendent, cascaded or ensemble
            self.execute("CREATE TABLE IF NOT EXISTS tbl_setup (setupID INTEGER PRIMARY KEY NOT NULL, description TEXT UNIQUE NOT NULL, bins INTEGER NOT NULL, clustertype TEXT NOT NULL, regressiontype TEXT NOT NULL, ytype TEXT NOT NULL, mode TEXT NOT NULL);")

            # tbl_match_setup_parameter
            # *setupID
            # *parameterID
            # ordering
            self.execute("CREATE TABLE IF NOT EXISTS tbl_match_setup_parameter (setupID INTEGER NOT NULL, parameterID INTEGER NOT NULL, ordering INTEGER NOT NULL, PRIMARY KEY (setupID, parameterID), FOREIGN KEY (setupID) REFERENCES tbl_setup(setupID), FOREIGN KEY (parameterID) REFERENCES tbl_parameter(parameterID));")

            # tbl_match_setup_data_segmentation
            # *setupID
            # *SOPinstanceUID
            # *segmentationID
            self.execute("CREATE TABLE IF NOT EXISTS tbl_match_setup_data_segmentation (setupID INTEGER NOT NULL, SOPinstanceUID TEXT NOT NULL, segmentationID INTEGER NOT NULL, PRIMARY KEY (setupID, SOPinstanceUID, segmentationID), FOREIGN KEY (setupID) REFERENCES tbl_setup(setupID), FOREIGN KEY (SOPinstanceUID) REFERENCES tbl_data(SOPinstanceUID), FOREIGN KEY (segmentationID) REFERENCES tbl_segmentation(segmentationID));")

            # STANDARDIZATION tables extra

            # tbl_standardization_setup
            # *setupID = unique identifier for a setup (HASH value of bins, clustertype, regressiontype, ytype, ordering)
            # bins = number of bins
            # clustertype = cluster methods (i.e. kmeans)
            # regressiontype = regression method (i.e. linear), only for numeric errorparameters valid
            # ytype = "absolute" or "percentage"
            # mode = indipendent, cascaded or ensemble
            self.execute("CREATE TABLE IF NOT EXISTS tbl_standardization_setup (setupID INTEGER PRIMARY KEY NOT NULL, description TEXT UNIQUE NOT NULL, bins INTEGER NOT NULL, clustertype TEXT NOT NULL, regressiontype TEXT NOT NULL, ytype TEXT NOT NULL, mode TEXT NOT NULL);")


            # tbl_standardization_data
            # *setupID
            # *SOPinstanceUID = SOP Instance UID of DICOM
            # *segmentationID = identififer of segmentation
            # segmentedvalues = array of segmented values (quantitative)
            # parameters = list of parameter values
            self.execute("CREATE TABLE IF NOT EXISTS tbl_standardization_data (setupID INTEGER NOT NULL, SOPinstanceUID TEXT NOT NULL, segmentationID INTEGER NOT NULL, segmentedvalues ARRAY, parameters LIST, PRIMARY KEY (setupID, SOPinstanceUID, segmentationID) FOREIGN KEY(setupID) REFERENCES tbl_standardization_setup(setupID));")

            # tbl_standardization_parameter
            # * setupID
            # * parameterID
            # ordering
            # description = description of the parameter
            # VR = value representation
            # VM = value multiplicity
            # formula = formula if it is agglomerated by multiple tags
            self.execute("CREATE TABLE IF NOT EXISTS tbl_standardization_parameter (setupID INTEGER NOT NULL, parameterID INTEGER NOT NULL, ordering INTEGER, description TEXT, VR TEXT, VM INTEGER, formula TEXT, PRIMARY KEY(setupID, parameterID), FOREIGN KEY(setupID) REFERENCES tbl_standardization_setup(setupID));")

            # tbl_standardization_match_data_setup_parameter
            # * setupID
            # * parameterID
            # * SOPinstanceUID
            # * segmentationID
            self.execute("CREATE TABLE IF NOT EXISTS tbl_standardization_match_data_setup_parameter (setupID INTEGER NOT NULL, parameterID INTEGER NOT NULL, SOPinstanceUID TEXT NOT NULL, segmentationID INTEGER NOT NULL, VMindex INTEGER, PRIMARY KEY(setupID, parameterID, SOPinstanceUID, segmentationID, VMindex), FOREIGN KEY(SOPinstanceUID, setupID, segmentationID) REFERENCES tbl_standardization_data(SOPinstanceUID, setupID, segmentationID), FOREIGN KEY(parameterID) REFERENCES tbl_standardization_parameter(parameterID));")

            # tbl_standardization
            # *setupID
            # *parameterID
            # *bin = bin number
            # *VMindex = index number if parameter has VM > 1
            # *x = errorparameter value (only for symbolic type) --> list for ensemble mode necessary
            # reference = list of reference values [x, y] where x = parameter value and y = mean of values for that x
            # regressor = regression model as pickled object
            # rmse = root mean squared error
            # p = p-value (only for linear regression model)
            # rsquared = rsquared value (only for numeric types)
            self.execute("CREATE TABLE IF NOT EXISTS tbl_standardization (setupID INTEGER NOT NULL, parameterID INTEGER NOT NULL, bin INTEGER NOT NULL, VMindex INTEGER DEFAULT 0, x LIST DEFAULT NULL, reference BLOB, regressor BLOB, rmse REAL, p REAL, rsquared REAL DEFAULT NULL, timestamp DATETIME, PRIMARY KEY (setupID, parameterID, bin, VMindex, x), FOREIGN KEY(setupID, parameterID) REFERENCES tbl_standardization_parameter(setupID, parameterID));")

            # add basic data
            self.add_basics()
        return

    @staticmethod
    def hashID(value):
        hashvalue = fnv.hash(str(value).upper().encode(), algorithm=fnv.fnv_1a, bits=64)
        if hashvalue > (pow(2,63)-1):
            hashvalue = hashvalue - pow(2,63)
        return hashvalue

    def add_basics(self):
        standard_tags = tool_pydicom.get_standard_tags()
        for tag in standard_tags:
            formula = "dcm[0x" + hex(tag[0])[2:].zfill(8) + "].value"
            name = tool_general.string_stripper(tag[1], [])
            VR = tag[2]
            VM = tag[3]
            self.insert_parameter(name, VR, VM, formula)
        return

    # tbl_data
    def import_data(self, path, description=None, overwrite=False):
        counter = 0
        if path.endswith(".marissadb"):
            counter = self.import_from_extern(path, "tbl_data", overwrite)
        else:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for root, _, files in os.walk(path):
                for file in files:
                    try:
                        dcm = pydicom.dcmread(os.path.join(root, file))
                        exist = (True if len(self.select("SELECT SOPinstanceUID FROM tbl_data WHERE SOPinstanceUID = '" + str(dcm[0x0008, 0x0018].value) + "'")) > 0 else False)
                        if overwrite and exist:
                            self.update_data(str(dcm[0x0008, 0x0018].value), str(dcm[0x0020, 0x000D].value), str(int(dcm[0x0020, 0x0011].value)), str(int(dcm[0x0020, 0x0013].value)), str(dcm.PatientName), ("NULL" if description is None else "'" + tool_general.string_stripper(description, []) + "'"), dcm.to_json(), dcm.file_meta.to_json(), str(len(dcm.preamble)))
                            counter = counter + 1
                        elif not exist:
                            self.insert_data(dcm, description, timestamp)
                            counter = counter + 1
                    except:
                        pass
        return counter

    def insert_data(self, dcm, description, timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
        ID = "'" + str(dcm[0x0008, 0x0018].value) + "'"
        StudyinstanceUID = "'" + str(dcm[0x0020, 0x000D].value) + "'"
        seriesnumber = str(int(dcm[0x0020, 0x0011].value))
        instancenumber = str(int(dcm[0x0020, 0x0013].value))
        seriesdescription = "'" + str(dcm[0x0008, 0x103e].value) + "'"
        identifier = "'" + str(dcm.PatientName) + "'"
        try:
            age = "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x1010].value, tool_pydicom.get_VR(0x00101010))) + "'"
        except:
            age = "NULL"

        try:
            gender = "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x0040].value, tool_pydicom.get_VR(0x00100040))) + "'"
        except:
            gender = "NULL"

        try:
            size = "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x1020].value, tool_pydicom.get_VR(0x00101020))) + "'"
        except:
            size = "NULL"

        try:
            weight = "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x1030].value, tool_pydicom.get_VR(0x00101030))) + "'"
        except:
            weight = "NULL"

        try:
            desc = "'" + tool_general.string_stripper(description, []) + "'"
        except:
            desc = "NULL"

        try:
            date = str(dcm[0x0008, 0x0022].value)
            time = str(dcm[0x0008, 0x0032].value)
            acquisitiondatetime = "'" + date[:4] + "-" + date[4:6] + "-" + date[6:] + " " + time[:2] + ":" + time[2:4] + ":" + time[4:6] + "'"
        except:
            acquisitiondatetime="NULL"
        data = [zlib.compress(dcm.to_json().encode("utf-8")), zlib.compress(dcm.file_meta.to_json().encode("utf-8")), zlib.compress(dcm.preamble)]

        self.execute(("INSERT OR IGNORE INTO tbl_data VALUES(" + ID + ", " + StudyinstanceUID + ", " + seriesnumber + ", " + instancenumber + ", " + seriesdescription + ", " + identifier + ", " + age + ", " + gender + ", " + size + ", " + weight + "," + desc + ", " + acquisitiondatetime + ", ?, '" + timestamp + "')", (data,)))

        return True

    def update_data(self, dcm, description, timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
        ID = "'" + str(dcm[0x0008, 0x0018].value) + "'"
        StudyinstanceUID = "'" + str(dcm[0x0020, 0x000D].value) + "'"
        seriesnumber = str(int(dcm[0x0020, 0x0011].value))
        instancenumber = str(int(dcm[0x0020, 0x0013].value))
        seriesdescription = "'" + str(dcm[0x0008, 0x103e].value) + "'"
        identifier = "'" + str(dcm.PatientName) + "'"
        age = ("NULL" if not (0x0010, 0x1010) in dcm else "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x1010].value, tool_pydicom.get_VR(0x00101010))) + "'")
        gender = ("NULL" if not (0x0010, 0x0040) in dcm else "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x0040].value, tool_pydicom.get_VR(0x00100040))) + "'")
        size = ("NULL" if not (0x0010, 0x1020) in dcm else "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x1020].value, tool_pydicom.get_VR(0x00101020))) + "'")
        weight = ("NULL" if not (0x0010, 0x1030) in dcm else "'" + str(tool_pydicom.get_value(dcm[0x0010, 0x1030].value, tool_pydicom.get_VR(0x00101030))) + "'")
        desc = ("NULL" if description is None else "'" + tool_general.string_stripper(description, []) + "'")
        try:
            date = str(dcm[0x0008, 0x0022].value)
            time = str(dcm[0x0008, 0x0032].value)
            acquisitiondatetime = date[:4] + "-" + date[4:6] + "-" + date[6:] + " " + time[:2] + ":" + time[2:4] + ":" + time[4:6]
        except:
            acquisitiondatetime="NULL"
        data = [zlib.compress(dcm.to_json().encode("utf-8")), zlib.compress(dcm.file_meta.to_json().encode("utf-8")), zlib.compress(dcm.preamble)]
        #data = [dcm.to_json().encode("utf-8"), dcm.file_meta.to_json().encode("utf-8"), dcm.preamble]

        self.execute(("UPDATE tbl_data SET StudyinstanceUID = " + StudyinstanceUID + ", seriesnumber = " + seriesnumber + ", instancenumber = " + instancenumber + ", seriesdescription = " + seriesdescription + ", identifier = " + identifier + ", age = " + age + ", gender = " + gender + ", size = " + size + ", weight = " + weight + ", description = " + desc + ", acquisitiondatetime = " + acquisitiondatetime + ", data = ?, timestamp = '" + str(timestamp) + "' WHERE SOPInstanceUID = " + ID, (data,)))
        return

    def delete_data(self, IDs):
        if type(IDs) == str:
            deleteIDs = [IDs]
        elif isinstance(IDs, np.ndarray):
            deleteIDs = IDs.flatten().tolist()
        else:
            deleteIDs = IDs

        self.execute("DELETE FROM tbl_match_setup_data_segmentation WHERE SOPinstanceUID IN ('" + "', '".join(deleteIDs) + "')")
        self.execute("DELETE FROM tbl_segmentation WHERE SOPinstanceUID IN ('" + "', '".join(deleteIDs) + "')")
        self.execute("DELETE FROM tbl_data WHERE SOPinstanceUID IN ('" + "', '".join(deleteIDs) + "')")
        return True

    def delete_data_all(self):
        self.execute("DELETE FROM tbl_match_setup_data_segmentation")
        self.execute("DELETE FROM tbl_segmentation")
        self.execute("DELETE FROM tbl_data")

        self.execute("DELETE FROM tbl_standardization_match_data_setup_parameter")
        self.execute("DELETE FROM tbl_standardization_data")
        return True

    def get_data(self, where=None):
        if where is None:
            selection = self.select("SELECT data FROM tbl_data;")
        elif type(where) == list:
            selection = self.select("SELECT data FROM tbl_data WHERE SOPinstanceUID in ('" + "', '".join(where) + "')")
        else:
            try:
                selection = self.select("SELECT data FROM tbl_data WHERE " + where + ";")
            except:
                selection = self.select("SELECT data FROM tbl_data WHERE SOPinstanceUID = '" + where + "';")

        dcms = []
        for i in range(len(selection)):
            dcm = pydicom.dataset.Dataset.from_json(zlib.decompressobj().decompress(selection[i][0][0]).decode("utf-8"))
            dcm.ensure_file_meta()
            dcm.file_meta = pydicom.dataset.FileMetaDataset.from_json(zlib.decompressobj().decompress(selection[i][0][1]).decode("utf-8"))
            dcm.preamble = zlib.decompressobj().decompress(selection[i][0][2])
            dcms.append(dcm)
        return dcms

    # tbl_segmentation
    def import_segmentation(self, path, description, creator, data_type="npy", overwrite = False):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        counter = 0

        for root, _, files in os.walk(path):
            for file in files:
                try:
                    if data_type == "npy" and file.endswith(".npy"):
                        ID = file.replace(".npy", "")
                        file = open(os.path.join(root, file), "rb")
                        mask = np.load(file, allow_pickle=True)
                        file.close()

                    elif data_type == "pickle" and file.endswith(".pickle"):
                        ID = file.replace(".pickle", "")
                        file = open(os.path.join(root, file), "rb")
                        data = pickle.load(file)
                        file.close()
                        mask = tool_hadler.from_polygon(data["lv_myo"]["cont"], data["lv_myo"]["imageSize"])

                    else:
                        continue

                    find_underscore = ID.find("_")
                    if find_underscore > 0:
                        ID = ID[:find_underscore]

                    points = tool_general.mask2contour(mask)
                    dimension = len(np.shape(mask))

                    worked = self.insert_segmentation(ID, description, creator, points, mask, dimension, timestamp)
                    if worked:
                        counter = counter + 1
                except:
                    print(traceback.format_exc())
        return counter

    def insert_segmentation(self, SOPinstanceUID, description, creator, points, mask, dimension, timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
        name = tool_general.string_stripper(description, [])
        name_creator = tool_general.string_stripper(creator, [])
        ID = str(self.hashID(SOPinstanceUID + "#" + name + "#" + name_creator))

        if len(self.select("SELECT SOPinstanceUID FROM tbl_data WHERE SOPinstanceUID = '" + SOPinstanceUID + "'")) > 0:
            self.execute(("INSERT OR IGNORE INTO tbl_segmentation VALUES(" + ID + ", '" + SOPinstanceUID + "', '" + name + "', '" + name_creator + "', ?, ?, " + str(dimension) + ", '" + timestamp + "')", (points, mask)))
            result = True
        else:
            result = False
        return result

    def update_segmentation(self, ID, SOPinstanceUID, description, creator, points, mask, dimension, timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
        name = tool_general.string_stripper(description, [])
        name_creator = tool_general.string_stripper(creator, [])
        newID = str(self.hashID(SOPinstanceUID + "#" + name + "#" + name_creator))


        if newID != str(ID):
            worked = self.insert_segmentation(SOPinstanceUID, description, creator, points, mask, dimension, timestamp)
            if worked:
                self.execute("UPDATE tbl_match_setup_data_segmentation SET segmentationID = " + newID + " WHERE segmentationID = " + str(ID))
                self.delete_segmentation(ID)
                result = True
            else:
                result = False
        else:
            self.execute("UPDATE tbl_segmentation SET description = '" + name + "', creator = '" + creator + "', points = ?, mask = ?, dimension = " + str(dimension) + ", timestamp = '" + timestamp + "' WHERE segmentationID = " + str(ID))
            result = True

        return result

    def delete_segmentation(self, IDs):
        if type(IDs) == str or type(IDs) == int:
            deleteIDs = [IDs]
        elif isinstance(IDs, np.ndarray):
            deleteIDs = IDs.flatten().tolist()
        else:
            deleteIDs = IDs

        self.execute("DELETE FROM tbl_match_setup_data_segmentation WHERE segmentationID IN (" + ", ".join(np.array(deleteIDs).astype(str).tolist()) + ")")
        self.execute("DELETE FROM tbl_segmentation WHERE segmentationID IN (" + ", ".join(np.array(deleteIDs).astype(str).tolist())  + ")")
        return True

    # tbl_parameter
    def insert_parameter(self, description, VR, VM, formula):
        name = tool_general.string_stripper(description, [])
        ID = str(self.hashID(name))

        if len(self.select("SELECT parameterID FROM tbl_parameter WHERE parameterID = " + ID)) > 0:
            result = False
        else:
            self.execute("INSERT INTO tbl_parameter VALUES(" + ID +", '" + name + "', '" + VR + "', " + str(VM) + ", '" + formula + "')")
            result = ID
        return result

    def update_parameter(self, ID, description, VR, VM, formula):
        name = tool_general.string_stripper(description, [])
        newID = str(self.hashID(name))


        if newID != str(ID):
            worked = self.insert_parameter(description, VR, VM, formula)
            if worked:
                self.execute("UPDATE tbl_match_setup_parameter SET parameterID = " + newID + " WHERE parameterID = " + str(ID))
                self.delete_parameter(ID)
                result = True
            else:
                result = False

        else:
            self.execute("UPDATE tbl_parameter SET description = '" + name + "', VR = '" + VR + "', VM = " + VM + ", formula = '" + formula + "' WHERE parameterID = " + str(ID))
            result = True
        return result

    def delete_parameter(self, ID):
        self.execute("DELETE FROM tbl_match_setup_parameter WHERE setupID IN (SELECT setupID FROM tbl_match_setup_parameter WHERE parameterID = " + str(ID) + ")")
        self.execute("DELETE FROM tbl_parameter WHERE parameterID = " + str(ID))
        return True

    # tbl_setup
    def insert_setup(self, description, bins, clustertype, regressiontype, ytype, mode, parameterIDs=[]):
        name = tool_general.string_stripper(description, [])
        ID = str(self.hashID(name))

        if len(self.select("SELECT setupID FROM tbl_setup WHERE setupID = " + ID)) > 0:
            result = False
        else:
            self.execute("INSERT INTO tbl_setup VALUES (" + ID + ", '" + name + "', " + str(bins) + ", '" + clustertype + "', '" + regressiontype + "', '" + ytype + "', '" + mode + "')")
            for i in range(len(parameterIDs)):
                self.execute("INSERT INTO tbl_match_setup_parameter VALUES(" + ID + ", " + str(parameterIDs[i]) + ", " + str(i+1) + ")")
            result = ID
        return result

    def update_setup(self, ID, description, bins, clustertype, regressiontype, ytype, mode, parameterIDs=[]):
        name = tool_general.string_stripper(description, [])
        newID = str(self.hashID(name))

        if newID != str(ID):
            worked = self.insert_setup(description, bins, clustertype, regressiontype, ytype, mode, parameterIDs)
            if worked:
                #self.execute("UPDATE tbl_match_setup_parameter SET setupID = " + newID + " WHERE setupID = " + str(ID))
                self.execute("UPDATE tbl_match_setup_data_segmentation SET setupID = " + newID + " WHERE setupID = " + str(ID))
                self.delete_setup(ID)

                result = True
            else:
                result = False
        else:
            self.execute("DELETE FROM tbl_match_setup_parameter WHERE setupID = " + str(ID))
            self.execute("UPDATE tbl_setup SET description = '" + name + "', bins = " + bins + ", clustertype = '" + clustertype + "', regressiontype = '" + regressiontype + "', ytype = '" + ytype + "', mode = '" + mode + "' WHERE setupID = " + str(ID))
            for i in range(len(parameterIDs)):
                self.execute("INSERT INTO tbl_match_setup_parameter VALUES(" + str(ID) + ", " + parameterIDs[i] + ", " + str(i+1) + ")")
            result = True
        return result

    def delete_setup(self, ID):
        self.execute("DELETE FROM tbl_match_setup_parameter WHERE setupID = " + str(ID))
        self.execute("DELETE FROM tbl_match_setup_data_segmentation WHERE setupID = " + str(ID))
        self.execute("DELETE FROM tbl_setup WHERE setupID = " + str(ID))
        return True

    # STANDARDIZATION
    def insert_standardization(self, setupID, reference=None):
        # PREPARATION
        # delete prior training
        self.execute("DELETE FROM tbl_standardization WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_match_data_setup_parameter WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_data WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_parameter WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_setup WHERE setupID = " + str(setupID))

        # copy into tbl_standardization_setup, tbl_standardization_parameter and tbl_standardization_data values from database
        # tables tbl_setup, tbl_match_setup_parameter, tbl_match_setup_data_segmentation
        self.insert_standardization_setup(setupID)
        self.insert_standardization_parameter(setupID)
        self.insert_standardization_data(setupID)

        # read setup
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mode = self.select("SELECT mode FROM tbl_standardization_setup WHERE setupID = " + str(setupID))[0][0]
        ytype =self.select("SELECT ytype FROM tbl_standardization_setup WHERE setupID = " + str(setupID))[0][0]
        bins = int(self.select("SELECT bins FROM tbl_standardization_setup WHERE setupID = " + str(setupID))[0][0])

        # read data and parameters
        data_list = self.select("SELECT SOPinstanceUID, segmentationID, segmentedvalues FROM tbl_standardization_data WHERE setupID = " + str(setupID) + " ORDER BY SOPinstanceUID")
        y_raw = [np.array(data_list[i][2]).flatten() for i in range(len(data_list))] # segmented values
        data_list = [[data_list[i][0], data_list[i][1]] for i in range(len(data_list))] # SOPinstanceUIDs and segmentationID for tracking

        x_raw = self.select("SELECT parameters FROM tbl_standardization_data WHERE setupID = " + str(setupID) + " ORDER BY SOPinstanceUID")
        x_raw = np.array([x_raw[i][0] for i in range(len(x_raw))], dtype=object)

        # delete data that has None values in a parameter or less than 10 times the number of bins as segmented pixels
        indeces_sort_out = np.unique(np.concatenate((np.argwhere(np.array([len(y_raw[ii]) for ii in range(len(y_raw))]) <= 10 * bins).flatten(), np.argwhere(x_raw.astype(str) == "None")[:,0].flatten()), axis=0))
        x_raw = np.delete(np.array(x_raw), indeces_sort_out, axis=0)
        y_raw = np.delete(np.array(y_raw), indeces_sort_out, axis=0)
        data_list = np.delete(np.array(data_list), indeces_sort_out, axis=0)

        parameters = self.select("SELECT parameterID, VR, VM FROM tbl_standardization_parameter WHERE setupID = " + str(setupID) + " ORDER BY ordering ASC")
        parametersID = np.concatenate([[parameters[i][0]] * parameters[i][2] for i in range(len(parameters))])
        parametersVR = np.concatenate([[parameters[i][1]] * parameters[i][2] for i in range(len(parameters))])
        parametersVMindex = np.concatenate([np.arange(parameters[i][2]).tolist() for i in range(len(parameters))])
        parameters = np.transpose(np.vstack((parametersID, parametersVR, parametersVMindex)))
        nparam = len(parameters)

        # determine parameter reference
        x_reference = None
        if not reference is None and len(reference) == len(parametersID):
            x_reference = []
            for i in range(len(reference)):
                x_reference.append(tool_pydicom.get_tag(reference[i], 1, parametersVR[i]))
            x_reference = np.array(x_reference, dtype=object)

        # load clustering and regression types
        exec("from marissa.modules.clustering import " + self.select("SELECT clustertype FROM tbl_standardization_setup WHERE setupID = " + str(setupID))[0][0] + " as clustering")
        exec("from marissa.modules.regression import " + self.select("SELECT regressiontype FROM tbl_standardization_setup WHERE setupID = " + str(setupID))[0][0] + " as regression")
        cm = eval("clustering.Model(bins=bins)")
        rm = eval("regression.Model(ytype=ytype)")

        # CLUSTERING
        if bins == 1:
            y_raw_bins = np.array([y_raw], dtype=object).reshape((-1, 1))
        else:
            y_raw_bins = np.empty((len(y_raw), bins)).astype(object)
            for i in range(len(y_raw)):
                values = y_raw[i]
                _, cindeces = cm.run(values, True)
                for j in range(bins):
                    y_raw_bins[i,j] = values[cindeces.astype(int)==j]

        # REGRESSION
        if mode == "ensemble":
            # if no reference given, automatically set one
            if x_reference is None:
                x_raw_str = ["#\t#".join(x_raw[ii, :].flatten().astype(str).tolist()) for ii in range(len(x_raw))]
                unique, counts = np.unique(x_raw_str, return_counts=True)
                x_reference = x_raw[np.argwhere(np.array(x_raw_str) == unique[np.argmax(counts)]).flatten()[0], :]

            # use complete parameter array
            x = np.copy(x_raw)
            xrefindex = np.where((x_raw == x_reference).all(axis=1))[0]
            xsymbolic = [(True if tool_pydicom.get_VR_type(parameters[i][1]) == str else False) for i in range(nparam)]

            # convert categorical variable to number
            for i in range(nparam):
                self.insert_standardization_match_data_setup_parameter(setupID, parameters[i][0], [data_list[di][0] for di in range(len(data_list))], [data_list[di][1] for di in range(len(data_list))], VMindex=int(parameters[i][2]))
                if xsymbolic[i]:
                    xx = list(filter(lambda item: item is not None, x[:,i].tolist()))
                    symbols = np.unique(xx)
                    if not x_reference is None:
                        try:
                            sl = symbols.tolist()
                            sl.remove(x_reference[i])
                            symbols = [symbols[np.argwhere(symbols==x_reference[i]).flatten()[0]]] + sl
                        except:
                            pass
                    symbolvalues = np.arange(0, len(symbols), 1)
                    x[:,i] = np.array([(-1 if ii is None else symbolvalues[symbols.index(ii)]) for ii in x[:,i].flatten()])

                    for j in range(bins):
                        self.insert_standardization_training(setupID, parameters[i][0], j, x_reference[i], "NULL", "NULL", "NULL", "NULL", VMindex=parameters[i][2], x=symbols, timestamp=timestamp)
                else:
                    for j in range(bins):
                        self.insert_standardization_training(setupID, parameters[i][0], j, x_reference[i], "NULL", "NULL", "NULL", "NULL", VMindex=parameters[i][2], x="NULL", timestamp=timestamp)

            for j in range(bins):
                # use all data
                yref = np.mean(np.concatenate(y_raw_bins[:,j][xrefindex]))
                x = np.concatenate([np.repeat(np.reshape(x[ii,:], (1,-1)), len(y_raw_bins[ii,j]), axis=0) for ii in range(len(y_raw_bins))], axis=0)
                y = np.concatenate([y_raw_bins[ii,j] for ii in range(len(y_raw_bins))])

                if ytype == "absolute":
                    dy = y - yref
                else: #percentage
                    if yref < 1e-6:
                        yref = 1e-6
                    dy = 100 * (y - yref) / yref

                regression_result = rm.train(x, dy)
                self.execute(("UPDATE tbl_standardization SET regressor = ?, rmse = " + str(regression_result["rmse"]) + ", p = " + str(regression_result["p"]) + ", rsquared = " + str(regression_result["rsquared"]) + " WHERE setupID = " + str(setupID) + " AND bin = " + str(j+1) + " AND parameterID = " + str(parameters[0][0]) + " AND VMindex = 1", (rm.get(),)))

            #self.get_standardized_values(x_raw[3], y_raw_bins[3,0], setupID, None, 1, VMindex=None)

        else:
            # TRAIN EACH PARAMETER
            for i in range(nparam):

                # CONSIDER OTHER PARAMETERS AND STANDARDIZE (IF CASCADED) WITH RESPECT TO PRIOR PARAMETERS
                x = copy.deepcopy(x_raw[:, i].flatten())
                # if cascaded, data needs to be standardized with respect to the prior training
                if mode == "cascaded":
                    # standardize for previous parameter
                    if i > 0:
                        for v in range(len(y_raw_bins)): # each case
                            for b in range(bins): # each bin
                                y_raw_bins[v, b] = self.get_standardized_values(x_raw[v, i-1], y_raw_bins[v, b], setupID, parameters[i-1][0], b+1, VMindex=int(parameters[i-1][2])+1)
                    if i == nparam-1: # if last parameter is trained, no consideration needed as all others were standardized
                        consider = None
                    else: # consider only parameters that were not trained yet
                        consider = np.delete(x_raw, np.arange(0,i+1,1), axis=1)
                elif mode == "individual": # not cascaded, consider all parameters except the one that is trained for
                    consider = np.delete(x_raw, np.array([i]), axis=1)
                else:
                    raise NotImplementedError("The mode " + mode + " is not implemented :(")

                # EVALUATE DATA TO CONSIDER FOR TRAINING
                # indeces of values used (y) and which of them are reference (y_ref)
                if consider is None: # if none to consider all train data can be used for the parameter training
                    if x_reference is None:
                        y_ref_indeces = np.argwhere(x==np.min(x)).flatten()
                    else:
                        y_ref_indeces = np.argwhere(x_raw[:,i]==x_reference[i]).flatten()
                    y_indices = np.arange(0, len(y_raw_bins), 1)
                else: # use train data with the highest amount respecting all other parameters
                    setting = np.array(['#\t#'.join(row) for row in consider.astype(str)])
                    unique, counter = np.unique(setting, return_counts=True)

                    if type(x_raw[0,i]) == str:
                        # if not a numeric parameter, then choose the setup that has highest variety of parameter values and if more than 1 setting are considerable, then choose the one with the highest amount of cases
                        indeces_settings = [np.argwhere(setting==unique[ii]).flatten() for ii in range(len(unique))]
                        counter_parameters = []

                        for indeces_setting in indeces_settings:
                            setting_parameters = []
                            for ii in indeces_setting:
                                setting_parameters.append(x_raw[ii,i])
                            u = np.unique(setting_parameters)
                            counter_parameters.append(len(u))

                        indeces_best = np.argwhere(counter_parameters == np.max(counter_parameters)).flatten()

                        if len(indeces_best) > 1:
                            index_best = np.argmax(counter[indeces_best])
                            train_setting = unique[indeces_best[index_best]]
                        else:
                            train_setting = unique[indeces_best]

                        x_ref_indeces = np.argwhere(x_raw[:,i]==x_reference[i]).flatten()
                        y_indices = np.argwhere(setting==train_setting).flatten()

                        # if a categorical variable couldnt be standardized
                        if np.max(counter_parameters) < len(np.unique(x_raw[:,i])):
                            ux = np.unique(x_raw[:,i])
                            sx = np.unique(x_raw[y_indices,i])

                            missing = [item for item in ux if item not in sx]

                            if mode == "cascaded":
                                # exclude data that could not be standardized
                                indeces_delete = []
                                for im in missing:
                                    i_d = np.argwhere(x_raw[:,i] == im)
                                    indeces_delete = indeces_delete + i_d.flatten().tolist()
                                x_raw = np.delete(np.array(x_raw), indeces_delete, axis=0)
                                y_raw_bins = np.delete(np.array(y_raw_bins), indeces_delete, axis=0)
                                data_list = np.delete(np.array(data_list), indeces_delete, axis=0)
                                consider = np.delete(np.array(consider), indeces_delete, axis=0)

                                # reperform
                                setting = np.array(['#\t#'.join(row) for row in consider.astype(str)])
                                unique, counter = np.unique(setting, return_counts=True)
                                indeces_settings = [np.argwhere(setting==unique[ii]).flatten() for ii in range(len(unique))]
                                counter_parameters = []
                                for indeces_setting in indeces_settings:
                                    setting_parameters = []
                                    for ii in indeces_setting:
                                        setting_parameters.append(x_raw[ii,i])
                                    u = np.unique(setting_parameters)
                                    counter_parameters.append(len(u))
                                indeces_best = np.argwhere(counter_parameters == np.max(counter_parameters)).flatten()
                                if len(indeces_best) > 1:
                                    index_best = np.argmax(counter[indeces_best])
                                    train_setting = unique[indeces_best[index_best]]
                                else:
                                    train_setting = unique[indeces_best]
                                x_ref_indeces = np.argwhere(x_raw[:,i]==x_reference[i]).flatten()
                                y_indices = np.argwhere(setting==train_setting).flatten()

                            # print in cascaded and individual mode which values couldnt be captured
                            for im in missing:
                                print("WARNING: Training could not be performed for parameter value " + im)

                        if x_reference is None:
                            train_x = x_raw[:,i].flatten()[y_indices]
                            x_ref_indeces = np.argwhere(train_x == np.min(train_x)).flatten()
                        else:
                            x_ref_indeces = np.argwhere(x_raw[:,i]==x_reference[i]).flatten()
                        y_ref_indeces = np.intersect1d(y_indices, x_ref_indeces)
                    else:
                        # for numeric values train in the standard setup
                        if x_reference is None:
                            index = np.argmax(counter).flatten()[0]
                            y_indices = np.argwhere(setting==unique[index]).flatten()
                            y_ref_indeces = np.argwhere(x==np.min(x)).flatten()
                        else:
                            x_ref_indeces = np.argwhere(x_raw[:,i]==x_reference[i]).flatten()
                            for train_setting in unique[np.argsort(counter)][::-1]:
                                if train_setting in setting[x_ref_indeces]:
                                    break
                            y_indices = np.argwhere(setting==train_setting).flatten()
                            y_ref_indeces = np.intersect1d(y_indices, x_ref_indeces)
                yref = [np.mean(np.concatenate(y_raw_bins[y_ref_indeces, ii])) for ii in range(bins)]
                y = y_raw_bins[y_indices, :]
                x = x[y_indices]
                d = np.array(data_list)[y_indices]

                self.insert_standardization_match_data_setup_parameter(setupID, parameters[i][0], [d[di][0] for di in range(len(d))], [d[di][1] for di in range(len(d))], VMindex=int(parameters[i][2]))

                # RUN PARAMETER TRAINING
                for j in range(bins): # run training for each bin
                    xx = np.concatenate([[x[i]] * len(y[i][j]) for i in range(len(y))]).reshape(-1,1)
                    yy = np.concatenate([y[i][j] for i in range(len(y))])

                    if ytype == "absolute":
                        dy = yy - yref[j]
                    else: #percentage
                        if yref[j] < 1e-6:
                            yref[j] = 1e-6
                        dy = 100 * (yy - yref[j]) / yref[j]

                    xtype = tool_pydicom.get_VR_type(parameters[i][1])

                    if xtype == float or xtype == int:
                        regression_result = rm.train(xx, dy)
                        self.insert_standardization_training(setupID, parameters[i][0], j, (np.min(x) if x_reference is None else x_reference[i]), rm.get(), regression_result["rmse"], regression_result["p"], regression_result["rsquared"], VMindex=parameters[i][2], x="NULL", timestamp=timestamp)
                    else:
                        xx = xx.astype(str)
                        x_str, counter = np.unique(xx, return_counts=True)
                        x_ref = x_raw[y_ref_indeces[0], i]

                        for k in range(len(x_str)):
                            index_0 = np.argwhere(xx==x_ref)
                            index_1 = np.argwhere(xx==x_str[k])
                            x_sym = np.array([0] * len(index_0) + [1] * len(index_1)).astype(int).reshape(-1,1)
                            y_sym = np.concatenate((dy[index_0[:,0].flatten()], dy[index_1[:,0].flatten()]))
                            regression_result = rm.train(x_sym, y_sym)
                            self.insert_standardization_training(setupID, parameters[i][0], j, x_ref, rm.get(), regression_result["rmse"], regression_result["p"], regression_result["rsquared"], VMindex=parameters[i][2], x=x_str[k], timestamp=timestamp)
        return

    def insert_standardization_setup(self, setupID):
        self.execute("INSERT INTO tbl_standardization_setup SELECT * FROM tbl_setup WHERE setupID = " + str(setupID))
        return

    def insert_standardization_parameter(self, setupID):
        self.execute("INSERT INTO tbl_standardization_parameter SELECT msp.setupID, msp.parameterID, msp.ordering, p.description, p.VR, p.VM, p.formula FROM tbl_match_setup_parameter AS msp INNER JOIN tbl_parameter as p ON msp.parameterID = p.parameterID WHERE msp.setupID = " + str(setupID))
        return

    def insert_standardization_data(self, setupID):
        new_data = self.select("SELECT SOPinstanceUID, segmentationID FROM tbl_match_setup_data_segmentation WHERE setupID = " + str(setupID))

        parameterIDs = self.select("SELECT parameterID FROM tbl_match_setup_parameter WHERE setupID = " + str(setupID) + " ORDER BY ordering ASC")
        parameterIDs = [parameterIDs[i][0] for i in range(len(parameterIDs))]

        for d in range(len(new_data)):
            SOPinstanceUID = new_data[d][0]
            segmentationID = new_data[d][1]
            parameters = self.get_data_parameters(SOPinstanceUID, parameterIDs)
            segmentedvalues = np.array(self.get_data_segmentation(SOPinstanceUID, segmentationID).tolist())

            self.execute(("INSERT INTO tbl_standardization_data VALUES (" + str(setupID) + ", '" + SOPinstanceUID + "', " + str(segmentationID) + ", ?, ?)", (segmentedvalues, parameters)))
        return True

    def insert_standardization_match_data_setup_parameter(self, setupID, parameterID, SOPinstanceUIDs, segmentationIDs, VMindex="NULL"):
        for i in range(len(SOPinstanceUIDs)):
            self.execute("INSERT INTO tbl_standardization_match_data_setup_parameter VALUES(" + str(setupID) + ", " + str(parameterID) + ", '" + SOPinstanceUIDs[i] + "', " + str(segmentationIDs[i]) + ", " + str((VMindex+1 if type(VMindex)==int else VMindex)) + ")")
        return

    def insert_standardization_training(self, setupID, parameterID, bin, reference, regressor, rmse, p, rsquared, VMindex=0, x="NULL", timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
        self.execute(("INSERT OR REPLACE INTO tbl_standardization VALUES (" + str(setupID) + ", " + (parameterID) + ", " + str(int(bin)+1) + "," + str(int(VMindex)+1) + ", ?, ?, ?, " + str(rmse) + ", " + str(p) + ", " + str(rsquared) + ", '" + timestamp + "')", ((None if (type(x) == str and x == "NULL") else str(x).replace("'", "").replace("\"", "") if type(x) == str else x), (None if type(reference) == str and reference == "NULL" else str(reference).replace("'", "").replace("\"", "")), (None if type(regressor) == str and regressor == "NULL" else regressor))))
        return

    def insert_copy_standardization(self, old_setupID, new_description):
        new_setupID = self.hashID(new_description)

        # setup part
        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_setup')")]
        col.remove("setupID")
        col.remove("description")
        self.execute("INSERT INTO tbl_setup SELECT " + str(new_setupID) + ", '" + new_description + "', " + ", ".join(col) + " FROM tbl_setup WHERE setupID = " + str(old_setupID))

        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_match_setup_data_segmentation')")]
        col.remove("setupID")
        self.execute("INSERT INTO tbl_match_setup_data_segmentation SELECT " + str(new_setupID) + ", " + ", ".join(col) + " FROM tbl_match_setup_data_segmentation WHERE setupID = " + str(old_setupID))

        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_match_setup_parameter')")]
        col.remove("setupID")
        self.execute("INSERT INTO tbl_match_setup_parameter SELECT " + str(new_setupID) + ", " + ", ".join(col) + " FROM tbl_match_setup_parameter WHERE setupID = " + str(old_setupID))

        # standardization part
        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_standardization_setup')")]
        col.remove("setupID")
        col.remove("description")
        self.execute("INSERT INTO tbl_standardization_setup SELECT " + str(new_setupID) + ", '" + new_description + "', " + ", ".join(col) + " FROM tbl_standardization_setup WHERE setupID = " + str(old_setupID))

        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_standardization_parameter')")]
        col.remove("setupID")
        self.execute("INSERT INTO tbl_standardization_parameter SELECT " + str(new_setupID) + ", " + ", ".join(col) + " FROM tbl_standardization_parameter WHERE setupID = " + str(old_setupID))

        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_standardization_data')")]
        col.remove("setupID")
        self.execute("INSERT INTO tbl_standardization_data SELECT " + str(new_setupID) + ", " + ", ".join(col) + " FROM tbl_standardization_data WHERE setupID = " + str(old_setupID))

        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_standardization_match_data_setup_parameter')")]
        col.remove("setupID")
        self.execute("INSERT INTO tbl_standardization_match_data_setup_parameter SELECT " + str(new_setupID) + ", " + ", ".join(col) + " FROM tbl_standardization_match_data_setup_parameter WHERE setupID = " + str(old_setupID))

        col = [col[0] for col in self.select("SELECT name FROM PRAGMA_TABLE_INFO('tbl_standardization')")]
        col.remove("setupID")
        self.execute("INSERT INTO tbl_standardization SELECT " + str(new_setupID) + ", " + ", ".join(col) + " FROM tbl_standardization WHERE setupID = " + str(old_setupID))
        return

    def delete_standardization(self, setupID):
        self.execute("DELETE FROM tbl_standardization WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_match_data_setup_parameter WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_parameter WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_data WHERE setupID = " + str(setupID))
        self.execute("DELETE FROM tbl_standardization_setup WHERE setupID = " + str(setupID))
        return

    def import_standardization(self, path, importsetupID):
        selection = self.select("SELECT 1 FROM tbl_standardization_setup WHERE setupID = " + str(importsetupID))

        if len(selection) == 0:
            self.execute_extern(path, "INSERT INTO main.tbl_standardization_setup SELECT * FROM externdb.tbl_standardization_setup WHERE setupID = " + str(importsetupID))
            self.execute_extern(path, "INSERT INTO main.tbl_standardization_parameter SELECT * FROM externdb.tbl_standardization_parameter WHERE setupID = " + str(importsetupID))
            self.execute_extern(path, "INSERT INTO main.tbl_standardization_data SELECT * FROM externdb.tbl_standardization_data WHERE setupID = " + str(importsetupID))
            self.execute_extern(path, "INSERT INTO main.tbl_standardization_match_data_setup_parameter SELECT * FROM externdb.tbl_standardization_match_data_setup_parameter WHERE setupID = " + str(importsetupID))
            self.execute_extern(path, "INSERT INTO main.tbl_standardization SELECT * FROM externdb.tbl_standardization WHERE setupID = " + str(importsetupID))
        else:
            raise ImportError("The setupID to import already exists.")
        return

    def get_data_parameters(self, SOPinstanceUID, parameterIDs=None):
        if parameterIDs is None:
            selection = self.select("SELECT parameterID FROM tbl_parameter")
            selection = [selection[i][0] for i in range(len(selection))]
        elif type(parameterIDs) == list:
            selection = parameterIDs
        elif type(parameterIDs) == tuple:
            selection = list(parameterIDs)
        elif type(parameterIDs) == np.array:
            selection = parameterIDs.tolist()
        else:
            selection = [parameterIDs]

        selection = np.array(selection).astype(str).flatten().tolist()

        dcm = self.get_data(SOPinstanceUID)[0]

        parameters = self.select("SELECT formula, VR, VM FROM tbl_parameter WHERE parameterID IN (" + ", ".join(selection) + ") ORDER BY CASE parameterID " + " ".join(["WHEN " + selection[i] + " THEN " + str(i+1) for i in range(len(selection))]) + " END")

        result = []
        for i in range(len(parameters)):
            VR = parameters[i][1]
            VM = int(parameters[i][2])
            try:
                value = tool_pydicom.get_tag(eval(parameters[i][0]), VM, VR)
                if VM == 1:
                    if type(value) == str:
                        value = value.replace("\"", "").replace("'", "")

                    result.append(value)
                else:
                    for j in range(VM):
                        if type(value[j]) == str:
                            value[j] = value[j].replace("\"", "").replace("'", "")

                        result.append(value[j])
            except:
                for j in range(VM):
                    result.append(None)
        return result

    def get_data_segmentation(self, SOPinstanceUID, segmentationID):
        dcm = self.get_data(SOPinstanceUID)[0]
        #array = dcm.pixel_array
        array = tool_pydicom.get_dcm_pixel_data(dcm, rescale=True, representation=False)
        mask = self.select("SELECT mask FROM tbl_segmentation WHERE SOPinstanceUID = '" + SOPinstanceUID + "' AND segmentationID = " + str(segmentationID))[0][0]
        mask = mask.astype(int)
        values = array[mask != 0].flatten()
        return values

    def get_standardization(self, dcm, mask, setupID, skip_unknown=True):
        # skip unknown = if symbolic value not in train, then skip and run standardization further, otherwise raise error

        marissadata = creator_marissadata.Inheritance(str(dcm[0x0008, 0x0018].value), self.select("SELECT parameter FROM tbl_info WHERE ID = 'version'")[0][0])

        try:
            marissadata.scaling = [float(dcm[0x00281052].value), float(dcm[0x00281053].value)]
        except:
            marissadata.scaling = []


        # PREPARATION
        array = tool_pydicom.get_dcm_pixel_data(dcm, rescale=True, representation=False) # original array
        values = array[mask.astype(int) != 0].flatten() # segmented values

        selection = self.select("SELECT bins, clustertype, mode FROM tbl_standardization_setup WHERE setupID=" + str(setupID))
        bins = selection[0][0]
        ct = selection[0][1]
        mode = selection[0][2]


        parameters = self.select("SELECT parameterID, VR, VM, formula FROM tbl_standardization_parameter WHERE setupID = " + str(setupID) + " ORDER BY ordering ASC")
        parametersID = np.concatenate([[parameters[i][0]] * parameters[i][2] for i in range(len(parameters))])
        parametersVR = np.concatenate([[parameters[i][1]] * parameters[i][2] for i in range(len(parameters))])
        parametersVM = np.concatenate([[parameters[i][2]] * parameters[i][2] for i in range(len(parameters))])
        parametersVMindex = np.concatenate([np.arange(parameters[i][2]).tolist() for i in range(len(parameters))])
        parametersFormula = np.concatenate([[parameters[i][3]] * parameters[i][2] for i in range(len(parameters))])

        parameters = np.transpose(np.vstack((parametersID, parametersVR, parametersVM, parametersVMindex, parametersFormula)))
        nparam = len(parameters)

        marissadata.array = array
        marissadata.mask = mask.astype(bool)

        marissadata.parameters = [self.select("SELECT description FROM tbl_standardization_parameter WHERE setupID = " + str(setupID) + " AND parameterID = " +str(parameters[i][0]))[0][0] + (" " + str(parameters[i][3]) if int(parameters[i][3]) > 0 else "") for i in range(len(parameters))]
        marissadata.value_progression.append(values)

        # load clustering and regression types
        exec("from marissa.modules.clustering import " + ct + " as clustering")

        # CLUSTERING DATA
        if bins == 1:
            value_array = np.array([values])
        else:
            cm = eval("clustering.Model(bins=bins)")

            value_array = np.empty((bins, 1)).astype(object).flatten()
            _, cindeces = cm.run(values, True)
            for j in range(bins):
                value_array[j] = values[cindeces.astype(int)==j]

            values_out = np.copy(values)

        # REGRESSION
        standardized_values = np.copy(value_array) # standardized values

        if mode=="ensemble":
            # run regression for each parameter
            for i in range(nparam):
                formula = parameters[i][4]
                try:
                    x = eval(formula)
                    if int(parameters[i][2]) > 1:
                        x = x[int(parameters[i][3])]
                    x = tool_pydicom.get_value(x, parameters[i][1])
                except:
                    x = None

                marissadata.parameters_values.append(x)

            x = marissadata.parameters_values

            for b in range(bins):
                try:
                    standardized_values[b] = self.get_standardized_values(x, standardized_values[b], setupID, None, b+1, None)
                except:
                    if not skip_unknown:
                        raise RuntimeError("Parameter not standardizable.")
            if bins == 1:
                marissadata.value_progression.append(np.copy(standardized_values[0]))
            else:
                for b in range(bins):
                    values_out[cindeces==b] = standardized_values[b]
                marissadata.value_progression.append(np.copy(values_out))

            marissadata.parameters = [" & ".join(marissadata.parameters)]

        else:
            # run regression for each parameter
            for i in range(nparam):
                formula = parameters[i][4]
                try:
                    x = eval(formula)
                    if int(parameters[i][2]) > 1:
                        x = x[int(parameters[i][3])]
                    x = tool_pydicom.get_value(x, parameters[i][1])
                except:
                    x = None

                marissadata.parameters_values.append(x)

                for b in range(bins):
                    if mode=="cascaded":
                        try:
                            standardization = self.get_standardized_values(x, standardized_values[b], setupID, parameters[i][0], b+1, int(parameters[i][3])+1)
                        except:
                            standardization = None

                        if (standardization is None or x is None) and not skip_unknown:
                            raise RuntimeError("Parameter not standardizable.")
                        elif (standardization is None or x is None) and skip_unknown:
                            pass
                        else:
                            standardized_values[b] = standardization
                    else:
                        try:
                            standardization = self.get_standardized_values(x, value_array[b], setupID, parameters[i][0], b+1, int(parameters[i][3])+1)
                        except:
                            standardization = None

                        if (standardization is None or x is None) and not skip_unknown:
                            raise RuntimeError("Parameter not standardizable.")
                        elif (standardization is None or x is None) and skip_unknown:
                            pass
                        else:
                            diff = value_array[b] - standardization
                            standardized_values[b] = standardized_values[b] - diff

                if bins == 1:
                    marissadata.value_progression.append(np.copy(standardized_values[0]))
                else:
                    for b in range(bins):
                        values_out[cindeces==b] = standardized_values[b]
                    marissadata.value_progression.append(np.copy(values_out))

        return marissadata

    def get_standardized_values(self, x, y, setupID, parameterID, bin, VMindex=None):
        selection = self.select("SELECT regressiontype, ytype, mode FROM tbl_standardization_setup WHERE setupID=" + str(setupID))
        rt = selection[0][0]
        ytype = selection[0][1]
        mode = selection[0][2]

        exec("from marissa.modules.regression import " + rt + " as regression")

        if mode == "ensemble":
            regressor = self.select("SELECT regressor FROM tbl_standardization WHERE setupID=" + str(setupID) + " AND bin=" + str(bin) + " AND VMindex=1 AND parameterID IN (SELECT parameterID FROM tbl_standardization_parameter WHERE setupID=" + str(setupID) + " AND ordering=1)")[0][0]
            rm = eval("regression.Model(ytype=ytype, load=regressor)")

            selection = self.select("SELECT s.x, s.reference FROM tbl_standardization AS s INNER JOIN tbl_standardization_parameter AS sp ON (s.setupID = sp.SetupID AND s.parameterID = sp.parameterID) WHERE s.bin=" + str(bin) + " AND s.setupID=" + str(setupID) + " ORDER BY sp.ordering, s.VMindex")

            x_in = []
            for i in range(len(selection)):
                if selection[i][0] is None:
                    x_in.append(x[i])
                else:
                    idx = np.argwhere(np.array(selection[i][0])==x[i])
                    if len(idx) == 0:
                        #x_in.append(-1)
                        raise ValueError("parameter unknown")
                        #x_in.append(0) # -1 would make something different, 0 behaves as being in reference value for the parameter -> same behaviour as in cascaded and individual mode
                    else:
                        x_in.append(idx.flatten()[0])

            new_values = rm.apply(np.array(x_in).reshape((1,-1)), y)
        else:
            xtype = self.select("SELECT VR FROM tbl_standardization_parameter WHERE setupID = " + str(setupID) + " AND parameterID = " + str(parameterID))
            xtype = tool_pydicom.get_VR_type(xtype[0][0])
            regressor = None

            if xtype==int or xtype==float:
                regressor = self.select("SELECT regressor FROM tbl_standardization WHERE setupID = " + str(setupID) + " AND parameterID = " + str(parameterID) + " AND bin = " + str(bin) + " AND VMindex = " + str(VMindex) + " AND x IS NULL")[0][0]
                x_in = np.array(x).reshape(1,-1)

            else:
                try:
                    regressor = self.select("SELECT regressor FROM tbl_standardization WHERE setupID = " + str(setupID) + " AND parameterID = " + str(parameterID) + " AND bin = " + str(bin) + " AND VMindex = " + str(VMindex) + " AND x='" + str(x).replace("'", "").replace("\"", "") + "'")[0][0]
                    x_in = np.array([1]).reshape(1,-1)
                except:
                    # not defined, skip
                    #new_values = y
                    raise ValueError("parameter not defined")

            if not regressor is None:
                rm = eval("regression.Model(ytype=ytype, load=regressor)")
                new_values = rm.apply(x_in, y)
        return new_values

    def standardization_exist(self):
        selection = self.select("SELECT description FROM tbl_standardization_setup LIMIT 1")
        if len(selection) == 0:
            result = False
        else:
            result = True
        return result

    def setup_exist(self):
        selection = self.select("SELECT description FROM tbl_setup LIMIT 1")
        if len(selection) == 0:
            result = False
        else:
            result = True
        return result


if __name__ == "__main__":
    pass

