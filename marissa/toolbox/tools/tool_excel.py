import xlsxwriter
import numpy as np
import os
import pandas

def num2col(number):
    return xlsxwriter.utility.xl_col_to_name(number)

def col2num(column):
    num = 0
    for c in column:
        num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num - 1

class Setup:
    def __init__(self, path_file=None):
        self.workbook = None
        self.worksheets = {}
        self.row = {}
        self.create(path_file)
        return

    def create(self, path_file=None):
        self.workbook = xlsxwriter.Workbook(path_file)
        return

    def add_worksheet(self, name):
        self.worksheets[name] = self.workbook.add_worksheet(name)
        self.row[name] = 0
        return

    def write_line(self, name, data, row=None, formating=None):
        if not row is None:
            if row > self.row[name]:
                max_row = row
            else:
                max_row = self.row[name]
            self.row[name] = row
        else:
            max_row = 0

        for i in range(len(data)):
            try:
                if formating is None:
                    self.worksheets[name].write_number(self.row[name], i, float(data[i]))
                else:
                    self.worksheets[name].write_number(self.row[name], i, float(data[i]), formating)
            except:
                if formating is None:
                    self.worksheets[name].write(self.row[name], i, str(data[i]))
                else:
                    self.worksheets[name].write(self.row[name], i, str(data[i]), formating)

        if max_row > self.row[name]:
            self.row[name] = max_row
        else:
            self.row[name] = self.row[name] + 1
        return

    def write_array(self, name, array, row=None, formating=None):
        if len(np.shape(array)) == 1:
            write_array = np.expand_dims(array, axis=1)
        elif len(np.shape(array)) == 2:
            write_array = np.copy(array)
        else:
            raise ValueError("Array can only have 1 or 2 dimensions")

        for i in range(np.shape(write_array)[0]):
            if row is None:
                self.write_line(name, write_array[i,:].flatten().tolist(), row, formating)
            else:
                self.write_line(name, write_array[i,:].flatten().tolist(), int(row+i), formating)
        return

    def write_list(self, name, list, row=None, formating=None):
        self.write_array(name, np.array(list), row, formating)
        return


    def write_header(self, name, header, row=None):
        header_format = self.workbook.add_format()
        header_format.set_align("center")
        header_format.set_align("vcenter")
        header_format.set_text_wrap()
        header_format.set_bold()
        self.write_line(name, header, row, header_format)
        return

    def set_row(self, name, row, height):
        self.worksheets[name].set_row(row, height)
        return

    def set_freeze_panes(self, name, row, col):
        self.worksheets[name].freeze_panes(row, col)
        return

    def save(self):
        self.workbook.close()
        return

class Reader:
    def __init__(self, path=None, **kwargs):
        self.data = None
        self.path = None

        if not path is None:
            self.load(path, **kwargs)
        return

    def load(self, path, **kwargs):
        if os.path.isfile(path):
            self.path = path
            if path.endswith(".xlsx") or path.endswith(".xls"):
                sheet_name = kwargs.get("sheet_name", 0)
                force_numeric = kwargs.get("force_numeric", False)

                self.data = {}

                #with open(path, "r") as read_file:
                read_data = pandas.read_excel(path, sheet_name=sheet_name)
                #read_file.close()

                for column in read_data.columns:
                    if force_numeric:
                        column_data = read_data[column].tolist()
                        for i in range(len(column_data)):
                            try:
                                column_data[i] = float(str(column_data[i]).replace(",", "."))
                            except:
                                pass
                        self.data[column] = column_data

                    else:
                        self.data[column] = read_data[column].tolist()
            else:
                self.path=None
                self.data= None
        else:
            self.path = None
            self.data =  None
        return

    def get_array(self, columns=None, rows=None):
        result = None
        if columns is None:
            columns = list(range(0, len(list(self.data.keys()))))

        if rows is None:
            list(range(0, len(self.data[list(self.data.keys())[0]])))

        for column in columns:
            column_data = self.data[list(self.data.keys())[column]]

            if result is None:
                result = np.expand_dims(np.squeeze(np.array(column_data)[rows]), axis=1)
            else:
                result = np.concatenate((result, np.expand_dims(np.squeeze(np.array(column_data)[rows]), axis=1)), axis=1)

        return result