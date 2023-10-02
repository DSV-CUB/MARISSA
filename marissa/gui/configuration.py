import os
import mock
import pickle
import datetime

from marissa.toolbox.creators import creator_configuration, creator_sqlitedb

class Setup(creator_configuration.Inheritance):
    # Hex Colors
    colours = mock.Mock()
    colours.colour_hex_black = "#000000"
    colours.colour_hex_white = "#ffffff"

    colours.colour_hex_blue_light = "#9bcaff"
    colours.colour_hex_blue_dark = "#0000ff"

    colours.colour_hex_green_light = "#aaffaa"
    colours.colour_hex_green_dark = "#00aa00"

    colours.colour_hex_grey_light = "#ececec"
    colours.colour_hex_grey_dark = "#555555"
    colours.colour_hex_grey_dark_extra = "#333333"

    colours.colour_hex_orange_light = "#ffd9b3"
    colours.colour_hex_orange_dark = "#ff8100"

    colours.colour_hex_pink_light = "#ffacf7"
    colours.colour_hex_pink_dark = "#c700b5"

    colours.colour_hex_purple_light = "#ddb8fd"
    colours.colour_hex_purple_dark = "#8700fb"

    colours.colour_hex_red_light = "#ffaaaa"
    colours.colour_hex_red_dark = "#ff0000"

    colours.colour_hex_turquoise_light = "#b8fbf6"
    colours.colour_hex_turquoise_dark = "#00ccbc"

    colours.colour_hex_yellow_light = "#fffac8"
    colours.colour_hex_yellow_dark = "#fbe400"

    def __init__(self, **kwargs):
        super().__init__()

        # mode
        self.name = "marissa"

        # paths
        self.path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) #.replace("/", "\\")
        self.path_data = os.path.join(os.path.dirname(self.path), "appdata")
        self.path_projects = os.path.join(self.path_data, "projects")

        # create working directories
        os.makedirs(self.path_data, exist_ok=True)
        os.makedirs(self.path_projects, exist_ok=True)

        # further paths
        self.path_icons = {}
        for root, _, files in os.walk(os.path.join(self.path, "gui", "images")):
            for file in files:
                if file.endswith(".png"):
                    imagename = file.replace(".png", "")
                    self.path_icons[imagename] = os.path.join(root, file)
        self.path_database = os.path.join(self.path_data, "MARISSA.sqlite")
        self.path_project = None
        self.path_export = None

        # databases
        self.database = _ConfigDB(self.path_database)# general config database
        self.project = None # project specific database


        #gui
        #self.window_width, self.window_height, self.window_stepsize = tool_general.get_window_size()

        self.set(**kwargs)
        return

    def save(self, path=None, timestamp=""):
        if path is None:
            save_to = os.path.join(self.path, self.name + ".pickle")
        else:
            save_to = os.path.join(path, self.name + ".pickle")

        if save_to.endswith(".pickle"):
            self.database = None
            self.project = None
            self.path_project = None

            with open(save_to, 'wb') as file:
                pickle.dump(self, file)
                file.close()

            self.database = _ConfigDB(self.path_database)
        return True

    def get_projects(self):
        projects = []
        for root, _, files in os.walk(self.path_projects):
            for file in files:
                if file.endswith(".marissadb") and not os.path.join(root, file) == self.path_project:
                    projects.append(file.replace(".marissadb", ""))
        return projects


class _ConfigDB(creator_sqlitedb.Inheritance):
    def __init__(self, path):
        super().__init__(path)

        project_info = []
        project_info.append(["subject", "Human"])
        project_info.append(["subject", "Phantom"])
        project_info.append(["organ", "Heart"])
        project_info.append(["quantitative", "T1 Map"])
        project_info.append(["quantitative", "T2 Map"])
        project_info.append(["quantitative", "T2star Map"])

        info_info = []
        info_info.append(["author", "Darian Steven Viezzer"])
        info_info.append(["version", "1.0"])
        info_info.append(["contact", "Working group for cardiovascular MRI\nExperimental and Clinical Research Center\nECRC - a joint institution of the Charité and MDC\nCharité Campus Buch\nLindenberger Weg 80\n13125 Berlin\nGermany (Europe)\nEarth, Solar System, Milky Way\ndarian-steven.viezzer@charite.de\nhttps://cmr-berlin.org"])
        info_info.append(["license", "MIT License\n\nCopyright (c) 2023 Darian Steven Viezzer, Charité Universitätsmedizin Berlin\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."])
        info_info.append(["projectpage", "https://github.com/DSV-CUB/MARISSA"])
        info_info.append(["timestamp", "27.09.2023"])

        if not os.path.isfile(self.path):
            self.execute("CREATE TABLE IF NOT EXISTS tbl_info (ID TEXT, parameter TEXT);")
            self.execute("CREATE TABLE IF NOT EXISTS tbl_project (ID TEXT, parameter TEXT);")

            for pi in project_info:
                self.execute("INSERT INTO tbl_project VALUES ('" + pi[0] +"', '" + pi[1] +"');")

            for ii in info_info:
                self.execute("INSERT INTO tbl_info VALUES ('" + ii[0] +"', '" + ii[1] +"');")
        else:
            for pi in project_info:
                selection = self.select("SELECT 1 FROM tbl_project WHERE ID='" + pi[0] + "' AND parameter='" + pi[1] + "'")
                if len(selection) == 0:
                    self.execute("INSERT INTO tbl_project VALUES ('" + pi[0] +"', '" + pi[1] +"');")

            for ii in info_info:
                self.execute("UPDATE tbl_info SET parameter='" + ii[1] +"' WHERE ID ='" + ii[0] +"';")
        return