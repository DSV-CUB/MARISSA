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

        if not os.path.isfile(self.path):
            self.execute("CREATE TABLE IF NOT EXISTS tbl_info (ID TEXT, parameter TEXT);")
            self.execute("CREATE TABLE IF NOT EXISTS tbl_project (ID TEXT, parameter TEXT);")

            self.execute("INSERT INTO tbl_project VALUES ('subject', 'Human'), ('subject', 'Phantom'), ('organ', 'Heart'), ('organ', 'Liver'), ('quantitative', 'T1 Map'), ('quantitative', 'T2 Map'), ('quantitative', 'T2star Map');")
            self.execute("INSERT INTO tbl_info VALUES ('author', 'Darian Steven Viezzer'), ('version', '0.1'), ('contact', 'Working group for cardiovascular MRI\nExperimental and Clinical Research Center\nECRC - a joint institution of the Charité and MDC\nCharité Campus Buch\nLindenberger Weg 80\n13125 Berlin\nGermany (Europe)\nEarth, Solar System, Milky Way\ndarian-steven.viezzer@charite.de\nhttps://cmr-berlin.org'), ('timestamp', '" + datetime.datetime.now().strftime("%d.%m.%Y") + "');")
        return