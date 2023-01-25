import numpy as np


class DataStore(object):

    def __init__(self, Data, CellList, OdorData, FVData, fileHeader):
        self.Data = Data
        self.Cell_List = CellList
        self.Odor_Data = OdorData
        self.FV_Data = FVData
        self.file_header = fileHeader
        self.number_cells = len(CellList)
        self.unique_odors, self.num_unique_odors = self.update_odor_trials(self.Odor_Data)
        # self.trialsWithThisOdor = None
        self.current_odor_trials = None
        self.odor_name = None
        self.cell_index = None
        self.current_cell_name = None

    def makeCopy(self):
        copy = DataStore(self.Data, self.Cell_List,
                         self.Odor_Data, self.FV_Data,
                         self.file_header)

        '''These values will always be the same across all instances of the
        class when copied. This allows us to copy an instance to allow some
        local variable storage while maintaining the important shared data.'''

        return copy

    def update_odor(self, odori: int):
        trialsWithThisOdor = np.array(self.unique_odors[odori] == self.Odor_Data)
        self.current_odor_trials = np.nonzero(trialsWithThisOdor > 0)[0]
        self.odor_name = self.unique_odors[odori]

    def update_cell(self, cellIndex: int):
        self.cell_index = cellIndex
        self.current_cell_name = self.Cell_List[cellIndex]

    @staticmethod
    def update_odor_trials(OdorData):
        unique_odors = np.unique(OdorData)
        num_unique_odors = len(unique_odors)

        return unique_odors, num_unique_odors


class AUROCdataStore(DataStore):

    def __init__(self, Data, CellList, OdorData, FVData, fileHeader,
                 FVonIdx, UnixTimeArray, baselineDuration,
                 responseDuration, doPlot):
        super().__init__(Data, CellList, OdorData, FVData,
                         fileHeader)

        self.FV_on_index = FVonIdx
        self.unix_time_array = UnixTimeArray
        self.baseline_duration = baselineDuration
        self.response_duration = responseDuration
        self.do_plot = doPlot

    def makeCopy(self):
        new_data_handler = AUROCdataStore(self.Data, self.Cell_List,
                                        self.Odor_Data, self.FV_Data, self.file_header, self.FV_on_index,
                                        self.unix_time_array, self.baseline_duration, self.response_duration,
                                        self.do_plot)

        return new_data_handler


class PlottingDataStore(AUROCdataStore):
    def __init__(self, Data, CellList, OdorData, FVData, fileHeader,
                 FVTimeMap, FVonIdx, UnixTimeArray, baselineDuration, responseDuration,
                 AUROCValueTable, lower_bounds, upper_bounds, percentiles):
        super().__init__(Data, CellList, OdorData, FVData, fileHeader, FVonIdx, UnixTimeArray,
                         baselineDuration, responseDuration, False)

        self.FV_time_map = FVTimeMap
        self.baseline_duration = baselineDuration
        self.response_duration = responseDuration
        self.significance_table = AUROCValueTable
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.percentiles = percentiles

    def makeCopy(self):
        copy = PlottingDataStore(self.Data, self.Cell_List, self.Odor_Data,
                                 self.FV_Data, self.file_header, self.FV_time_map, self.FV_on_index,
                                 self.unix_time_array, self.baseline_duration,
                                 self.response_duration, self.significance_table, self.lower_bounds,
                                 self.upper_bounds,
                                 self.percentiles)

        return copy


class AUROCReturn(object):
    def __init__(self):
        self.response_chart = []
        self.auroc_values = []
        self.all_lower_bounds = []
        self.all_upper_bounds = []
        self.percentiles = []
        self.baseline_start_indexes = []
        self.baseline_end_indexes = []
        self.evoked_start_indexes = []
        self.evoked_end_indexes = []
