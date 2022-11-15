from src.data_utils.race import prepare_racepp_dataset
from src.data_utils.arc import prepare_arc_dataset

# TODO possibly: change the correct choice column so that it is an integer (to use ot as index for accessing the option)

# RACE++
race_data_dir = 'data/raw/RACE'
race_c_data_dir = 'data/raw/race-c-master/data'
prepare_racepp_dataset(race_data_dir, race_c_data_dir, 'data/processed')

# ARC
arc_data_dir = 'data/raw/ARC-V1-Feb2018'
prepare_arc_dataset(arc_data_dir, 'data/processed')

# ASSISTments
pass  # TODO

# CloudAcademy
pass  # TODO
