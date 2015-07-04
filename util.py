def get_pulse_path():

    base_dir = "/Users/zac/Research/muon_g2_2015/nmr/candella-lab-pulse-data/"

    month = "06" # raw_input('month: ')
    day   = "29" # raw_input('day: ')
    year  = "15" # raw_input('year: ')
    run   = raw_input('run: ')
    pulse = raw_input('pulse: ')

    return base_dir + month + "_" + day + "_" + year + "/run-" + run + "/" + pulse + ".bin"

DARKGREY  = "#9f9f9f"
GREY      = "#D7D7D7"
LIGHTGREY = "#E8E8E8"
RED       = "#990000"
GREEN     = "#3BA03B"
BLUE      = "#5959FF"
