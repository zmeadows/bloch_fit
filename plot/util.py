def get_pulse_path():

    base_dir = "/Users/zac/Research/muon_g2_2015/nmr/candella-lab-pulse-data/"

    default_month = "06"
    default_day   = "29"
    default_year  = "15"
    default_run   = "8"
    default_pulse = "1"

    month = raw_input('month[' + default_month + ']: ') or default_month
    day   = raw_input('day['   + default_day   + ']: ') or default_day
    year  = raw_input('year['  + default_year  + ']: ') or default_year
    run   = raw_input('run['   + default_run   + ']: ') or default_run
    pulse = raw_input('pulse[' + default_pulse + ']: ') or default_pulse

    print " "

    return base_dir + month + "_" + day + "_" + year + "/run-" + run + "/" + pulse + ".bin"

DARKGREY  = "#9f9f9f"
GREY      = "#D7D7D7"
LIGHTGREY = "#E8E8E8"
RED       = "#990000"
GREEN     = "#3BA03B"
BLUE      = "#5959FF"
