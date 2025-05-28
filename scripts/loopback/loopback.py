from ccatkidlib.rfsoc.rfsoc_daq import R

def loopback_measurement(R):
    # Take VNA sweeps with different drive & sense attenuations to confirm that VNA sweeps and attenuators are working
    # ----------------------------------------------------------------------------------------------------------------
    R.take_vna_sweep()
    R.set_atten(drive=10)
    R.take_vna_sweep(write_comb=False)
    R.set_atten(sense=10)
    R.take_vna_sweep(write_comb=False)

    # Take target sweep with a single tone
    # ------------------------------------
    R.take_target_sweep(write_comb=True)
    R.take_timestream(5)

if __name__ == '__main__':
    RC = R(cfg_path='/home/rfsoc/ccatkidlib/scripts/loopback/system_config.yaml', initialize_boards=True)
    loopback_measurement(RC)
    RC.close()