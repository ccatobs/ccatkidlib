from ccatkidlib.rfsoc.rfsoc_daq import R
import numpy as np

def loopback_measurement(R):
    NCLOs = [768, 1280]
    for NCLO in NCLOs:
        # Take VNA sweeps with different drive & sense attenuations to confirm that VNA sweeps and attenuators are working
        # ----------------------------------------------------------------------------------------------------------------
        R.take_vna_sweep(NCLO=NCLO) # Take VNA sweep with no attenuation

        # Take VNA sweeps with varying drive attens (and enough sense atten to not overdrive ADC)
        drive_attens = np.arange(0, 32, 2)
        for drive in drive_attens:
            R.set_atten(drive=drive, sense=16)
            R.take_vna_sweep(write_comb=False)

        # Take VNA sweeps with varying sense attens (and enough drive atten to not overdrive ADC)
        sense_attens = np.arange(0, 32, 2)
        for sense in sense_attens:
            R.set_atten(drive=16, sense=sense)
            R.take_vna_sweep(write_comb=False)

        # Take target sweeps with varying tone powers (across full band so effectively VNA sweep)
        tone_powers = np.arange(25, 525, 25)
        R.set_atten(drive=15, sense=15) # Set reasonable attenuations to not overdrive ADC
        for power in tone_powers:
            R.take_target_sweep(tone_powers=power)

        # Take target sweep with a single tone
        # ------------------------------------
        R.take_target_sweep(tone_freqs=[[(NCLO - 12)*1e6] for _ in range(len(R.drone_list))], tone_powers=20000, tone_phis=0)
        R.take_timestream(5)

if __name__ == '__main__':
    RC = R(cfg_path='/home/pcs/ccatkidlib/scripts/loopback/system_config.yaml', initialize_boards=True)
    loopback_measurement(RC)
    RC.close()