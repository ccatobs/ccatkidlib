import sys

sys.path.append(r'/home/rfsoc/MKID-characterization')
import numpy as np
import matplotlib.pyplot as plt
import os
from resonator import resonator
import resonator_model_v3 as rm
import logging
from tqdm import tqdm


class drone():
    def __init__(self, drone_id: str, data_dir: str, output_dir: str, targ_sweep_s21_fname: str,
                 targ_sweep_f0s_fname: str, tau: float = None, base_temp: int = None, load_temp: int = None, load_name='',
                 amp_gain: int = None):
        """
        single line of fdm'd MKIDs under specific loading conditions

        :param drone_id: identifier of the drone
        :param data_dir: path to dir where rfsoc targ sweeps and targ sweep f0s are saved
        :param output_dir: path to save plots
        :param targ_sweep_s21_fname: rfsoc .npy file storing targ sweeps
        :param targ_sweep_f0s_fname: rfsoc .npy file saving f0s measured from targ sweeps
        :param base_temp: temp of mxc plate during measurements in mK (if known); ex 150
        :param load_temp: temperature of whatever detectors looking at in K (if known); ex 77
        :param load_name: name of whatever detectors looking at (if known); ex "ir_source"
        :param amp_gain: after-fridge amplification (if known) in dB; ex 30
        """
        
        self.targ_sweep_s21_fname = targ_sweep_s21_fname
        self.tau = tau
        self.drone_id = drone_id
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.targ_sweep_s21 = np.load(data_dir + targ_sweep_s21_fname, allow_pickle = True)
        try:
            self.rfsoc_f0s = np.load(data_dir + targ_sweep_f0s_fname).real
        except:
            self.rfsoc_f0s = np.array([i.real for i in np.load(data_dir + targ_sweep_f0s_fname, allow_pickle = True)])

        self.f, self.s21 = self.targ_sweep_s21[0].real, self.targ_sweep_s21[1]

        # Convert loading conditions to strings for filenames, if they are provided
        if (not base_temp == None):
            self.base_temp = "_" + str(base_temp) + "_mK"
        else:
            self.base_temp = ""

        if (not load_temp == None):
            self.load_temp = "_" + str(load_temp) + "_K"
        else:
            self.load_temp = ""

        if (len(load_name)):
            self.load_name = "_" + load_name
        else:
            self.load_name = load_name

        if (not amp_gain == None):
            self.amp_gain = "_" + str(amp_gain) + "_dB"
        else:
            self.amp_gain = ""

        self.resonators = []

    def init_resonators(self, window=75e6, output=False, savefig=None, manual_f0=False, showfig = False):
        """

        :param window: freq range around rfsoc f0 where target data is
                Will need to be narrower if you have more than 5 detectors on a single line.
        :param output: whether or not to display to the command line
        :param savefig: whether or not to save figures 
        :return:
        """

        self.window = window
        if output: self.send_msg(
            "Initializing " + str(len(self.rfsoc_f0s)) + " resonators on drone " + str(self.drone_id) + "...")

        for i in range(len(self.rfsoc_f0s)):
            f_range = self.f[np.abs(self.f - self.rfsoc_f0s[i]) < self.window]
            s21_range = self.s21[np.abs(self.f - self.rfsoc_f0s[i]) < self.window]
            self.send_msg("RFSoC Resonant Frequencies")
            self.send_msg(f"{self.rfsoc_f0s[i]}")
            self.resonators.append(resonator(f0=self.rfsoc_f0s[i], f=f_range, s21=s21_range, tau = self.tau))
            if output:
                self.send_msg("Resonator " + str(i + 1) + ":")
                self.send_msg("\tf0: %.3f MHz" % float(self.resonators[i].f0 / 1e6))
                self.send_msg("\tQ: %.2f " % self.resonators[i].Q)
        if savefig is not None:
            dir_name = "drone_" + self.drone_id + self.base_temp + self.load_name + self.load_temp + self.amp_gain + "_target_sweeps_raw"
            try:
                os.mkdir(self.output_dir + dir_name)
            except:
                pass

        drone_fig, drone_ax = plt.subplots(len(self.rfsoc_f0s), 1)
        for i in range(len(self.rfsoc_f0s)):
            res_fig, res_ax = plt.subplots()
            res_fig.set_figwidth(8);
            res_fig.set_figheight(17 / 4);
            if output and manual_f0:
                self.send_msg("Manually finding resonant frequency for Resonator " + str(i)
                      + " (%.3f MHz nominal)..." % (self.resonators[i].f0 / 1e6))
            if manual_f0: self.resonators[i].find_manual_f0()
            self.resonators[i].make_raw_s21_plot(res_ax, self.rfsoc_f0s[i])
            self.resonators[i].make_raw_s21_plot(drone_ax[i], self.rfsoc_f0s[i])
            self.send_msg("Resonator " + str(i + 1) + " manual f0: %.3f MHz" % (self.resonators[i].f0 / 1e6))
            res_fig.suptitle(
                "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
                        self.resonators[i].f0 / 1e6) + "MHz")
            res_fig.tight_layout()
            if savefig is not None:
                res_fig.savefig(
                    self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_%.3f" % (
                            self.resonators[i].f0 / 1e6) + "MHz.png")
                self.send_msg(self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_%.3f" % (
                            self.resonators[i].f0 / 1e6) + "MHz.png")
            plt.close(res_fig)

        drone_fig.set_figwidth(8);
        drone_fig.set_figheight(17 / 4 * len(self.rfsoc_f0s));
        drone_fig.suptitle(
            "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain, fontsize=20);
        drone_fig.tight_layout();
        if savefig is not None:
            drone_fig.savefig(
                self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + ".png")
            if showfig:
                plt.show()
                self.send_msg("Drone " + self.drone_id + " raw S21 plots saved to: " + self.output_dir + dir_name + "/")
        plt.close(drone_fig)
        return self.resonators

    def remove_cable(self, output=False, savefig=None, showfig = False):
        if output: self.send_msg("Removing cable delays for all resonators on drone " + self.drone_id + "...")

        if savefig is not None:
            try:
                dir_name = "drone_" + self.drone_id + self.base_temp + self.load_name + self.load_temp + self.amp_gain + "_cable_delay_removal"
                os.mkdir(self.output_dir + dir_name)
            except:
                pass
        drone_fig, drone_ax = plt.subplots(len(self.resonators), 2)
        drone_fig.set_figwidth(10)
        drone_fig.set_figheight(len(self.rfsoc_f0s) * 6)
        for i in range(len(self.resonators)):
            self.resonators[i].remove_cable()
            res_fig, res_ax = plt.subplots(1, 2)
            res_fig.set_figwidth = 10
            res_fig.set_figheight = 6
            self.resonators[i].make_cable_plot(res_ax)
            self.resonators[i].make_cable_plot(drone_ax[i, :])
            res_fig.suptitle(
                "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
                        self.resonators[i].f0 / 1e6) + "MHz_no_cable_delay", fontsize = 20)
            res_fig.tight_layout()
            if output: self.send_msg("\tCable noise removed from %.3f MHz resonator" % (self.resonators[i].f0 / 1e6))
            if savefig is not None:
                res_fig.savefig(
                    self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
                            self.resonators[i].f0 / 1e6) + "MHz.png")
            plt.close(res_fig)
        drone_fig.suptitle(
            "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain, fontsize=20);
        drone_fig.tight_layout(rect = [0, 0, 1, 0.98])
        if savefig is not None:
            drone_fig.savefig(
                self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + ".png")
            if showfig:
                plt.show()
                self.send_msg("Drone " + self.drone_id + " cable removal plots saved to: " + self.output_dir + dir_name + "/")
        plt.close(drone_fig)

    def plot_s21_dB(self, output=False, savefig=None, showfig = None):
        if output: self.send_msg("Plotting s21s (dB) for " + self.drone_id + "...")

        if savefig is not None:
            try:
                dir_name = "drone_" + self.drone_id + self.base_temp + self.load_name + self.load_temp + self.amp_gain + "_s21s_dB"
                os.mkdir(self.output_dir + dir_name)
            except:
                pass
        drone_fig, drone_ax = plt.subplots(len(self.rfsoc_f0s), 1)
        drone_fig.set_figheight(len(self.rfsoc_f0s) * 6)
        #self.remove_cable(savefig = True)
        for i in range(len(self.resonators)):
            res_fig, res_ax = plt.subplots()
            res_fig.set_figheight = 6
            self.resonators[i].make_magnitude_dB_plot(res_ax)
            self.resonators[i].make_magnitude_dB_plot(drone_ax[i])
            res_fig.suptitle(
                "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
                        self.resonators[i].f0 / 1e6) + "MHz", fontsize = 20)
            res_fig.tight_layout()
            if savefig is not None:
                res_fig.savefig(
                    self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
                            self.resonators[i].f0 / 1e6) + "MHz.png")
            plt.close(res_fig)
        drone_fig.suptitle(
            "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain, fontsize=20);
        drone_fig.tight_layout(rect = [0, 0, 1, 0.98])
        if savefig is not None:
            drone_fig.savefig(
                self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + ".png")
            if showfig:
                plt.show()
                self.send_msg("Drone " + self.drone_id + " magnitude dB plots saved to: " + self.output_dir + dir_name + "/")
        plt.close(drone_fig)

    def calibrate_IQ_circle(self, output = False, savefig = None):

        if output: self.send_msg("Calibrating IQ circles on drone " + self.drone_id + "...")
        if savefig is not None:
            try:
                dir_name = "drone_" + self.drone_id + self.base_temp + self.load_name + self.load_temp + self.amp_gain + "_calibrated_IQ_circles"
                os.mkdir(self.output_dir + dir_name)
            except:
                pass
        dir_name = ""
        # Calibrate IQ circle for each resonator
        for i in range(len(self.rfsoc_f0s)):
            #self.resonators[i].rotate_IQ_ellipse(savefig=self.output_dir + dir_name + "/" + "drone_" 
            #                + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
            #                self.resonators[i].f0 / 1e6) + "MHz.png")
            #self.resonators[i].rotate_IQ_circle(savefig=self.output_dir + dir_name + "/" + "drone_" 
            #                + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
            #                self.resonators[i].f0 / 1e6) + "MHz.png")
            self.resonators[i].rotate_IQ_circle(savefig = None)
        if output and savefig is not None: self.send_msg("Drone " + self.drone_id + " IQ ellipse plots saved to " + self.output_dir + dir_name + "/")

    def freq_vs_phase_fit(self, output = False, savefig = None, showfig = None):
        if output: self.send_msg("Finding frequency vs phase fits for drone " + self.drone_id + "...")
        if savefig is not None:
            try:
                dir_name = "drone_" + self.drone_id + self.base_temp + self.load_name + self.load_temp + self.amp_gain + "_freq_vs_phase"
                os.mkdir(self.output_dir + dir_name)
            except:
                pass

        drone_fig, drone_ax = plt.subplots(len(self.rfsoc_f0s), 1)
        drone_fig.set_figheight(len(self.rfsoc_f0s) * 6)
        for i in range(len(self.resonators)):
            res_fig, res_ax = plt.subplots()
            res_fig.set_figheight = 6
            self.resonators[i].freq_vs_phase_fit()
            self.resonators[i].plot_freq_vs_phase(res_ax)
            self.resonators[i].plot_freq_vs_phase(drone_ax[i])
            res_fig.suptitle(
                "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
                        self.resonators[i].f0 / 1e6) + "MHz", fontsize=20)
            res_fig.tight_layout()
            if savefig is not None:
                res_fig.savefig(
                    self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_ %.3f" % (
                            self.resonators[i].f0 / 1e6) + "MHz.png")
            plt.close(res_fig)
        drone_fig.suptitle(
            "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain, fontsize=20);
        drone_fig.tight_layout(rect=[0, 0, 1, 0.98])

        if savefig is not None:
            drone_fig.savefig(
                self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + ".png")
            if showfig:
                plt.show()
                self.send_msg("Drone " + self.drone_id + " magnitude dB plots saved to: " + self.output_dir + dir_name + "/")
        plt.close(drone_fig)

    def get_phase_and_diss_directions(self, output = False, savefig = None):
        if output: self.send_msg("Getting direction of phase and dissipation for drone " + self.drone_id + "...")
        figs = [resonator.get_phase_and_diss_directions(output = output, savefig = savefig) for resonator in self.resonators]
        if savefig is not None:
            try:
                dir_name = "drone_" + self.drone_id + self.base_temp + self.load_name + self.load_temp + self.amp_gain + "_phase_and_diss_directions"
                os.mkdir(self.output_dir + dir_name)
            except:
                pass
            for i in range(len(figs)):
                figs[i].tight_layout()
                figs[i].savefig(self.output_dir + dir_name + "/" + "drone_" + self.drone_id + self.load_name + self.load_temp + self.base_temp + self.amp_gain + "_%.0f_MHz" % (self.resonators[i].f0/1e6)+ ".png")

                if not output: plt.close(figs[i])

    def send_msg(self, msg):
        logger = logging.getLogger(__name__)
        try:
            logger.info(msg)
            tqdm.write(msg)
        except:
            tqdm.write("Error writing message. Ensure that the message is a string.")

    def send_err(self, err):
        logger = logging.getLogger(__name__)
        try:
            logger.error(msg)
            tqdm.write(f"| ERROR | {msg}")
        except:
            tqdm.write("Error writing message. Ensure that the message is a string.")

    def style(self, string, styl = "default"):
        curr_str = ""
        if styl == "header":
            bar_len = 150
            for i in range(bar_len):
                curr_str += "="
            curr_str += "\n"
            for i in range(int((bar_len - len(string))/2)):
                curr_str += " "
            curr_str += string + "\n"

            for i in range(bar_len):
                curr_str += "="
            return curr_str
        if styl == "command":
            return f">>> {string}"
        if styl == "saving":
            return f"| SAVING | {string}"

