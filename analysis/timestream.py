import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

class timestream():
    def __init__(self, stream_s21s, drone):
        print("test")
        self.drone = drone
        self.stream_s21s = np.array([[complex(stream_s21s[j, 16 + 2 * i],
                                             stream_s21s[j, 16 + 2 * i + 1]) for j in
                                     range(len(stream_s21s[:, 16 + 2 * i]))][int(len(stream_s21s[:, 16 + 2 * i])*.25):] for i in
                                     range(len(self.drone.rfsoc_f0s))])

        self.phase_noise = []
        self.diss_noise = []
        self.noise_in_phase = []
        self.phase_psd = []
        self.diss_psd = []
        self.f = []
        self.tau = self.drone.tau
        self.f_noise = []
        self.df_noise = []
        self.tau = 0.002048
        self.time = self.tau * np.arange(len(self.stream_s21s[0]))
        self.fac = 1
    def remove_cable(self):
        for i in range(len(self.stream_s21s)):
            self.stream_s21s[i] = np.divide(self.stream_s21s[i], self.drone.resonators[i].cable[
                self.drone.resonators[i].f0_index]*np.ones(len(self.stream_s21s[i])))
    def realign_time_stream(self, output = False, savefig = None, showfig = False):
        for i in range(len(self.stream_s21s)):
            #print(self.drone.resonators[i].raw_s21.real)
            if savefig is not None:
                try:
                    dir_name = "drone_" + self.drone.drone_id + self.drone.base_temp + self.drone.load_name + self.drone.load_temp + self.drone.amp_gain + "_time_stream"
                    os.mkdir(self.drone.output_dir + dir_name)
                except:
                    print("")

            fig, ax = plt.subplots()
            ax.scatter(self.stream_s21s[i].real, self.stream_s21s[i].imag, label = "Time Stream Raw")

            ax.plot(self.drone.resonators[i].raw_s21.real[self.drone.resonators[i].res_mask],
                     self.drone.resonators[i].raw_s21.imag[self.drone.resonators[i].res_mask],label = "Targ Sweep Raw" )
            ax.plot(self.drone.resonators[i].raw_s21.real[self.drone.resonators[i].f0_index],
                    self.drone.resonators[i].raw_s21.imag[self.drone.resonators[i].f0_index], marker="x", markersize=11,
                    label="f0")
            ax.legend(fontsize = 18)
            ax.tick_params(labelsize = 18)
            ax.set_xlabel("I", fontsize = 18)
            ax.set_ylabel("Q", fontsize = 18)
            ax.set_title("%.3f MHz"% (self.drone.resonators[i].f0/1e6))
            if savefig is not None:

                fig.tight_layout()
                fig.savefig(
                    self.drone.output_dir + dir_name + "/" + "drone_" + self.drone.drone_id + self.drone.load_name + self.drone.load_temp + self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                                self.drone.resonators[i].f0 / 1e6) + "_time_stream_raw.png")

            if showfig: plt.show()
            else: plt.close(fig)
            #self.remove_cable()
            
            #ret_Is = (self.stream_s21s[i].real * np.cos(self.drone.resonators[i].align_angle) + self.stream_s21s[i].imag * np.sin(self.drone.resonators[i].align_angle))
            #ret_Qs = (self.stream_s21s[i].real * np.sin(self.drone.resonators[i].align_angle) - self.stream_s21s[i].imag * np.cos(self.drone.resonators[i].align_angle))
            #off_0_I = ret_Is - self.drone.resonators[i].x_center
            #off_0_Q = ret_Qs - self.drone.resonators[i].y_center
            #I = off_0_I
            #Q = off_0_Q
            #I = off_0_I * np.cos(self.drone.resonators[i].res_angle) + off_0_Q * np.sin(self.drone.resonators[i].res_angle)
            #Q = off_0_I * np.sin(self.drone.resonators[i].res_angle) - off_0_Q * np.cos(self.drone.resonators[i].res_angle)
            res = self.drone.resonators[i]
            self.stream_s21s[i] = (self.stream_s21s[i] - (res.xc+res.yc*1j))*np.exp(-1j*res.arg)/res.R
            try:    
                self.stream_s21s[i] = self.stream_s21s[i]*np.exp(-1j*res.res_angle)
            except:
                print('res angle failed')
            #stream_s21_i = stream_s21_i_temp * np.cos(
            #    self.drone.resonators[i].res_angle) + stream_s21_q_temp * np.sin(self.drone.resonators[i].res_angle)
            #stream_s21_q = stream_s21_i_temp * np.sin(
            #    self.drone.resonators[i].align_angle) - stream_s21_q_temp * np.cos(self.drone.resonators[i].res_angle)
            #self.stream_s21s[i] = np.array(
            #    [complex(stream_s21_i[j], stream_s21_q[j]) for j in range(len(stream_s21_i))])
            #self.stream_s21s[i] = np.array([complex(I[j], Q[j]) for j in range(len(I))])

            fig, ax = plt.subplots()
            ax.scatter(self.stream_s21s[i].real, self.stream_s21s[i].imag, label = "Time Stream")
            ax.plot(self.drone.resonators[i].s21.real[self.drone.resonators[i].res_mask],
                     self.drone.resonators[i].s21.imag[self.drone.resonators[i].res_mask],label = "Targ Sweep" )

            ax.plot(self.drone.resonators[i].s21.real[self.drone.resonators[i].f0_index],
                    self.drone.resonators[i].s21.imag[self.drone.resonators[i].f0_index], marker = "x", markersize = 11, label="f0")
            ax.legend(fontsize = 18)
            ax.tick_params(labelsize = 18)
            ax.set_xlabel("I", fontsize = 18)
            ax.set_ylabel("Q", fontsize = 18)
            ax.set_title("%.3f MHz"% (self.drone.resonators[i].f0/1e6))
            if savefig is not None:

                fig.tight_layout()
                fig.savefig(
                    self.drone.output_dir + dir_name + "/" + "drone_" + self.drone.drone_id + self.drone.load_name + self.drone.load_temp + self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                                self.drone.resonators[i].f0 / 1e6) + "_time_stream.png")

            if showfig: plt.show()
            else: plt.close(fig)
    def get_noise_in_phase(self):
        self.noise_in_phase = []
        for i in range(len(self.drone.resonators)):
            self.noise_in_phase.append(np.arctan(self.stream_s21s[i].imag, self.stream_s21s[i].real))

        return self.noise_in_phase
    def get_f_noise(self):
        for i in range(len(self.drone.resonators)):
            f_noise = self.drone.resonators[i].p(self.noise_in_phase[i])
            self.f_noise.append(f_noise)
        return self.time, self.f_noise
    def get_df_noise(self):
        for i in range(len(self.drone.resonators)):
            self.df_noise.append(self.f_noise[i]-self.drone.resonators[i].f0)
        return self.time, self.df_noise
    def get_phase_and_diss_noise(self):
        print("check")
        for i in range(len(self.drone.resonators)):
            delta_I = self.stream_s21s[i].real - self.drone.resonators[i].I_res
            delta_Q = self.stream_s21s[i].imag - self.drone.resonators[i].Q_res

            self.phase_noise.append(delta_I * self.drone.resonators[i].phase_direction[0] + delta_Q *
                                    self.drone.resonators[i].phase_direction[1])
            self.diss_noise.append(delta_I * self.drone.resonators[i].diss_direction[0] +  delta_Q *
                                   self.drone.resonators[i].diss_direction[1])
            f, phase_psd = signal.welch(self.phase_noise[i], fs=512e6 / 2 ** 20, nperseg=256)
            f, diss_psd = signal.welch(self.diss_noise[i], fs=512e6 / 2 ** 20, nperseg=256)
            phase_psd = np.sqrt(phase_psd)
            diss_psd = np.sqrt(diss_psd)
            self.f.append(f)
            self.phase_psd.append(phase_psd)
            self.diss_psd.append(diss_psd)
    def plot_log_phase_and_diss_noise(self, savefig = None, output = False, showfig = False, ax_in = None, label = ""):
        axs = []
        for i in range(len(self.drone.resonators)):
            try:
                if ax_in is None:
                    fig, ax = plt.subplots()
            except:
                ax = ax_in[i]
            ax.semilogx(self.f[i], 20 * np.log10(self.phase_psd[i]), label = label + " Phase Noise")
            ax.semilogx(self.f[i], 20 * np.log10(self.diss_psd[i]), label = label + " Diss Noise")
            ax.legend(fontsize = 18)
            ax.set_ylabel("$dB/\sqrt{Hz}$")
            ax.set_xlabel("Hz")
            ax.set_title("%.3f MHz Log Phase and Dissipation Noise" % (self.drone.resonators[i].f0/1e6))
            if savefig is not None:
                try:
                    dir_name = "drone_" + self.drone.drone_id + self.drone.base_temp + self.drone.load_name + self.drone.load_temp + self.drone.amp_gain + "_noise_psd"
                    os.mkdir(self.drone.output_dir + dir_name)
                except:
                    print("")
                fig.tight_layout()
                fig.savefig(
                    self.drone.output_dir + dir_name + "/" + "drone_" + self.drone.drone_id + self.drone.load_name + self.drone.load_temp + self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                            self.drone.resonators[i].f0 / 1e6) + "_noise_psd_log.png")
            axs.append(ax)
            if showfig:plt.show()
            if ax_in == None: plt.close(fig)
        return axs

    def plot_phase_and_diss_noise(self, savefig=None, output=False, showfig = False, ax_in = None, label = ""):
        axs = []
        for i in range(len(self.drone.resonators)):
            try:
                if ax_in == None:
                    fig, ax = plt.subplots()
            except:
                ax = ax_in[i]
            ax.semilogx(self.f[i], self.phase_psd[i], label = label + " Phase Noise")
            ax.semilogx(self.f[i], self.diss_psd[i], label = label +" Diss Noise")
            ax.set_title("%.0f MHz " % (self.drone.resonators[i].f0 / 1e6))
            ax.set_ylabel("$Raw/\sqrt{Hz}$")
            ax.set_xlabel("Hz")
            ax.legend(fontsize = 18)
            axs.append(ax)
            if savefig is not None:
                try:
                    dir_name = "drone_" + self.drone.drone_id + self.drone.base_temp + self.drone.load_name + self.drone.load_temp + self.drone.amp_gain + "_noise_psd"
                    os.mkdir(self.drone.output_dir + dir_name)
                    fig.tight_layout()
                    fig.savefig(self.drone.output_dir + dir_name + "/" + "drone_" + 
                                self.drone.drone_id + self.drone.load_name + self.drone.load_temp + 
                                self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                            self.drone.resonators[i].f0 / 1e6) + "_noise_psd.png")
                except:
                    print("")

            if showfig:
                plt.show()

            try:
                if ax_in == None: plt.close(fig)
            except:
                print("")
        return axs
    def plot_noise_in_phase(self, savefig=None, output=False, showfig = False, ax_in = [], label = ""):
        axs = []
        for i in range(len(self.drone.resonators)):
            if len(ax_in) == 0:
                fig, ax = plt.subplots()
            else: ax = ax_in[i]
            f, psd_in_phase = signal.welch(self.noise_in_phase[i], fs=512e6 / 2 ** 20, nperseg=256)
            ax.semilogx(f, np.sqrt(psd_in_phase), label = label)
            if label != "":
                ax.legend(fontsize = 18, bbox_to_anchor=(1.15, 1.15))
            ax.set_ylabel("$rad/\sqrt{Hz}$")
            ax.set_xlabel("Hz")
            ax.set_title("%.3f MHz Noise in Phase" % (self.drone.resonators[i].f0 / 1e6))
            axs.append(ax)
            if savefig is not None:
                try:
                    dir_name = "drone_" + self.drone.drone_id + self.drone.base_temp + self.drone.load_name + self.drone.load_temp + self.drone.amp_gain + "_noise_psd"
                    os.mkdir(self.drone.output_dir + dir_name)
                except:
                    print("")
                fig.tight_layout()
                fig.savefig(
                    self.drone.output_dir + dir_name + "/" + "drone_" + self.drone.drone_id + self.drone.load_name + self.drone.load_temp + self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                            self.drone.resonators[i].f0 / 1e6) + "_noise_psd_in_phase.png")
            if showfig:
                plt.show()

            if len(ax_in) == 0: plt.close(fig)
        return axs
    def plot_noise_in_mag(self, savefig=None, output=False, showfig = False, ax_in = [], label = ""):
        axs = []

        for i in range(len(self.drone.resonators)):

            if len(ax_in) == 0:
                fig, ax = plt.subplots()
            else: ax = ax_in[i]
            f, psd_in_mag = signal.welch([np.sqrt(self.stream_s21s[i][j].real**2 + self.stream_s21s[i][j].imag**2) for j in range(len(self.stream_s21s[i]))], fs=512e6 / 2 ** 20, nperseg=256)
            ax.semilogx(f, np.sqrt(psd_in_mag), label = label)
            if label != "":
                ax.legend(fontsize = 18)
            ax.set_ylabel("$raw/\sqrt{Hz}$")
            ax.set_xlabel("Hz")
            ax.set_title("%.3f MHz Noise in Mag" % (self.drone.resonators[i].f0 / 1e6))
            axs.append(ax)
            if savefig is not None:
                try:
                    dir_name = "drone_" + self.drone.drone_id + self.drone.base_temp + self.drone.load_name + self.drone.load_temp + self.drone.amp_gain + "_noise_psd"
                    os.mkdir(self.drone.output_dir + dir_name)
                except:
                    print("")

                fig.tight_layout()
                fig.savefig(
                    self.drone.output_dir + dir_name + "/" + "drone_" + self.drone.drone_id + self.drone.load_name + self.drone.load_temp + self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                            self.drone.resonators[i].f0 / 1e6) + "_noise_psd_in_mag.png")
            if showfig:
                plt.show()

            if len(ax_in) == 0: plt.close(fig)

        return axs
    def plot_freq_noise(self, savefig=None, output=False, showfig = False, ax_in = None, label = ""):
        axs = []
        for i in range(len(self.drone.resonators)):

            if ax_in == None:
                fig, ax = plt.subplots()
            else:
                ax = ax_in[i]
            f, psd_in_freq = signal.welch(self.f_noise[i], fs=512e6 / 2 ** 20, nperseg=256)
            ax.semilogx(self.f[i], np.sqrt(psd_in_freq), label =label)
            if label != "":
                ax.legend(fontsize = 18)
            ax.set_ylabel("Hz/\sqrt{Hz}$")
            ax.set_xlabel("Hz")
            ax.set_title("%.3f MHz" % (self.drone.resonators[i].f0 / 1e6))
            axs.append(ax)
            if savefig is not None:
                try:
                    dir_name = "drone_" + self.drone.drone_id + self.drone.base_temp + self.drone.load_name + self.drone.load_temp + self.drone.amp_gain + "_noise_psd"
                    os.mkdir(self.drone.output_dir + dir_name)
                except:
                    print("")
                fig.tight_layout()
                fig.savefig(
                    self.drone.output_dir + dir_name + "/" + "drone_" + self.drone.drone_id + self.drone.load_name + self.drone.load_temp + self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                            self.drone.resonators[i].f0 / 1e6) + "_noise_psd_in_freq.png")
            if showfig:
                plt.show()

            if ax_in == None: plt.close(fig)
        return axs

    def plot_df_noise(self, savefig=None, output=False, showfig = False, ax_in = None, label = ""):
        axs = []
        for i in range(len(self.drone.resonators)):
            try:
                if ax_in == None:
                    fig, ax = plt.subplots()
            except:
                ax = ax_in[i]
            f, psd_in_freq = signal.welch(self.df_noise[i], fs=512e6 / 2 ** 20, nperseg=256)
            ax.semilogx(self.f[i], np.sqrt(psd_in_freq),label = label)
            ax.set_ylabel("Hz/\sqrt{Hz}$")
            ax.set_xlabel("Hz")
            ax.set_title("%.3f MHz" % (self.drone.resonators[i].f0 / 1e6))
            axs.append(ax)
            if label != "":
                ax.legend(fontsize = 18)
            if savefig is not None:
                try:
                    dir_name = "drone_" + self.drone.drone_id + self.drone.base_temp + self.drone.load_name + self.drone.load_temp + self.drone.amp_gain + "_noise_psd"
                    os.mkdir(self.drone.output_dir + dir_name)
                except:
                    print("")
                try:
                    fig.tight_layout()
                    fig.savefig(
                        self.drone.output_dir + dir_name + "/" + "drone_" + self.drone.drone_id + self.drone.load_name + self.drone.load_temp + self.drone.base_temp + self.drone.amp_gain + "_%.0f_MHz" % (
                                self.drone.resonators[i].f0 / 1e6) + "_noise_psd_in_df.png")
                    if ax_in == None: plt.close(fig)
                except:
                    print("")
            if showfig:
                plt.show()

            
        return axs

    def block_mean(self, ar, fac):
        N = len(ar)
        nn = int(N / fac)
        ar = ar[:nn * fac].reshape(nn, -1)
        return (np.mean(ar, axis=1))

    def set_fac(self, fac):
        self.fac = fac
    def plot_timestreams(self, axs = None, i = None, label = ""):
        
        if axs is None:
            for i in range(len(self.stream_s21s)):
                if axs is None:
                    fig, ax = plt.subplots(3, 1)
                    fig.set_figheight(12)
                    fig.set_figwidth(10)
                    print("hi")

                ax[0].plot(self.block_mean(self.time, self.fac), self.block_mean(self.f_noise[i], self.fac), label= label)
                ax[0].set_xlabel("Time [s]", fontsize=18)
                ax[0].set_ylabel("df [Hz]", fontsize=18)
                ax[0].set_title("%.3f Hz Resonator" % (self.drone.resonators[i].f0 / 1e6), fontsize=20)
                ax[0].tick_params(labelsize=18)
                ax[0].legend()

                ax[1].plot(self.block_mean(self.time, self.fac), self.block_mean((self.df_noise[i] * 1e6 / self.drone.resonators[i].f0), self.fac),
                           label=label)
                ax[1].set_xlabel("Time [s]", fontsize=18)
                ax[1].set_ylabel("df/f0 [ppm]", fontsize=18)
                ax[1].tick_params(labelsize=18)
                ax[1].legend()

                ax[2].plot(self.block_mean(self.time, self.fac), self.block_mean(self.noise_in_phase[i], self.fac),
                           label=label)
                ax[2].set_xlabel("Time [s]", fontsize=18)
                ax[2].set_ylabel("phase [rad]", fontsize=18)
                ax[2].tick_params(labelsize=18)
                ax[2].legend()

                if axs is None:
                    fig.tight_layout()

                    plt.savefig(self.drone.output_dir + "timestream_plots_%.0f_MHz.png" % (self.drone.resonators[i].f0 / 1e6))
        else:
            print("meep")
            ax = axs
            #ax[0].plot(self.block_mean(self.time, self.fac), self.block_mean(self.f_noise[i], self.fac),
            #           label=label, marker=".")
            ax[0].plot(self.time, self.f_noise[i],
                       label=label)
            print(self.f_noise[i])
            ax[0].set_xlabel("Time [s]", fontsize=18)
            ax[0].set_ylabel("df [Hz]", fontsize=18)
            ax[0].set_title("%.3f Hz Resonator" % (self.drone.resonators[i].f0 / 1e6), fontsize=20)
            ax[0].tick_params(labelsize=18)
            ax[0].legend()

            # ax[1].plot(self.block_mean(self.time, self.fac),
            #            self.block_mean((self.df_noise[i] * 1e6 / self.drone.resonators[i].f0), self.fac),
            #            label=label, marker=".")
            ax[1].plot(self.time,
                       (self.df_noise[i] * 1e6 / self.drone.resonators[i].f0),
                       label=label)
            ax[1].set_xlabel("Time [s]", fontsize=18)
            ax[1].set_ylabel("df/f0 [ppm]", fontsize=18)
            ax[1].tick_params(labelsize=18)
            ax[1].legend()

            # ax[2].plot(self.block_mean(self.time, self.fac), self.block_mean(self.noise_in_phase[i], self.fac),
            #            label=label, marker=".")
            ax[2].plot(self.time, self.noise_in_phase[i],
                       label=label)
            ax[2].set_xlabel("Time [s]", fontsize=18)
            ax[2].set_ylabel("phase [rad]", fontsize=18)
            ax[2].tick_params(labelsize=18)
            ax[2].legend()

