import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import sys
sys.path.append("/home/rfsoc/MKID-characterization/mkid_cal")
from drone import drone
from timestream import timestream
import matplotlib
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib import cm
import math
from scipy.interpolate import CubicSpline
from scipy import interpolate
from matplotlib.gridspec import GridSpec

class setup():
    
    def __init__(self, output_dir:str):
        self.output_dir = output_dir
        dir_name = self.output_dir + "setup/"
        try:
            os.mkdir(dir_name)
            
        except:
            print("Directory " + dir_name + " already exists or given path is not valid.")
        if os.path.exists(dir_name):
            self.output_dir = dir_name
    
    def fit_tau(self, vna_sweep_file):
        f, s21 = np.load(vna_sweep_file)
        phase = np.unwrap(np.arctan2(s21.imag, s21.real))
        tau = np.polyfit(f, phase, 1)[0].real/(2*np.pi)
    
    def get_optical_power_from_temp(self, interp_temps, savefig = None):
        
        power_from_temp = np.load('/home/rfsoc/MKID-characterization/coldload_sweep_params/231004_temp_to_power_spline.pkl', 
            allow_pickle = True)
        
        if savefig is not None:
            plt.scatter(interp_temps, power_from_temp(interp_temps));
            plt.xlabel("Temperature [K]");
            plt.ylabel("Power [pW]");
            plt.tight_layout();
            plt.savefig(self.output_dir + "interpolated_optical_powers.png");
            plt.close();
            
        optical_powers = power_from_temp(interp_temps)
        return optical_powers
    
    def get_temps_from_logs(self, temperature_fnames, savefig = None):
        temp_dfs = []
        temps = []

        for i, fname in enumerate(temperature_fnames):
            temp_dfs.append(pd.read_csv(fname))
            temps.append(temp_dfs[i].iloc[:, 2][100:-100].mean())
        del temp_dfs
        return temps

class sweep():
    def __init__(self, sweep_dir, sweep_id, date, applied_powers, readout_powers, base_temp, tau):
        self.sweep_id = sweep_id
        self.sweep_dir = sweep_dir + self.sweep_id + "/"
        try:
            os.mkdir(self.sweep_dir)
        except:
            print("Directory " + self.sweep_dir + " already exists or given path is not valid.")
            
        self.date = date
        self.data_dir = {"low": "/home/rfsoc/rfsoc_result/Colin/" +date["low"] + "/", 
            "medium": "/home/rfsoc/rfsoc_result/Colin/" +date["medium"] + "/"}
        self.set_up = setup(self.sweep_dir + "fig/")
        self.applied_powers = applied_powers
        self.tau = tau
        self.readout_powers = readout_powers
        self.base_temp = base_temp #100
    def save_object(self, output_dir, object_to_save, fname):
        with open(self.sweep_dir + output_dir + fname, "wb") as f: 
            pickle.dump(object_to_save,f)
            
    def load_object(self, output_dir, fname):
        with open(self.sweep_dir + output_dir + fname, "rb") as f: 
            return pickle.load(f)
        
    def get_temps_and_powers(self):
        temperature_fnames_coldload = []
        temperature_fnames_bath = []


        temperature_fnames_medium = []
        for heater_range in self.applied_powers:
            for heater_power in self.applied_powers[heater_range]:
                temperature_fnames_coldload.append(self.data_dir[heater_range] + "20" + self.date[heater_range] 
                    + "_temp_log_" + heater_range + "_" + str(heater_power) + "%_" + "%.0f_dB_coldload_temp.csv" %(np.max(self.readout_powers)))
                temperature_fnames_bath.append(self.data_dir[heater_range] + "20" + self.date[heater_range] 
                    + "_temp_log_" + heater_range + "_" + str(heater_power) + "%_" + "%.0f_dB_bath_temp.csv" % (np.max(self.readout_powers)))

        self.coldload_temps = self.set_up.get_temps_from_logs(temperature_fnames_coldload)
        self.bath_temps =  self.set_up.get_temps_from_logs(temperature_fnames_bath)
        self.optical_powers = self.set_up.get_optical_power_from_temp(self.coldload_temps, savefig = True)
        return self.coldload_temps, self.optical_powers
    
    def locate_targ_sweeps_and_frequencies(self):
        targ_s21s = {}
        targ_s21s_2_MHz = {}
        targ_s21s_3_MHz = {}
        targ_f0s = {}
        timestream_fnames = {}
        timestream_off_res_fnames = {}
        for heater_range in self.applied_powers:

            for heater_power in self.applied_powers[heater_range]:

                temp_targ_f0s = []
                temp_timestream_fnames = []
                temp_timestream_off_res_fnames = []
                temp_targ_s21s = []
                temp_targ_s21s_2_MHz = []
                temp_targ_s21s_3_MHz = []

                for readout_power in self.readout_powers:
                    temp_targ_f0s.append(self.date[heater_range] + "_s21_f0s_" + heater_range + "_" 
                            + str(heater_power) +"%_" + str(readout_power) + "_dB.npy")
                    temp_timestream_fnames.append(self.date[heater_range] + "_timestream_"+heater_range+"_" 
                            + str(heater_power) + "%_" + str(readout_power) + "dB.npy")
                    temp_timestream_off_res_fnames.append(self.date[heater_range] + "_timestream_"+heater_range+"_" 
                            + str(heater_power) + "%_" + str(readout_power) + "dB_1_MHz.npy")
                    temp_targ_s21s.append(self.date[heater_range] + "_s21_targ_" + heater_range +  "_" 
                            + str(heater_power) +"%_" + str(readout_power) + "_dB.npy")
                    temp_targ_s21s_2_MHz.append(self.date[heater_range] + "_s21_targ_" + heater_range + "_" + str(heater_power) 
                            +"%_" + str(readout_power) + "_dB_2MHz.npy")
                    temp_targ_s21s_3_MHz.append(self.date[heater_range] + "_s21_targ_" 
                            + heater_range + "_" + str(heater_power) +"%_" + str(readout_power) + "_dB_3MHz.npy")
                    
                targ_f0s[heater_range + "_" + str(heater_power)] = temp_targ_f0s
                timestream_fnames[heater_range + "_" + str(heater_power)] = temp_timestream_fnames
                timestream_off_res_fnames[heater_range + "_" + str(heater_power)] = temp_timestream_off_res_fnames
                targ_s21s[heater_range + "_" + str(heater_power)] = temp_targ_s21s
                targ_s21s_2_MHz[heater_range + "_" + str(heater_power)] = temp_targ_s21s_2_MHz
                targ_s21s_3_MHz[heater_range + "_" + str(heater_power)] = temp_targ_s21s_3_MHz

        self.targ_f0s = pd.DataFrame(targ_f0s, index = self.readout_powers )
        self.timestream_fnames = pd.DataFrame(timestream_fnames, index = self.readout_powers )
        self.timestream_off_res_fnames = pd.DataFrame(timestream_off_res_fnames, index = self.readout_powers )
        self.targ_s21s = pd.DataFrame(targ_s21s, index = self.readout_powers )
        self.targ_s21s_2_MHz = pd.DataFrame(targ_s21s_2_MHz, index = self.readout_powers )
        self.targ_s21s_3_MHz = pd.DataFrame(targ_s21s_3_MHz, index = self.readout_powers )
        
        return self.targ_f0s, self.timestream_fnames, self.targ_s21s, self.targ_s21s_2_MHz, self.targ_s21s_3_MHz
    
    def initialize_drones(self):
        drones = {}
        drones_2_MHz = {}
        drones_3_MHz = {}
        fig_directories=["narrow_drones/", "2_MHz_drones/", "3_MHZ_drones/"]
        for name in fig_directories:
            try:
                os.mkdir(self.sweep_dir +"fig/" + name)
            except:
                print("Directory " + self.sweep_dir + " already exists or given path is not valid.")

        for temp_count, coldload_temp in enumerate(self.coldload_temps):
            drone_temp = []
            drone_temp_2_MHz = []
            drone_temp_3_MHz = []
            if temp_count < len(self.applied_powers['low']):
                heater_range = "low"
                key = heater_range + "_" + str(self.applied_powers[heater_range][temp_count])

            if temp_count >= len(self.applied_powers['low']):
                heater_range = "medium"
                key = heater_range + "_" + str(self.applied_powers[heater_range][temp_count-len(self.applied_powers['low'])])

            for readout_power in self.readout_powers:
                drone_temp.append(drone(drone_id = "narrow", tau = self.tau, data_dir = self.data_dir[heater_range], 
                        output_dir = self.sweep_dir +"fig/" + fig_directories[0], 
                        targ_sweep_s21_fname = self.targ_s21s.loc[readout_power, key], 
                        targ_sweep_f0s_fname = self.targ_f0s.loc[readout_power, key], 
                        base_temp = self.base_temp, load_temp = coldload_temp, 
                        load_name = "coldload", amp_gain = readout_power))

                drone_temp_2_MHz.append(drone(drone_id = "2_MHz", tau = self.tau, data_dir = self.data_dir[heater_range], 
                        output_dir = self.sweep_dir +"fig/" + fig_directories[1], 
                        targ_sweep_s21_fname = self.targ_s21s_2_MHz.loc[readout_power, key], 
                        targ_sweep_f0s_fname = self.targ_f0s.loc[readout_power, key], base_temp = self.base_temp, 
                        load_temp = coldload_temp, 
                        load_name = "coldload", amp_gain = readout_power))

                drone_temp_3_MHz.append(drone(drone_id = "3_MHz", tau = self.tau, data_dir = self.data_dir[heater_range], 
                        output_dir = self.sweep_dir +"fig/" + fig_directories[2] , 
                        targ_sweep_s21_fname = self.targ_s21s_3_MHz.loc[readout_power, key], 
                        targ_sweep_f0s_fname = self.targ_f0s.loc[readout_power, key], base_temp = self.base_temp, 
                        load_temp = coldload_temp, 
                        load_name = "coldload", amp_gain = readout_power))

                drones[str(coldload_temp) + " K"] = drone_temp
                drones_2_MHz[str(coldload_temp) + " K"] = drone_temp_2_MHz
                drones_3_MHz[str(coldload_temp) + " K"] = drone_temp_3_MHz
        self.drones = pd.DataFrame(drones, index = self.readout_powers)
        self.drones_2_MHz = pd.DataFrame(drones_2_MHz, index = self.readout_powers)
        self.drones_3_MHz = pd.DataFrame(drones_3_MHz, index = self.readout_powers)
        
        return self.drones, self.drones_2_MHz, self.drones_3_MHz
    
    def process_3_MHz_sweeps(self):
        for temp3 in self.drones_3_MHz:
            for d3 in self.drones_3_MHz[temp3].tolist():
                d3.init_resonators(output = False, savefig = True )
                d3.remove_cable(output = False, savefig = True)
                d3.plot_s21_dB(output = False, savefig = True)
                
    def process_2_MHz_sweeps(self):
        for temp2 in self.drones_2_MHz:
            for d2 in self.drones_2_MHz[temp2].tolist():
                d2.init_resonators(output = False, savefig = True)
                d2.remove_cable(savefig = True)
                d2.plot_s21_dB(output = False, savefig = True)
    
    def initialize_timestreams(self):
        timestreams = {}
        

        for temp_count, coldload_temp in enumerate(self.coldload_temps):
            timestreams_temp = []
            drone_temp = []
            drone_temp_2_MHz = []
            drone_temp_3_MHz = []
            if temp_count < len(self.applied_powers['low']):
                heater_range = "low"
                key = heater_range + "_" + str(self.applied_powers[heater_range][temp_count])

            if temp_count >= len(self.applied_powers['low']):
                heater_range = "medium"
                key = heater_range + "_" + str(self.applied_powers[heater_range][temp_count-len(self.applied_powers['low'])])

            for readout_count, readout_power in enumerate(self.readout_powers):
                timestreams_temp.append(timestream(np.load(self.data_dir[heater_range] 
                    + self.timestream_fnames.loc[readout_power, key]),
                    self.drones.iloc[readout_count, temp_count]))
            timestreams[str(self.coldload_temps[temp_count]) + " K"] = timestreams_temp 
            print(timestreams[str(self.coldload_temps[temp_count]) + " K"])
        self.timestreams = pd.DataFrame(timestreams, index = self.readout_powers)
        return self.timestreams
    
    def initialize_off_res_timestreams(self):
        off_res_timestreams = {}
        

        for temp_count, coldload_temp in enumerate(self.coldload_temps):
            timestreams_temp = []
            drone_temp = []
            drone_temp_2_MHz = []
            drone_temp_3_MHz = []
            if temp_count < len(self.applied_powers['low']):
                heater_range = "low"
                key = heater_range + "_" + str(self.applied_powers[heater_range][temp_count])

            if temp_count >= len(self.applied_powers['low']):
                heater_range = "medium"
                key = heater_range + "_" + str(self.applied_powers[heater_range][temp_count-len(self.applied_powers['low'])])

            for readout_count, readout_power in enumerate(self.readout_powers):
                timestreams_temp.append(timestream(np.load(self.data_dir[heater_range] 
                    + self.timestream_off_res_fnames.loc[readout_power, key]),
                    self.drones.iloc[readout_count, temp_count]))
            off_res_timestreams[str(self.coldload_temps[temp_count]) + " K"] = timestreams_temp 
            print(off_res_timestreams[str(self.coldload_temps[temp_count]) + " K"])
        self.off_res_timestreams = pd.DataFrame(off_res_timestreams, index = self.readout_powers)
        return self.off_res_timestreams

    def initialize_shifted_timestreams(self):
        
        with open(self.data_dir['low'] + 'all_shifts.txt') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                lines[i]=float(lines[i])

        tone_offsets = {}
        opt_power_counter = 0
        counter = 0
        mode_count = 0
        for mode in self.applied_powers:
            
            for i in range(len(self.applied_powers[mode])):
                
                loading_condition = []
                for j in range(len(self.readout_powers))[::(-1)**(opt_power_counter+mode_count)]:
                    tones = []
                    timestreams = []
                    for k in range(4):
                        tones.append(self.date[mode] + "_timestream_"+mode+"_" + str(self.applied_powers[mode][i]) + "%_" + str(self.readout_powers[j]) + "dB_%.4f" % lines[counter] + "_MHz.npy")
                        timestreams.append(timestream(np.load(self.data_dir[mode] + tones[k]),
                    self.drones.iloc[j, opt_power_counter]))
                        counter += 1
                    loading_condition.append(timestreams)
                tone_offsets[str(self.optical_powers[i]) + " pW"] = loading_condition
                opt_power_counter += 1
            mode_count = mode_count + 1
        self.shifted_timestreams = pd.DataFrame(tone_offsets, index = self.readout_powers)
        return self.off_res_timestreams
    
    def x_squared(self, x, B):
        return B*x**2
    
    def phase_vs_optical_power(self):
        output_dir = self.sweep_dir + "fig/phase_vs_optical_power/"
        try:
            os.mkdir(output_dir)
        except:
            print("Directory exists or is not valid.")
            
        for k in range(len(self.readout_powers)):
            for i in range(len(self.drones.iloc[0,0].resonators)):
                plt.figure();
                fig, ax = plt.subplots(1, 1);
                ax.grid();
                colormap = plt.cm.coolwarm;
                colors = colormap(np.linspace(0, 1, len(self.coldload_temps)));
                colormap = plt.cm.coolwarm;
                ax.set_prop_cycle('color', colors);
                ax.set_prop_cycle('color', colors);
                fig.set_figwidth(15)
                fig.set_figheight(10)
                
                for j in range(len(self.coldload_temps)):
                    res = self.drones.iloc[k, j].resonators[i] 
                    phase =  np.unwrap(np.arctan2(res.s21.imag, res.s21.real))
                    if np.max(np.abs(np.ediff1d(phase))) > 1:
                        phase =  np.unwrap(np.arctan(res.s21.imag/res.s21.real))
                    if np.min(phase) < -math.pi and np.min(phase) > -2*math.pi:
                        phase += math.pi
                    
                    if np.min(phase) < -2*math.pi:
                        phase += 2*math.pi
                    
                    #phase = phase - np.mean(phase)
                    ax.plot(np.real(res.f)/1e6, phase, marker = ".", 
                        label = "%0.1f" %(self.coldload_temps[j]) + " K/%0.3f" %(self.optical_powers[j]) + "pW");

             
                #ax.legend(bbox_to_anchor=(1.05, 1), loc = "upper left", fontsize = 22);
                bounds = range(len(self.optical_powers)+1)
                ticks = np.array(bounds) + .5
                norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
                labels = ["%.1f pW" % power for power in self.optical_powers]
                labels.append(0)
                cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
                cbar.set_ticklabels(labels)
            
                width = 1.5*self.drones_2_MHz.iloc[k,j].resonators[i].f0/self.drones_2_MHz.iloc[k,j].resonators[i].Q;
                ax.tick_params(labelsize = 22);
                ax.set_xlabel("Frequency [MHz]", fontsize = 24);
                ax.set_ylabel("Phase [rad]", fontsize = 24);
                fig.suptitle("%.3f MHz " %(self.zero_power_f0s[i][self.readout_powers[k]]) + "%.1f dB" % (self.readout_powers[k]), fontsize = 36)
                fig.tight_layout()
                fig.savefig(output_dir + "%.1fMHz" %
                    (self.zero_power_f0s[i][self.readout_powers[k]])+"_%.1f_dB_phase_vs_optical_power.png" % (self.readout_powers[k]))
                plt.close(fig)
    
    def calculate_responsivity(self, savefig = None):
        output_dir = self.sweep_dir + "fig/targ_sweep_vs_temp/"
        
        responsivity = []
        all_f0s = []
        zero_power_f0s = []
        for i in range(len(self.drones_2_MHz.iloc[0,0].resonators)):
            responsivity.append({})
            all_f0s.append({})
            zero_power_f0s.append({})
        try:
            os.mkdir(output_dir)
        except:
            print("Directory exists or is not valid.")
            
        for k in range(len(self.readout_powers)):
            temp_drones = self.drones_2_MHz.iloc[k].tolist()
            for i in range(len(temp_drones[0].resonators)):
                if savefig is not None:
                    plt.figure();
                    fig, ax = plt.subplots(3, 1);
                    ax[0].grid();
                    ax[1].grid();
                    colormap = plt.cm.nipy_spectral;
                    colors = colormap(np.linspace(0, 1, len(self.coldload_temps)));
                    ax[0].set_prop_cycle('color', colors);
                    fig.set_figheight(22.5);
                    fig.set_figwidth(15);
                f0s = []
                Qs = []
                Qcs = []
                for j in range(len(self.coldload_temps)):
                    if savefig is not None:
                        ax[0].plot(np.real(temp_drones[j].resonators[i].f)/1e6, 
                            20 * np.log10(np.abs(temp_drones[j].resonators[i].s21)), marker = ".", 
                            label = "%0.1f" %(self.coldload_temps[j]) + " K/%0.3f" %(self.optical_powers[j]) + "pW");
                    try:
                        f0s.append(temp_drones[j].resonators[i].f0[0]/1e6)
                    except:
                        f0s.append(temp_drones[j].resonators[i].f0/1e6)
                    #f0s.append(self.drones.iloc[k, j].resonators[i].f_r/1e6)
                    Qs.append(temp_drones[j].resonators[i].Q)
                    Qcs.append(temp_drones[j].resonators[i].params['Q_e_real'].value)
                Qis = [1/(1/Qs[i] - 1/Qcs[i]) for i in range(len(Qs))]
                if savefig is not None:
                    ax[0].legend(bbox_to_anchor=(1.05, 1), loc = "upper left", fontsize = 22);
                    width = 1.5*self.drones_2_MHz.iloc[k,j].resonators[i].f0/self.drones_2_MHz.iloc[k,j].resonators[i].Q;
                    ax[0].set_xlim((temp_drones[1].resonators[i].f0 - width*4)/1e6, 
                        (temp_drones[1].resonators[i].f0 + width*4)/1e6);
                    ax[0].tick_params(labelsize = 22);
                    ax[0].set_xlabel("Frequency [MHz]", fontsize = 24);
                    ax[0].set_ylabel("S21 [dB]", fontsize = 24);
                    ax[1].set_xlabel("Coldload Temperature [K]", fontsize = 24);
                    ax[1].set_ylabel("f0 [MHz]", fontsize = 22, color = "tab:blue");

                    ax[1].scatter(self.coldload_temps, f0s, color = "tab:blue")
                    ax[1].tick_params(labelsize = 22)
                    ax[1].tick_params(axis = "y", labelcolor = "tab:blue")
                    ax[1].legend(fontsize = 18)

                    ax1_r = ax[1].twinx()
                    ax1_r.scatter(self.coldload_temps, Qs, c = "black", label = "Q")
                    ax1_r.scatter(self.coldload_temps, Qcs, c = "dimgrey", label = "Qc")
                    ax1_r.scatter(self.coldload_temps, Qis, c = "lightgrey", label = "Qi")
                    ax1_r.legend(bbox_to_anchor=(1.05, 1), loc = "upper left", fontsize = 22)
                    ax1_r.tick_params(axis = "y", labelsize = 24)
                    ax1_r.set_ylabel("Q", fontsize = 24)
                print(f0s)
                print(self.optical_powers)
                responsivity_fit = np.polyfit(f0s, self.optical_powers, 2)
                zeroes = np.poly1d(responsivity_fit).roots
                diff = 10e9
                self.zero_power_f0s = []
                for zero in zeroes:
                    if zero - f0s[-1] > 0:
           
                        if zero-f0s[-1] < diff:
                            diff = zero-f0s[-1]
                            zero_power_f0 = zero
                zero_power_f0s[i][self.readout_powers[k]] = zero_power_f0
                print("fit parameters")
                #p0 = [responsivity_fit[-1]]
                #popt, pcov = curve_fit(self.x_squared, (f0s-zero_power_f0)/zero_power_f0*1e6, self.optical_powers, p0)
                #responsivity[i][self.readout_powers[k]] = np.poly1d([popt[-1], 0, popt[0]])
                #responsivity[i][self.readout_powers[k]] = interpolate.CubicSpline(np.abs((f0s-zero_power_f0)/zero_power_f0*1e6), self.optical_powers)
                responsivity[i][self.readout_powers[k]] = np.poly1d(np.polyfit(np.abs((f0s-zero_power_f0)/zero_power_f0*1e6), self.optical_powers, 2))
                all_f0s[i][self.readout_powers[k]] = f0s
                

                
                
                if savefig is not None:
                    ax[2].scatter(self.optical_powers, (f0s-zero_power_f0)/zero_power_f0*1e6, color = "tab:blue")
                    ax[2].set_xlabel("Optical Power [pW]", fontsize = 24)
                    ax[2].set_ylabel("df/f0 [ppm]", fontsize = 24, color = "tab:blue")
                    fit_f0s = np.linspace(f0s[0], f0s[-1], 200)
                    ax[2].plot(responsivity[i][self.readout_powers[k]](np.abs((fit_f0s-zero_power_f0)/zero_power_f0*1e6)), (fit_f0s-zero_power_f0)/zero_power_f0*1e6, ls = "--")

                    ax[2].tick_params(labelsize = 22)
                    ax[2].tick_params(axis="y", labelcolor = "tab:blue")
                    ax[2].legend(fontsize = 18, bbox_to_anchor = (1.10, 1), loc = "upper left") 
                    ax[2].grid()
                    ax2_r = ax[2].twinx()
                    ax2_r.scatter(self.optical_powers,  Qs, color = "tab:red")
                    ax2_r.set_ylabel("Q", fontsize = 24,color = "tab:red")
                    ax2_r.tick_params(axis = "y", labelcolor = "tab:red", labelsize = 22)

                    fig.suptitle("%.3f MHz " %(zero_power_f0) + "%.1f dB" % (self.readout_powers[k]), fontsize = 36)
                    fig.tight_layout()
                    fig.savefig(output_dir + "%.3fMHz" %
                        (temp_drones[0].resonators[i].f0/1e6)+"_%.1f_dB.png" % (self.readout_powers[k]))
                    plt.close(fig)
        self.responsivity = responsivity
        self.all_f0s = all_f0s
        self.zero_power_f0s = zero_power_f0s
        return self.responsivity, self.all_f0s
    
    def responsivity_at_various_tone_power(self):
        
        output_dir = self.sweep_dir + "fig/responsivity_at_various_tone_power/"
        try:
            os.mkdir(output_dir)
        except:
            print("Directory exists or is not valid.")
            
        for k in range(len(self.drones.iloc[0,0].resonators)):
            fig, ax = plt.subplots(1, 1);
            ax.grid();
            colormap = plt.cm.coolwarm;
            colors = colormap(np.linspace(0, 1, len(self.readout_powers)));
            ax.set_prop_cycle('color', colors);
            for read_pow in range(len(self.readout_powers)):
                
                ax.scatter(self.optical_powers, (self.all_f0s[k][self.readout_powers[read_pow]]-self.zero_power_f0s[k][self.readout_powers[read_pow]])/self.zero_power_f0s[k][self.readout_powers[read_pow]]*1e6, label = "%.1f dB" %self.readout_powers[read_pow], marker = '.')
            for read_pow in range(len(self.readout_powers)):
                fit_f0s = np.linspace(self.all_f0s[k][self.readout_powers[read_pow]][0], self.all_f0s[k][self.readout_powers[read_pow]][-1], 200)
                ax.plot(self.responsivity[k][self.readout_powers[read_pow]](np.abs((fit_f0s-self.zero_power_f0s[k][self.readout_powers[read_pow]])/self.zero_power_f0s[k][self.readout_powers[read_pow]]*1e6)), (fit_f0s-self.zero_power_f0s[k][self.readout_powers[read_pow]])/self.zero_power_f0s[k][self.readout_powers[read_pow]]*1e6)
            bounds = range(len(self.readout_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
 
            labels = ["%.1f dB" % power for power in self.readout_powers]
            labels.append(0)
            cbar= fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar.set_ticklabels(labels)
            
            ax.set_xlabel("Optical Power [pW]")
            ax.set_ylabel("${\delta f/ f_0}$ [ppm]")
            ax.set_title("%.1f MHz Resonator"%(self.all_f0s[k][self.readout_powers[0]][0]))
            fig.tight_layout()
            
            plt.savefig(output_dir + self.date['low'] + "_%.3f_MHz_responsivity_at_various_tone_powers.png" %(self.all_f0s[k][self.readout_powers[0]][0]), bbox_inches="tight")
            
    def four_panel_plot(self):
        output_dir = self.sweep_dir + "fig/four_panel_plots/"
        matplotlib.rcParams.update({'font.size': 14, "font.family": "serif", "mathtext.fontset": "dejavuserif"})

        try:
            os.mkdir(output_dir)
        except:
            print("Directory exists or is not valid.")
        plt.rcParams['figure.facecolor'] = 'white'

        optimal_tone_powers = [78, 78, 78]
        optimal_tone_index = [6, 6, 6]
        lower_lims = [510.1, 618.65, 704.6]
        upper_lims = [510.7, 619.3, 705.3]
        colormap = plt.cm.coolwarm;
        colors_readout = colormap(np.linspace(0, 1, len(self.readout_powers[2:-1])));
        colors_optical = colormap(np.linspace(0, 1, len(self.optical_powers)));
        
        
        for i in range(3):
            fig = plt.figure()
            gs = GridSpec(2,2, height_ratios=[1.3,1])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax1.set_prop_cycle('color', colors_optical);
            ax2.set_prop_cycle('color', colors_optical);
            ax3.set_prop_cycle('color', colors_readout);
            
            fig.set_figwidth(15)
            fig.set_figheight(8)
            ax1.tick_params(labelsize = 18)
            ax3.tick_params(labelsize = 18)
            ax2.tick_params(labelsize = 18)
            ax4.tick_params(labelsize = 18)
            
            for j in range(len(self.coldload_temps)):
                    
                ax1.plot(np.real(self.drones_2_MHz.iloc[optimal_tone_index[i], j].resonators[i].f)/1e6, 
                    20 * np.log10(np.abs(self.drones_2_MHz.iloc[optimal_tone_index[i], j].resonators[i].s21)), marker = ".", 
                    label = "%0.1f" %(self.coldload_temps[j]) + " K/%0.3f" %(self.optical_powers[j]) + "pW");
            ax1.set_xlim(lower_lims[i], upper_lims[i])
            ax1.set_ylim([np.min(20 * np.log10(np.abs(self.drones_2_MHz.iloc[optimal_tone_index[i], 0].resonators[i].s21)))*1.025, 0.25])
            #ax[0,0].axvlines(self.drones_2_MHz.iloc[optimal_tone_index[i], -1].resonators[i].f0/1e6, ymin = -10, ymax = 0.5, color = "black")
            condition = np.where(np.abs(self.drones_2_MHz.iloc[optimal_tone_index[i], -1].resonators[i].s21) == np.min(np.abs(self.drones_2_MHz.iloc[optimal_tone_index[i], -1].resonators[i].s21)))
            #ax[0,0].axvline(self.drones_2_MHz.iloc[optimal_tone_index[i], -1].resonators[i].f[condition]/1e6, color = "black")
            condition = np.where(np.abs(self.drones_2_MHz.iloc[optimal_tone_index[i], -2].resonators[i].s21) == np.min(np.abs(self.drones_2_MHz.iloc[optimal_tone_index[i], -2].resonators[i].s21)))

            #ax[0,0].axvline(self.drones_2_MHz.iloc[optimal_tone_index[i], -2].resonators[i].f[condition]/1e6, color = "black")

            #ax[0,0].axvlines(self.drones_2_MHz.iloc[optimal_tone_index[i], -2].resonators[i].f0/1e6, ymin = -10, ymax = 0.5, color = "black")
            #width = np.real(self.drones_2_MHz.iloc[optimal_tone_index[i], len(self.optical_powers)-1].resonators[i].f[-1])/1e6 + np.real(self.drones_2_MHz.iloc[optimal_tone_index[i], 0].resonators[i].f[0])/1e6
            #ax[0,0].set_xlim(0.75*width)    
            ax1.set_xlabel("Frequency [MHz]", fontsize = 20)
            ax1.set_ylabel("S21 [dB]", fontsize = 20)
            
            bounds = range(len(self.optical_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f" % power for power in self.optical_powers]
            labels.append(0)
            cbar2 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ax = ax1, ticks = ticks)
            cbar2.set_ticklabels(labels)
            cbar2.ax.tick_params(labelsize=18)
            cbar2.ax.set_title("Optical Power [pW]", fontsize = 14)
            
            for j in range(len(self.coldload_temps)):
                res = self.drones.iloc[optimal_tone_index[i], j].resonators[i] 
                phase =  np.unwrap(np.arctan2(res.s21.imag, res.s21.real))
                if np.max(np.abs(np.ediff1d(phase))) > 1:
                    phase =  np.unwrap(np.arctan(res.s21.imag/res.s21.real))
                if np.min(phase) < -math.pi and np.min(phase) > -2*math.pi:
                    phase += math.pi

                if np.min(phase) < -2*math.pi:
                    phase += 2*math.pi

                #phase = phase - np.mean(phase)
                ax2.plot(np.real(res.f)/1e6, phase, marker = ".", 
                    label = "%0.1f" %(self.coldload_temps[j]) + " K/%0.3f" %(self.optical_powers[j]) + "pW");
                phase = np.array(phase)
                #if j in [len(self.coldload_temps)-1, len(self.coldload_temps)-2]:
                #    condition = np.where(phase == np.min(np.abs(phase)))
                #    ax[0,1].axvline(res.f[condition]/1e6, color = "black")

            ax2.set_xlabel("Frequency [MHz]", fontsize = 20)
            ax2.set_ylabel("Phase [radians]", fontsize = 20)
            bounds = range(len(self.optical_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f" % power for power in self.optical_powers]
            labels.append(0)
            cbar2 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ax = ax2, ticks = ticks)
            cbar2.set_ticklabels(labels)
            cbar2.ax.tick_params(labelsize=18)
            cbar2.ax.set_title("Optical Power [pW]", fontsize = 14)
            
            for read_pow in range(len(self.readout_powers[2:-1])):
                print(self.all_f0s[i][self.readout_powers[read_pow]][-1]*1.15)
                fit_f0s = np.linspace(self.all_f0s[i][self.readout_powers[read_pow]][0], self.all_f0s[i][self.readout_powers[read_pow]][-1]*0.99990, 200)
                ax3.scatter(self.optical_powers, (self.all_f0s[i][self.readout_powers[read_pow]]-self.zero_power_f0s[i][self.readout_powers[read_pow]])/self.zero_power_f0s[i][self.readout_powers[read_pow]]*1e6, label = "%.1f dB" %self.readout_powers[read_pow], marker = '.')
                ax3.plot(self.responsivity[i][self.readout_powers[read_pow]](np.abs((fit_f0s-self.zero_power_f0s[i][self.readout_powers[read_pow]])/self.zero_power_f0s[i][self.readout_powers[read_pow]]*1e6)), (fit_f0s-self.zero_power_f0s[i][self.readout_powers[read_pow]])/self.zero_power_f0s[i][self.readout_powers[read_pow]]*1e6)
                ax3.set_xlim([0,8.5])
            
            bounds = range(len(self.readout_powers[2:-1])+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f" % power for power in self.readout_powers[2:-1]]
            labels.append(0)
            cbar2 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ax = ax3, ticks = ticks)
            cbar2.set_ticklabels(labels)
            cbar2.ax.tick_params(labelsize=18)
            cbar2.ax.set_title("Readout Power [dB]", fontsize = 16)
            ax3.set_xlabel("Optical Power [pW]", fontsize = 20)
            ax3.set_ylabel("Fractional freq. shift [ppm]", fontsize = 20)
            
            p0 = [self.nep_avg[i].iloc[0,optimal_tone_index[i]], 0.65]  
            fit_powers = np.linspace(self.optical_powers[0], self.optical_powers[-1], 100)
            #print(np.sqrt(np.diag(pcov))[1])
           
            popt_photon, pcov = curve_fit(self.nep_photon, self.optical_powers[2:], self.nep_avg[i].iloc[:, optimal_tone_index[i]].tolist()[2:], p0)
            ax4.errorbar(self.optical_powers, (self.nep_avg[i].iloc[:, optimal_tone_index[i]]+self.nep_amp_avg[i].iloc[:, optimal_tone_index[i]]), yerr = self.nep_std[i].iloc[:, optimal_tone_index[i]], marker = '.', color = 'black', ls = 'none')
            p0 = [popt_photon[0], popt_photon[1]]
            popt, pcov = curve_fit(self.nep_r, self.optical_powers[4:], self.nep_avg[i].iloc[:, optimal_tone_index[i]].tolist()[4:], p0, sigma = self.nep_std[i].iloc[:, optimal_tone_index[i]].tolist()[4:])
            cs = CubicSpline(self.optical_powers, self.nep_r(self.optical_powers, popt[0], popt[1]))
            z_amp = np.polyfit(self.optical_powers, self.nep_amp_avg[i].iloc[:, optimal_tone_index[i]].tolist(), 2)
            cs_amp = np.poly1d(z_amp)
            fit_powers = np.linspace(self.optical_powers[0], 8.5, 100)
            fit_neps = cs(fit_powers)
            fit_neps_amp = cs_amp(fit_powers)
            ax4.plot(fit_powers, fit_neps + fit_neps_amp, label="$\eta = $ %.2f $\pm$ %.2f, \n $nep_0$ = %.1fE-17 $\pm$ %.1fE-17" %(popt[1], np.sqrt(np.diag(pcov))[1],  popt[0]*1e17, np.sqrt(np.diag(pcov))[0]*1e17), color = 'black')
            ax4.set_xlabel("Optical Power [pW]", fontsize = 20)
            ax4.set_ylabel("NEP [$W/\sqrt{Hz}$]", fontsize = 20)
            ax4.set_xlim([1.0, 8.2])
            ax4.ticklabel_format(style = "plain")
            ax4.set_xticks([1,2,3,4,6])
            #ax4.errorbar(self.optical_powers, self.nep_avg[i].iloc[:, (optimal_tone_index[i]-1)]+self.nep_amp_avg[i].iloc[:, (optimal_tone_index[i]-1)], yerr = self.nep_std[i].iloc[:, (optimal_tone_index[i]-1)], marker = '.', color = 'gray', ls = 'none')
            ax4.set_yscale("log")
            ax4.set_xscale("log")
            ax4.tick_params(axis = "both", labelsize=14)

            #ax4.legend()
            fig.tight_layout()
            fig.savefig(output_dir + "%.1fMHz" %
                (self.zero_power_f0s[i][optimal_tone_powers[i]])+"_four_panel_plot.png", dpi = 300)
            plt.close(fig)
    
    def targ_sweep_vs_readout_power(self):
        output_dir = self.sweep_dir + "fig/targ_sweep_vs_readout_power/"

        try:
            os.mkdir(output_dir)
        except:
            print("Directory exists or is not valid.")
        for j in range(len(self.coldload_temps)):
    
            temp_drones = self.drones_2_MHz.iloc[:,j].tolist()
            
            for i in range(len(temp_drones[0].resonators)):   
                plt.figure();
                fig, ax = plt.subplots(3, 1);
                ax[0].grid();
                ax[1].grid();
                colormap = plt.cm.nipy_spectral;
                colors = colormap(np.linspace(0, 1, len(self.readout_powers)));
                ax[0].set_prop_cycle('color', colors);
                fig.set_figheight(22.5);
                fig.set_figwidth(15);
                f0s = []
                Qs = []
                Qcs = []
                for k in range(len(self.readout_powers)):
                
   
                    ax[0].plot(np.real(temp_drones[j].resonators[i].f)/1e6, 
                        20 * np.log10(np.abs(temp_drones[j].resonators[i].s21)), marker = ".", 
                        label = "%0.1f" %(self.coldload_temps[j]) + " K/%0.3f" %(self.optical_powers[j]) + "pW");
                    f0s.append(temp_drones[j].resonators[i].f0/1e6)
                    #f0s.append(self.drones.iloc[k, j].resonators[i].f_r/1e6)
                    Qs.append(temp_drones[j].resonators[i].Q)
                    Qcs.append(temp_drones[j].resonators[i].params['Q_e_real'].value)
                print(len(f0s))
                print(len(self.readout_powers))
                Qis = [1/(1/Qs[i] - 1/Qcs[i]) for i in range(len(Qs))]
                ax[0].legend(bbox_to_anchor=(1.05, 1), loc = "upper left", fontsize = 22);
                width = 1.5*self.drones_2_MHz.iloc[k,j].resonators[i].f0/self.drones_2_MHz.iloc[k,j].resonators[i].Q;
                ax[0].set_xlim((temp_drones[4].resonators[i].f0 - width*4)/1e6, 
                    (temp_drones[4].resonators[i].f0 + width*4)/1e6);
                ax[0].tick_params(labelsize = 22);
                ax[0].set_xlabel("Frequency [MHz]", fontsize = 24);
                ax[0].set_ylabel("S21 [dB]", fontsize = 24);
                ax[1].set_xlabel("Coldload Temperature [K]", fontsize = 24);
                ax[1].set_ylabel("f0 [MHz]", fontsize = 22, color = "tab:blue");

                ax[1].scatter(self.readout_powers, f0s, color = "tab:blue")
                ax[1].tick_params(labelsize = 22)
                ax[1].tick_params(axis = "y", labelcolor = "tab:blue")
                ax[1].legend(fontsize = 18)

                ax1_r = ax[1].twinx()
                ax1_r.scatter(self.readout_powers, Qs, c = "black", label = "Q")
                ax1_r.scatter(self.readout_powers, Qcs, c = "dimgrey", label = "Qc")
                ax1_r.scatter(self.readout_powers, Qis, c = "lightgrey", label = "Qi")
                ax1_r.legend(bbox_to_anchor=(1.05, 1), loc = "upper left", fontsize = 22)
                ax1_r.tick_params(axis = "y", labelsize = 24)
                ax1_r.set_ylabel("Q", fontsize = 24)
                ax[2].scatter(self.readout_powers, (f0s-self.zero_power_f0s[i][self.readout_powers[k]])/self.zero_power_f0s[i][self.readout_powers[k]]*1e6, color = "tab:blue")
                ax[2].set_xlabel("Optical Power [pW]", fontsize = 24)
                ax[2].set_ylabel("df/f0 [ppm]", fontsize = 24, color = "tab:blue")
                fit_f0s = np.linspace(f0s[0], f0s[-1], 200)
                #ax[2].plot(responsivity[i][self.readout_powers[k]](np.abs((fit_f0s-zero_power_f0)/zero_power_f0*1e6)), (fit_f0s-zero_power_f0)/zero_power_f0*1e6, ls = "--")

                ax[2].tick_params(labelsize = 22)
                ax[2].tick_params(axis="y", labelcolor = "tab:blue")
                ax[2].legend(fontsize = 18, bbox_to_anchor = (1.10, 1), loc = "upper left") 
                ax[2].grid()
                ax2_r = ax[2].twinx()
                ax2_r.scatter(self.readout_powers,  Qs, color = "tab:red")
                ax2_r.set_ylabel("Q", fontsize = 24,color = "tab:red")
                ax2_r.tick_params(axis = "y", labelcolor = "tab:red", labelsize = 22)

        fig.suptitle("%.3f MHz " %(self.zero_power_f0s[i][self.readout_powers[k]]) + "%.1f K" % (self.optical_powers[j]), fontsize = 36)
        fig.tight_layout()
        fig.savefig(output_dir + "%.3fMHz" %
            (self.zero_power_f0s[i][self.readout_powers[k]])+"_%.1f_K.png" % (self.coldload_temps[j]))
        plt.close(fig)

    def iq_circle_fits(self, savefig = None):
        output_dir = self.sweep_dir + "fig/iq_circle_fits/"
        try:
            os.mkdir(output_dir)
        except:
            print("Directory exists or is not valid.")
        for i in range(len(self.coldload_temps)):
            for j in range(len(self.readout_powers)):
                for k in range(len(self.drones.iloc[0,0].resonators)):

                    fig, ax = plt.subplots(1, 1)
                    fig.set_figwidth(10)
                    ax.scatter(self.drones.iloc[j,i].resonators[k].s21.real, self.drones.iloc[j,i].resonators[k].s21.imag, color = "black", marker = ".")
                    ax.plot(self.drones.iloc[j,i].resonators[k].res_model.real, self.drones.iloc[j,i].resonators[k].res_model.imag, color = "grey", label = "$f_0$ = %.1f $\pm$ %.1f, \n$Q$ = %.1f $\pm$ %.1f, \n$Q_c$ = %.1f $\pm$ %.1f" % (self.drones.iloc[j,i].resonators[k].res_fit.params['f_0'].value, self.drones.iloc[j,i].resonators[k].res_fit.params['f_0'].stderr, self.drones.iloc[j,i].resonators[k].res_fit.params['Q'].value, self.drones.iloc[j,i].resonators[k].res_fit.params['Q'].stderr, self.drones.iloc[j,i].resonators[k].res_fit.params['Q_e_real'].value, self.drones.iloc[j,i].resonators[k].res_fit.params['Q_e_real'].stderr))
                    ax.legend(bbox_to_anchor=(1.05, 1.00))
                    ax.set_xlabel("I")
                    ax.set_ylabel("Q")
                    
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(output_dir + "/%.1f_K_%.0f_dB_%.3f_MHz_iq_circle.png" %(self.coldload_temps[i], self.readout_powers[j], self.drones.iloc[j,i].resonators[k].f0/1e6))
                    plt.close(fig)
   
    
    def calibrate_drones_and_timestreams(self, savefig = None):
        if savefig is not None: 
            output_dir = self.sweep_dir + "fig/narrow_drones/"
            try:
                os.mkdir(output_dir)
            except:
                print("Directory exists or is not valid.")
        for i in range(len(self.coldload_temps)):
            for j in range(len(self.readout_powers)):
                self.drones.iloc[j,i].init_resonators(output = False, savefig = True)
                
                for k in range(len(self.drones.iloc[0,0].resonators)):
                    off_tmstrm = self.off_res_timestreams.iloc[j,i].stream_s21s[k]
                    tmstrm = self.timestreams.iloc[j,i].stream_s21s[k]
                    f0_index = self.drones.iloc[j, i].resonators[k].f0_index
                    #q_res = self.drones.iloc[j, i].resonators[k].s21[f0_index].imag
                    #i_res = self.drones.iloc[j, i].resonators[k].s21[f0_index].real
                    q_res = np.mean(tmstrm.imag)
                    i_res = np.mean(tmstrm.real)
                    i_avg = np.mean(off_tmstrm.real)
                    q_avg = np.mean(off_tmstrm.imag)
                    self.off_res_timestreams.iloc[j,i].stream_s21s[k] = off_tmstrm - (i_avg-i_res + 1j *(q_avg-q_res) )
                    s21_max = np.max(np.abs(self.drones.iloc[j,i].resonators[k].s21))
                    #self.drones.iloc[j, i].resonators[k].s21 = self.drones.iloc[j, i].resonators[k].s21/s21_max

                    #self.timestreams.iloc[j, i].stream_s21s[k] = self.timestreams.iloc[j, i].stream_s21s[k]/s21_max

                #self.drones.iloc[j,i].plot_s21_dB(output = False, savefig = True)
                self.drones.iloc[j, i].calibrate_IQ_circle(savefig = True)
                self.timestreams.iloc[j,i].realign_time_stream(savefig = True)
                self.off_res_timestreams.iloc[j,i].realign_time_stream(savefig = True)
                self.drones.iloc[j,i].freq_vs_phase_fit(savefig = True)

                for k in range(len(self.drones.iloc[0,0].resonators)):
                    res = self.drones.iloc[j, i].resonators[k]
                    tmstrm = self.timestreams.iloc[j,i].stream_s21s[k]
                    tmstrm_off_res = self.off_res_timestreams.iloc[j,i].stream_s21s[k]
                    if savefig is not None:
                        #fig, ax = plt.subplots(2, 2)
                        fig = plt.figure()
                        gs = fig.add_gridspec(5,2)
                        ax1 = fig.add_subplot(gs[0, 0])
                        ax2 = fig.add_subplot(gs[0, 1])
                        ax3 = fig.add_subplot(gs[1, :])
                        ax4 = fig.add_subplot(gs[2, :])
                        ax5 = fig.add_subplot(gs[3, :])
                        ax6 = fig.add_subplot(gs[4,:])
                        fig.set_figheight(18)
                        fig.set_figwidth(11)
                        ds21 = np.ediff1d(np.abs(res.s21))
                        df = np.ediff1d(np.abs(res.f))
                      
                        phase = res.phase
                        dphase = np.ediff1d(np.abs(phase))
                        #index = np.where(ds21/df == np.max(ds21[10:]/df[10:]))[0]
                        index = np.where(dphase/df == np.max(dphase[10:]/df[10:]))[0]
                        mask = np.where(np.abs(res.f-res.f0) <= res.f0/res.Q * 1.5)
                        ax1.scatter(np.real(res.s21), np.imag(res.s21))
                        ax1.scatter(np.real(tmstrm), np.imag(tmstrm))
                        ax1.scatter(np.real(tmstrm_off_res), np.imag(tmstrm_off_res))
                        fig.suptitle("%.1f K, %.0f dB, %.3f MHz" % (self.coldload_temps[i], self.readout_powers[j], 
                            res.f0/1e6), fontsize = 24)
                        ax1.set_xlabel("I", fontsize = 14)
                        ax1.set_ylabel("Q", fontsize = 14)
                        ax1.set_aspect('equal')
                        ax2.tick_params(labelsize = 14)
                        res.plot_freq_vs_phase(ax2)
                        ax1.tick_params(labelsize = 14)
                        ax2.set_title("")
                    frequencies = []
                    frequencies = res.interpolate_freq(np.real(tmstrm), np.imag(tmstrm))
                    sxx = (frequencies/1e6 - self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]*1e6
                    q_res = res.s21[res.f0_index].imag
                    i_res = res.s21[res.f0_index].real
                    print(i_res)
                    next_index = res.f0_index[0][0] + 1
                    di = res.s21[next_index].real - i_res
                    dq = res.s21[next_index].imag - q_res
                    df = res.f[next_index] - res.f[res.f0_index]
                    didf = di/df
                    dqdf = dq/df
                    res_index = res.f0_index[0][0]
                    zpf0 = self.zero_power_f0s[k][self.readout_powers[j]]
                    df_readout = ((tmstrm.imag - q_res)* di - (tmstrm.real - i_res) * dq) /(di**2 + dq**2)
                    frequencies_off_res = []
                    frequencies_off_res = res.interpolate_freq(np.real(tmstrm_off_res), np.imag(tmstrm_off_res))
                    sxx_readout = (frequencies_off_res/1e6 - self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]*1e6
                    
                    #sxx_readout = ((df_readout + res.f0)/1e6 - zpf0)/zpf0*1e6
                    if savefig is not None:
                        ax2.scatter(frequencies, 
                                np.arctan2(np.imag(tmstrm), np.real(tmstrm)))
                        ax2.scatter(frequencies, 
                                np.arctan2(np.imag(tmstrm_off_res), np.real(tmstrm_off_res)))
                        ax2.scatter(res.f[res.f0_index], np.arctan2(np.imag(res.s21[res.f0_index]),
                                np.real(res.s21[res.f0_index])),color = "red", marker = "x", s= 200)
                        ax2.scatter(res.f[index], np.arctan2(np.imag(res.s21[index]), np.real(res.s21[index])),
                                    color = "green", marker = "x", s= 200)
                        ax2.scatter(res.f_r, res.theta(res.f_r, res.theta_0, res.Q_r, res.f_r),
                                    color = "pink", marker = "x", s= 200)
                        ax4.scatter(self.timestreams.iloc[j,i].time, sxx)
                        ax4.scatter(self.timestreams.iloc[j,i].time, sxx_readout)
                    
                
                    if savefig is not None:
                        ax1.scatter(np.real(res.s21[res.f0_index]), np.imag(res.s21[res.f0_index]), color = "red", marker = "x", s= 200)
                        ax1.scatter(np.real(res.s21[index]), np.imag(res.s21[index]), color = "green", marker = "x", s= 200)

                        ax3.scatter(res.f, np.abs(res.s21))
                        ax3.scatter(frequencies, np.abs(tmstrm))
                        ax3.scatter(frequencies, np.abs(tmstrm_off_res))
                        ax3.scatter(res.f0, np.abs(res.s21[res.f0_index]), color = "red", marker = "x", s= 200)

                        ax3.set_xlabel("Freq [MHz]")
                        ax3.set_ylabel("S21 [mag]")
                        ax3_r = ax3.twinx()
                        dS21 = np.ediff1d(np.abs(res.s21))
                        df = np.ediff1d(np.abs(res.f))
                        
                        ax3_r.scatter(res.f[1:], np.abs(dS21/df), color = "red")
                        ax3.scatter(res.f[index], np.abs(res.s21[index]), color = "green", marker = "x", s= 200)
                        print(index)
                        ax5.scatter(self.timestreams.iloc[j,i].time, self.responsivity[k][self.readout_powers[j]](-sxx))
                        ax5.scatter(self.timestreams.iloc[j,i].time, self.responsivity[k][self.readout_powers[j]](-sxx_readout))
                        ax6.scatter(self.timestreams.iloc[j,i].time, sxx_readout)
                        ax5.set_xlabel("Time [s]", fontsize = 14)
                        ax5.set_ylabel("Optical Power [pW]", fontsize = 14)
                        ax4.set_xlabel("Time [s]", fontsize = 14)
                        ax4.tick_params(labelsize = 14)
                        ax4.set_ylabel("df/f0 [ppm]", fontsize = 14)

                        plt.tight_layout()
                        dir_name = "drone_" + self.drones.iloc[j,i].drone_id + self.drones.iloc[j,i].base_temp + self.drones.iloc[j,i].load_name + self.drones.iloc[j,i].load_temp + self.drones.iloc[j,i].amp_gain + "_time_stream_summary"
                        try:
                            os.mkdir(output_dir+ dir_name)
                        except:
                                print("")
                        plt.savefig(output_dir + dir_name + "/%.1f_K_%.0f_dB_%.3f_MHz_timestream_summary.png" %(self.coldload_temps[i], self.readout_powers[j], res.f0/1e6))
                        plt.close(fig)
    def get_sxx_and_nep(self):
        sxx = []
        nep = []
        sxx_amp = []
        nep_amp = []
        sxx_avg = []
        nep_avg = []
        nep_std = []
        nep_amp_avg = []
        sxx_amp_avg = []
        nep_psd = []
        sxx_psd = []
        sxx_amp_psd = []
        nep_amp_psd = []
        
        for i in range(len(self.drones.iloc[0,0].resonators)):
            sxx.append({})
            nep.append({})
            sxx_amp.append({})
            nep_amp.append({})
            
            sxx_psd.append({})
            nep_psd.append({})
            sxx_amp_psd.append({})
            nep_amp_psd.append({})
            
            sxx_avg.append({})
            nep_avg.append({})
            nep_std.append({})
            sxx_amp_avg.append({})
            nep_amp_avg.append({})
            
        for j in range(len(self.readout_powers)):
            for k in range(len(self.drones.iloc[0,0].resonators)):
                sxx[k][self.readout_powers[j]] = []
                nep[k][self.readout_powers[j]] = []
                sxx_amp[k][self.readout_powers[j]] = []
                nep_amp[k][self.readout_powers[j]] = []
                
                nep_psd[k][self.readout_powers[j]] = []
                nep_avg[k][self.readout_powers[j]] = []
                nep_std[k][self.readout_powers[j]] = []
                sxx_amp_psd[k][self.readout_powers[j]] = []
                nep_amp_psd[k][self.readout_powers[j]] = []
                
                sxx_amp_avg[k][self.readout_powers[j]] = []
                sxx_psd[k][self.readout_powers[j]] = []
                sxx_avg[k][self.readout_powers[j]] = []
                nep_amp_avg[k][self.readout_powers[j]] = []
                for i in range(len(self.coldload_temps)):
                    res = self.drones.iloc[j, i].resonators[k]
                    stream = self.timestreams.iloc[j, i].stream_s21s[k]
                    tmstrm_off_res = self.off_res_timestreams.iloc[j, i].stream_s21s[k]
                    q_res = res.s21[res.f0_index].imag
                    i_res = res.s21[res.f0_index].real
                    next_index = res.f0_index[0][0] + 1
                    di = res.s21[next_index].real - i_res
                    dq = res.s21[next_index].imag - q_res
                    df = res.f[next_index] - res.f[res.f0_index]
                    
                    didf = di/df
                    dqdf = dq/df
                    res_index = res.f0_index[0][0]
                    zpf0 = self.zero_power_f0s[k][self.readout_powers[j]]
                    df_readout = ((stream.imag - q_res)* di - (stream.real - i_res) * dq) /(di**2 + dq**2)*df
                    sxx_dissipation = ((res.f[res.f0_index] + df_readout)/1e6 - self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]
                    stream_90 = stream.copy()
                    stream_90_i_center = np.mean(stream_90.real)
                    stream_90_q_center = np.mean(stream_90.imag)
                    stream_90 = stream_90 - (stream_90_i_center + 1j*stream_90_q_center)
                    stream_90 = stream_90 * np.exp(1j*np.pi/2)
                    stream_90 = stream_90 + (stream_90_i_center + 1j*stream_90_q_center)
                    
                    frequencies_off_res = res.interpolate_freq(np.real(tmstrm_off_res), np.imag(tmstrm_off_res))
                    #frequencies_off_res = res.interpolate_freq(np.real(stream_90), np.imag(stream_90))
                    
                    sxx_readout = (frequencies_off_res/1e6 - self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]
                    #sxx_readout = sxx_dissipation
                    sxx_amp[k][self.readout_powers[j]].append(sxx_readout)
                    nep_amp[k][self.readout_powers[j]].append(np.array(self.responsivity[k][self.readout_powers[j]](-1*sxx_amp[k][self.readout_powers[j]][i]*1e6))/1e12)
                    f, psd_sxx_amp = signal.welch(sxx_amp[k][self.readout_powers[j]][i], fs=512e6 / 2 ** 20, nperseg=256);
                    sxx_amp_psd[k][self.readout_powers[j]].append((f, psd_sxx_amp))
                    sxx_amp_avg[k][self.readout_powers[j]].append(np.mean(psd_sxx_amp[np.where((f > 90) & (f < 110))]))
                    f, psd_nep_amp = signal.welch(nep_amp[k][self.readout_powers[j]][i], fs=512e6 / 2 ** 20, nperseg=256);
                    nep_amp_psd[k][self.readout_powers[j]].append((f, psd_nep_amp))
                    nep_amp_avg[k][self.readout_powers[j]].append(np.mean(np.sqrt(psd_nep_amp[np.where((f > 90) & (f < 110))])))
                    
                    freq = res.interpolate_freq(np.real(stream), np.imag(stream))/1e6
                    sxx[k][self.readout_powers[j]].append(np.array((freq - self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]))
                    nep[k][self.readout_powers[j]].append(np.array(self.responsivity[k][self.readout_powers[j]](-1*sxx[k][self.readout_powers[j]][i]*1e6))/1e12)
                    f, psd = signal.welch(nep[k][self.readout_powers[j]][i], fs=512e6 / 2 ** 20, nperseg=256) 
                    
                    #psd = psd - psd_nep_amp
                    nep_psd[k][self.readout_powers[j]].append((f, psd))
                    nep_avg[k][self.readout_powers[j]].append(np.mean(np.sqrt(psd[np.where((f > 90) & (f < 110))])))
                    nep_std[k][self.readout_powers[j]].append(np.std(np.sqrt(psd[np.where((f > 90) & (f < 110))])))

                    f, psd_sxx = signal.welch(sxx[k][self.readout_powers[j]][i], fs=512e6 / 2 ** 20, nperseg=256);
                    sxx_psd[k][self.readout_powers[j]].append((f, psd_sxx))
                    sxx_avg[k][self.readout_powers[j]].append(np.mean(psd_sxx[np.where((f > 90) & (f < 110))]))
                    
        for i in range(len(self.drones.iloc[0,0].resonators)):
            print('test')
            sxx[i] = pd.DataFrame(sxx[i], index = self.optical_powers)
            sxx_psd[i] = pd.DataFrame(sxx_psd[i], index = self.optical_powers)
            sxx_avg[i] = pd.DataFrame(sxx_avg[i], index = self.optical_powers)
            
            nep[i] = pd.DataFrame(nep[i], index = self.optical_powers)
            nep_avg[i] = pd.DataFrame(nep_avg[i], index = self.optical_powers)
            nep_std[i] = pd.DataFrame(nep_std[i], index = self.optical_powers)
            nep_psd[i] = pd.DataFrame(nep_psd[i], index = self.optical_powers)
            
            sxx_amp[i] = pd.DataFrame(sxx_amp[i], index = self.optical_powers)
            sxx_amp_psd[i] = pd.DataFrame(sxx_amp_psd[i], index = self.optical_powers)
            sxx_amp_avg[i] = pd.DataFrame(sxx_amp_avg[i], index = self.optical_powers)
            
            nep_amp[i] = pd.DataFrame(nep_amp[i], index = self.optical_powers)
            nep_amp_avg[i] = pd.DataFrame(nep_amp_avg[i], index = self.optical_powers)
            nep_amp_psd[i] = pd.DataFrame(nep_amp_psd[i], index = self.optical_powers)
        self.sxx = sxx
        self.sxx_psd = sxx_psd
        self.sxx_avg = sxx_avg
        
        self.nep = nep
        self.nep_psd = nep_psd
        self.nep_avg = nep_avg
        self.nep_std = nep_std
        self.sxx_amp = sxx_amp
        self.sxx_amp_psd = sxx_amp_psd
        self.sxx_amp_avg = sxx_amp_avg
        self.nep_amp = nep_amp
        self.nep_amp_psd = nep_amp_psd
        self.nep_amp_avg = nep_amp_avg
         
    def detector_characterization_summary(self):
        try:
            os.mkdir(self.sweep_dir + "fig/detector_characterization_summary")
        except:
            print("Directory already exists.")
        dir_name = "detector_characterization_summary/"
        for j in range(len(self.readout_powers)):

            #fig, ax = plt.subplots(7, 1)


            for k in range(len(self.drones.iloc[0,0].resonators)):

                fig, ax = plt.subplots(2, 2);
                matplotlib.rcParams.update({'font.size': 22, "font.family": "serif", "mathtext.fontset": "dejavuserif"})
                fig.suptitle("%.1f dB, %.3f MHz" % (self.readout_powers[j], self.all_f0s[k][self.readout_powers[j]][0]));
                fig.set_figheight(12.5);
                fig.set_figwidth(12.5);
                colormap = plt.cm.nipy_spectral;
                colors = colormap(np.linspace(0, 1, len(self.coldload_temps)));

                ax[0,0].scatter(self.optical_powers, (self.all_f0s[k][self.readout_powers[j]]-self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]*1e6, color = "black");
                ax[0,0].set_xlabel("Optical Power [$pW$]");
                ax[0,0].set_ylabel("${\delta f/ f_0}$ [$ppm$]");
                sxx_0 = (self.all_f0s[k][self.readout_powers[j]]-self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]*1e6
                ax[0, 0].plot(self.responsivity[k][self.readout_powers[j]](-sxx_0), sxx_0, ls = "--", color = "black");

                ax[0, 1].set_prop_cycle('color', colors);
                ax[1, 0].set_prop_cycle('color', colors);
                for i in range(len(self.coldload_temps)):
                    ax[0, 1].set_prop_cycle('color', colors);

                    res = self.drones.iloc[j, i].resonators[k]
                    stream = self.timestreams.iloc[j, i].stream_s21s[k] 
                    ax[0, 1].set_xlabel("f [$Hz$]");
                    ax[0, 1].set_ylabel("$NEP$ [$W/\sqrt{Hz}$]");
                    ax[0, 1].plot();

                    #freq = res.interpolate_freq(np.real(stream), np.imag(stream))/1e6
                    #sxx = (freq - self.zero_power_f0s[k][self.readout_powers[j]])/self.zero_power_f0s[k][self.readout_powers[j]]*1e6
                    #power = self.responsivity[k][self.readout_powers[j]](-sxx)/1e12


                    #f, psd = signal.welch(power, fs=512e6 / 2 ** 20, nperseg=256)  
                    f, psd = self.nep_psd[k].iloc[i, j]
                    #avg_noise.append(np.mean(psd[np.where((f > 90) & (f < 110))]));
                    ax[0, 1].plot(f[:-1], np.sqrt(psd[:-1]), label = "%.2f $pW$" % (self.optical_powers[i]), color = colors[i]);
                    
                    ax[0, 1].legend(fontsize = "14");
                    ax[0, 1].set_yscale("log");
                    ax[0, 1].set_xscale("log");

                    ax[1,0].set_xlabel("f [Hz]");
                    ax[1,0].set_ylabel("$S_{\delta f / f_0}$ [$Hz^{-1}$]");
                    ax[1,0].plot();
                    #f, psd_sxx = signal.welch(sxx/1e6, fs=512e6 / 2 ** 20, nperseg=256);
                    f, psd_sxx = self.sxx_psd[k].iloc[i, j]
                    ax[1, 0].plot(f[:-1], psd_sxx[:-1], label = "%.2f $pW$"% (self.optical_powers[i]));
                    ax[1, 0].legend(fontsize = "14");
                    ax[1, 0].set_yscale("log");
                    ax[1, 0].set_xscale("log");
                    
                for i in range(len(self.coldload_temps)):
                    f, psd = self.nep_amp_psd[k].iloc[i, j]
                    ax[0, 1].plot(f[:-1], np.sqrt(psd[:-1]), linestyle = "dashed", color = colors[i]);
                    
                ax[1,1].scatter(self.optical_powers, self.nep_avg[k][self.readout_powers[j]], color = "black", label = "$NEP$")
                ax[1,1].scatter(self.optical_powers, self.sxx_avg[k][self.readout_powers[j]], color = "red", label = "$S_{xx}$")
                ax[1,1].scatter(self.optical_powers, self.sxx_amp_avg[k][self.readout_powers[j]], color = "blue", label = "$S_{xx}$ readout")
                ax[1,1].scatter(self.optical_powers, self.nep_amp_avg[k][self.readout_powers[j]], color = "green", label = "$NEP$ readout")

                ax[1,1].set_yscale("log")
                ax[1,1].set_xscale("log")
                ax[1,1].set_xlabel("Optical Power [$pW$]")
                ax[1,1].set_ylabel("$NEP$ [$W/\sqrt{Hz}$] & $S_{xx}$ [$Hz$]")
                ax[1,1].legend()
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(self.sweep_dir + "fig/" + dir_name + self.date['low'] + "_%.0f_dB_%.3f_MHz_det_char_summ.png" %(self.readout_powers[j], self.all_f0s[k][self.readout_powers[j]][0]))
                
                #ax[k].scatter(np.real(stream[50:]-np.mean(stream[50:])), np.imag(stream[50:]-np.mean(stream[50:])))
                #ax[k].set_title("%.3f MHz" % (np.real(res.f0/1e6)))

    
    def plot_nep_only(self, plot_readout = False): 
        matplotlib.rcParams.update({'font.size': 14, "font.family": "serif", "mathtext.fontset": "dejavuserif"})

        try:
            os.mkdir(self.sweep_dir + "fig/nep_surface")
        except:
            print("Directory already exists.")
        dir_name = "nep_surface/"
        for k in range(len(self.drones.iloc[0,0].resonators)):
            colormap = plt.cm.coolwarm
            colors2 = colormap(np.linspace(0, 1, len(self.readout_powers)));

            spec = matplotlib.gridspec.GridSpec(ncols=3, nrows = 1, width_ratios = [1, 1,1])
            fig = plt.figure()
            fig.suptitle("%.3f MHz" %(self.all_f0s[k][self.readout_powers[0]][0]), y=1.0)

            fig.set_figwidth(25)
            ax =  fig.add_subplot(spec[0]);
            ax.set_prop_cycle('color', colors2);
            for read_pow in range(len(self.readout_powers)):
                ax.plot(self.optical_powers, self.nep_avg[k].iloc[:, read_pow], label = "%.1f dB" %self.readout_powers[read_pow])

            bounds = range(len(self.readout_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f dB" % power for power in self.readout_powers]
            labels.append(0)
            cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar.set_ticklabels(labels)
            
            #norm = matplotlib.colors.Normalize(vmin= self.readout_powers[0], vmax = self.readout_powers[-1])
            #cbar3= fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm = norm), orientation = 'vertical', pad=.08)
            #cbar3.ax.tick_params(labelsize = 14)
            #cbar3.ax.set_ylabel('Readout Powers [$dB$]', rotation=270, fontsize = 14, labelpad=25)
            ax.set_xlabel("Optical Power [$W$]")
            ax.set_ylabel("$NEP$ [$W/\sqrt{Hz}]$")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title("On-Resonance NEP")
           
            colors2 = colormap(np.linspace(0, 1, len(self.readout_powers)));

            ax2 =  fig.add_subplot(spec[1]);
            ax2.set_prop_cycle('color', colors2);
            for read_pow in range(len(self.readout_powers)):
                ax2.plot(self.optical_powers, self.nep_amp_avg[k].iloc[:, read_pow], label = "%.1f dB" %self.readout_powers[read_pow])

            bounds = range(len(self.readout_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f dB" % power for power in self.readout_powers]
            labels.append(0)
            cbar3 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar3.set_ticklabels(labels)
            
            #norm = matplotlib.colors.Normalize(vmin= self.readout_powers[0], vmax = self.readout_powers[-1])
            #cbar3= fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm = norm), orientation = 'vertical', pad=.08)
            #cbar3.ax.tick_params(labelsize = 14)
            #cbar3.ax.set_ylabel('Readout Powers [$dB$]', rotation=270, fontsize = 14, labelpad=25)
            ax2.set_xlabel("Optical Power [$W$]")
            ax2.set_ylabel("$NEP$ [$W/\sqrt{Hz}]$")
            ax2.set_yscale("log")
            ax2.set_xscale("log")
            ax2.set_title("Off Resonance NEP")
            
            ax3 =  fig.add_subplot(spec[2]);
            ax3.set_prop_cycle('color', colors2);
            for read_pow in range(len(self.readout_powers)):
            
                ax3.plot(self.optical_powers,  self.nep_avg[k].iloc[:, read_pow]- self.nep_amp_avg[k].iloc[:, read_pow], label = "%.1f dB" %self.readout_powers[read_pow])

            bounds = range(len(self.readout_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f dB" % power for power in self.readout_powers]
            labels.append(0)
            cbar3 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar3.set_ticklabels(labels)
            
            #norm = matplotlib.colors.Normalize(vmin= self.readout_powers[0], vmax = self.readout_powers[-1])
            #cbar3= fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm = norm), orientation = 'vertical', pad=.08)
            #cbar3.ax.tick_params(labelsize = 14)
            #cbar3.ax.set_ylabel('Readout Powers [$dB$]', rotation=270, fontsize = 14, labelpad=25)
            ax3.set_xlabel("Optical Power [$W$]")
            ax3.set_ylabel("$NEP$ [$W/\sqrt{Hz}]$")
            ax3.set_yscale("log")
            ax3.set_xscale("log")
            ax3.set_title("NEP On Resonance minus NEP Off Resonance")
            
            
            #ax3.legend(bbox_to_anchor=(1.1, 1.00))
                        
            
            fig.tight_layout()
            
            plt.savefig(self.sweep_dir + "fig/" + dir_name + self.date['low'] + "%.3f_MHz_nep_only.png" %(self.all_f0s[k][self.readout_powers[0]][0]), bbox_inches="tight")

    
    def plot_nep_surface(self, plot_readout = False): 
        matplotlib.rcParams.update({'font.size': 14, "font.family": "serif", "mathtext.fontset": "dejavuserif"})

        try:
            os.mkdir(self.sweep_dir + "fig/nep_surface")
        except:
            print("Directory already exists.")
        dir_name = "nep_surface/"
        for k in range(len(self.drones.iloc[0,0].resonators)):
            spec = matplotlib.gridspec.GridSpec(ncols=3, nrows = 1, width_ratios = [3, 1,1])
            fig = plt.figure()
            fig.set_figwidth(25)
            
            ax =  fig.add_subplot(spec[0], projection = "3d")
            Y = self.optical_powers
            X = np.array(self.readout_powers)
            Z_psd = np.empty([Y.shape[0], X.shape[0]], dtype = float)
            Z_sxx = np.empty([Y.shape[0], X.shape[0]], dtype = float)
            for x in range(len(X)):
                for y in range(len(Y)):
                    if not plot_readout:
                        Z_psd[y,x] = self.nep_avg[k].iloc[y,x]
                    else:
                        Z_psd[y,x] = self.nep_amp_avg[k].iloc[y,x]
            X, Y = np.meshgrid(X, Y)
            p = ax.plot_surface(X, Y, Z_psd,cmap=cm.coolwarm, norm=matplotlib.colors.LogNorm())
            ax.scatter(X, Y, Z_psd,color = "black", label = "$NEP$ %.3f MHz" %(self.all_f0s[k][self.readout_powers[0]][0]))
            #ax.legend(loc= "upper right", fontsize = 14)
            ax.set_ylabel("\nOptical Power (pW)", fontsize = 14)
            ax.set_xlabel("\nReadout Power (dB)", fontsize = 14)
            ax.tick_params(labelsize = 14)
            ax.set_zscale("log")
            if not plot_readout:
                fig.suptitle("%.3f MHz" %(self.all_f0s[k][self.readout_powers[0]][0]), y=0.9)
            else:
                fig.suptitle("%.3f MHz (Readout, Off-Resonance)" %(self.all_f0s[k][self.readout_powers[0]][0]), y=0.9)
            cbar = fig.colorbar(p, shrink = 0.6, pad=.05)
            cbar.ax.tick_params(labelsize = 14)
            cbar.ax.set_ylabel('$NEP$ [$W/\sqrt{Hz}$]', rotation = 270, fontsize = 14, labelpad=20)
            #ax.invert_xaxis()
            #ax.set_zlim(np.min(Z_psd), .6*np.max(Z_psd))
            ax.set_zticks([])
            ax2 =  fig.add_subplot(spec[1])
            ax2.set_yscale("log")
            
            colormap = plt.cm.coolwarm;
            colors = colormap(np.linspace(0, 1, len(self.optical_powers)));
            colors2 = colormap(np.linspace(0, 1, len(self.readout_powers)));

            ax2.set_prop_cycle('color', colors);
            
            for cold_temp in range(len(self.coldload_temps)):
                if not plot_readout:
                    p2 = ax2.plot(self.readout_powers, self.nep_avg[k].iloc[cold_temp, :], label = "%.1f K" %self.coldload_temps[cold_temp])
                else:
                    p2 = ax2.plot(self.readout_powers, self.nep_amp_avg[k].iloc[cold_temp, :], label = "%.1f K" %self.coldload_temps[cold_temp])
            bounds = range(len(self.optical_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f pW" % power for power in self.optical_powers]
            labels.append(0)
            cbar2 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar2.set_ticklabels(labels)
            
            #norm = matplotlib.colors.Normalize(vmin= self.optical_powers[0], vmax = self.optical_powers[-1])
            #cbar2 = fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm = norm), orientation = 'vertical', pad=.08)
            #cbar2.ax.tick_params(labelsize = 14)
            #cbar2.ax.set_ylabel('Optical Powers [$W$]', rotation=270, fontsize = 14, labelpad=25)
            ax2.set_xlabel("Readout Power [$dB$]")
            ax2.set_ylabel("$NEP$ [$W/\sqrt{Hz}]$")
            ax3 =  fig.add_subplot(spec[2]);
            ax3.set_prop_cycle('color', colors2);
            for read_pow in range(len(self.readout_powers)):
                if not plot_readout:
                    ax3.plot(self.optical_powers, self.nep_avg[k].iloc[:, read_pow], label = "%.1f dB" %self.readout_powers[read_pow])
                else:
                    ax3.plot(self.optical_powers, self.nep_amp_avg[k].iloc[:, read_pow], label = "%.1f dB" %self.readout_powers[read_pow])
            bounds = range(len(self.readout_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f dB" % power for power in self.readout_powers]
            labels.append(0)
            cbar3 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar3.set_ticklabels(labels)
            
            #norm = matplotlib.colors.Normalize(vmin= self.readout_powers[0], vmax = self.readout_powers[-1])
            #cbar3= fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm = norm), orientation = 'vertical', pad=.08)
            #cbar3.ax.tick_params(labelsize = 14)
            #cbar3.ax.set_ylabel('Readout Powers [$dB$]', rotation=270, fontsize = 14, labelpad=25)
            ax3.set_xlabel("Optical Power [$W$]")
            ax3.set_ylabel("$NEP$ [$W/\sqrt{Hz}]$")
            ax3.set_yscale("log")
            ax3.set_xscale("log")
            
            #ax3.legend(bbox_to_anchor=(1.1, 1.00))
                        
            
            fig.tight_layout()
            if not plot_readout:
                plt.savefig(self.sweep_dir + "fig/" + dir_name + self.date['low'] + "%.3f_MHz_nep_surface.png" %(self.all_f0s[k][self.readout_powers[0]][0]), bbox_inches="tight")
            else:
                plt.savefig(self.sweep_dir + "fig/" + dir_name + self.date['low'] + "%.3f_MHz_nep_surface_readout_off_res.png" %(self.all_f0s[k][self.readout_powers[0]][0]), bbox_inches="tight")
    
    
    
    def plot_qi(self):
        matplotlib.rcParams.update({'font.size': 14, "font.family": "serif", "mathtext.fontset": "dejavuserif"})
        try:
            os.mkdir(self.sweep_dir + "fig/qi_vs_readout")
            
        except:
            print("Directory already exists.")
        dir_name = "qi_vs_readout/"
        for k in range(len(self.drones.iloc[0,0].resonators)):
            fig, ax = plt.subplots(1, 3)
            #cycle = pplt.Cycle('reds', len(self.optical_powers))
            colormap = plt.cm.coolwarm;
            #norm = matplotlib.colors.BoundaryNorm(self.optical_powers, colormap.N)
            colors = colormap(np.linspace(0, 1, len(self.optical_powers)));
            #print(colors)
            fig.set_figwidth(13)
            ax[0].set_prop_cycle('color', colors);
            ax[1].set_prop_cycle('color', colors);
            ax[2].set_prop_cycle('color', colors);
            Qs = []
            Qcs = []
            Qis = []
            read_pows = []
            for j in range(len(self.coldload_temps)):
                Qcs.append([])
                Qs.append([])
                Qis.append([])
                read_pows.append([])
                for i in range(len(self.readout_powers)):
                   
                    Qs[j].append(self.drones_2_MHz.iloc[i, j].resonators[k].Q)
                    Qcs[j].append(self.drones_2_MHz.iloc[i, j].resonators[k].params['Q_e_real'].value)
                    
                Qis[j] = [1/(1/Qs[j][m] - 1/Qcs[j][m]) for m in range(len(Qs[j]))]
                read_pows[j] = self.readout_powers
                lines = ax[0].plot(self.readout_powers, Qs[j][:], label = "%.1f" % self.coldload_temps[j])
                lines_2 = ax[1].plot(self.readout_powers, Qis[j][:], label = "%.1f" % self.coldload_temps[j])
                lines_3 = ax[2].plot(self.readout_powers, Qcs[j][:], label = "%.1f" % self.coldload_temps[j])
            bounds = range(len(self.optical_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f pW" % power for power in self.optical_powers]
            labels.append(0)
            cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar.set_ticklabels(labels)
            ax[0].set_xlabel("Readout Power [dB]")
            ax[0].set_ylabel("$Q$")
            ax[1].set_xlabel("Readout Power [dB]")
            ax[1].set_ylabel("$Q_i$")
            ax[2].set_xlabel("Readout Power [dB]")
            ax[2].set_ylabel("$Q_c$")
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
            ax[2].set_yscale("log")
            fig.suptitle("%.3f MHz" %(self.all_f0s[k][self.readout_powers[0]][0]))
            fig.tight_layout()
            
            plt.savefig(self.sweep_dir + "fig/" + dir_name + self.date['low'] + "%.3f_MHz_qi_vs_readout.png" %(self.all_f0s[k][self.readout_powers[0]][0]), bbox_inches="tight")
    def plot_sxx_surface(self, plot_readout = False): 
        matplotlib.rcParams.update({'font.size': 14, "font.family": "serif", "mathtext.fontset": "dejavuserif"})

        try:
            os.mkdir(self.sweep_dir + "fig/sxx_surface")
        except:
            print("Directory already exists.")
        dir_name = "sxx_surface/"
        for k in range(len(self.drones.iloc[0,0].resonators)):
            spec = matplotlib.gridspec.GridSpec(ncols=3, nrows = 1, width_ratios = [3, 1,1])
            fig = plt.figure()
            fig.set_figwidth(25)
            
            ax =  fig.add_subplot(spec[0], projection = "3d")
            Y = self.optical_powers
            X = np.array(self.readout_powers)
            Z_psd = np.empty([Y.shape[0], X.shape[0]], dtype = float)
            Z_sxx = np.empty([Y.shape[0], X.shape[0]], dtype = float)
            for x in range(len(X)):
                for y in range(len(Y)):
                    if not plot_readout:
                        Z_sxx[y,x] = self.sxx_avg[k].iloc[y,x]
                    else:
                        Z_sxx[y,x] = self.sxx_amp_avg[k].iloc[y,x]
            X, Y = np.meshgrid(X, Y)
            p = ax.plot_surface(X, Y, Z_sxx,cmap=cm.coolwarm, norm=matplotlib.colors.LogNorm())
            ax.scatter(X, Y, Z_sxx,color = "black", label = "$S_{xx}$ %.3f MHz" %(self.all_f0s[k][self.readout_powers[0]][0]))
            #ax.legend(loc= "upper right", fontsize = 14)
            if not plot_readout:
                fig.suptitle("%.3f MHz" %(self.all_f0s[k][self.readout_powers[0]][0]), y = 0.90)
            else:
                fig.suptitle("%.3f MHz (Readout)" %(self.all_f0s[k][self.readout_powers[0]][0]), y = 0.90)
            #ax.invert_xaxis()
            ax.set_ylabel("\nOptical Power (pW)", fontsize = 14)
            ax.set_xlabel("\nReadout Power (dB)", fontsize = 14)
            ax.tick_params(labelsize = 14)
            ax.set_zscale("log")
            fig.suptitle("%.3f MHz" %(self.all_f0s[k][self.readout_powers[0]][0]), y=0.9)
            cbar = fig.colorbar(p, shrink = 0.6, pad=.05)
            cbar.ax.tick_params(labelsize = 14)
            cbar.ax.set_ylabel('$S_{xx}$ [$Hz$]', rotation = 270, fontsize = 14, labelpad=20)
            #ax.invert_xaxis()
            #ax.set_zlim(np.min(Z_psd), .6*np.max(Z_psd))
            ax.set_zticks([])
            ax2 =  fig.add_subplot(spec[1])
            ax2.set_yscale("log")
            
            
            colormap = plt.cm.coolwarm;
            colors = colormap(np.linspace(0, 1, len(self.optical_powers)));
            colors2 = colormap(np.linspace(0, 1, len(self.readout_powers)));

            ax2.set_prop_cycle('color', colors);
            
            for cold_temp in range(len(self.coldload_temps)):
                if not plot_readout:
                    p2 = ax2.plot(self.readout_powers, self.sxx_avg[k].iloc[cold_temp, :], label = "%.1f K" %self.coldload_temps[cold_temp])
                else:
                    p2 = ax2.plot(self.readout_powers, self.sxx_amp_avg[k].iloc[cold_temp, :], label = "%.1f K" %self.coldload_temps[cold_temp])
            bounds = range(len(self.optical_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f pW" % power for power in self.optical_powers]
            labels.append(0)
            cbar2 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar2.set_ticklabels(labels)
            
            #norm = matplotlib.colors.Normalize(vmin= self.optical_powers[0], vmax = self.optical_powers[-1])
            #cbar2 = fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm = norm), orientation = 'vertical', pad=.08)
            #cbar2.ax.tick_params(labelsize = 14)
            #cbar2.ax.set_ylabel('Optical Powers [$W$]', rotation=270, fontsize = 14, labelpad=25)
            ax2.set_xlabel("Readout Power [$dB$]")
            ax2.set_ylabel("$S_{xx}$ [$Hz$]")
            ax3 =  fig.add_subplot(spec[2]);
            ax3.set_prop_cycle('color', colors2);
            ax3.set_yscale("log")
            ax3.set_xscale("log")
            for read_pow in range(len(self.readout_powers)):
                if not plot_readout:
                    ax3.plot(self.optical_powers, self.sxx_avg[k].iloc[:, read_pow], label = "%.1f dB" %self.readout_powers[read_pow])
                else:
                    ax3.plot(self.optical_powers, self.sxx_amp_avg[k].iloc[:, read_pow], label = "%.1f dB" %self.readout_powers[read_pow])
            bounds = range(len(self.readout_powers)+1)
            ticks = np.array(bounds) + .5
            norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
            labels = ["%.1f dB" % power for power in self.readout_powers]
            labels.append(0)
            cbar3 = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
            cbar3.set_ticklabels(labels)
            
            
            #norm = matplotlib.colors.Normalize(vmin= self.readout_powers[0], vmax = self.readout_powers[-1])
            #cbar3= fig.colorbar(cm.ScalarMappable(cmap=cm.coolwarm, norm = norm), orientation = 'vertical', pad=.08)
            #cbar3.ax.tick_params(labelsize = 14)
            #cbar3.ax.set_ylabel('Readout Powers [$dB$]', rotation=270, fontsize = 14, labelpad=25)
            ax3.set_xlabel("Optical Power [$W$]")
            ax3.set_ylabel("$S_{xx}$ [$Hz$]")
            fig.tight_layout()
            if not plot_readout:
                plt.savefig(self.sweep_dir + "fig/" + dir_name + self.date['low'] + "%.3f_MHz_sxx_surface.png" %(self.all_f0s[k][self.readout_powers[0]][0]), bbox_inches="tight")
            else:
                plt.savefig(self.sweep_dir + "fig/" + dir_name + self.date['low'] + "%.3f_MHz_sxx_surface_readout.png" %(self.all_f0s[k][self.readout_powers[0]][0]), bbox_inches="tight")
    
    def nep_model(self, power_pw, nep_0, eta, B):
        h = 6.62606015E-34
        nu = 256.42614e9
        power = np.array(power_pw)/1e12
        return np.sqrt(nep_0**2 + 2*h*nu*power/eta + 2*power**2/B)
    
    
    def m(self, nu, T):
        h = 6.62606015E-34
        k_b = 1.3806493e-23
        T = np.array(T)
        return 1/(np.exp(h*nu/k_b*T)-1)
    
    def nep_r(self, power_pw, nep_0, eta, nu = 256.42614e9, T_c = 1.2):
        h = 6.62606015E-34
        k_b = 1.38*10**(-23)
        super_gap = 1.76*k_b*T_c
        power = np.array(power_pw)/1e12
        eta_pb = 0.57
        start_index = len(self.optical_powers)-len(power_pw)
        m_calc = self.m(nu, self.coldload_temps[start_index:])
        return np.sqrt(nep_0**2 +2*h*nu*power*(1+m_calc*eta)/eta + 2*power * super_gap/(eta_pb*eta))
    def nep_photon(self, power_pw, nep_0, eta):
        h = 6.62606015E-34
        nu = 256.42614e9
        start_index = len(self.optical_powers)-len(power_pw)
        m_calc = self.m(nu, self.coldload_temps[start_index:])
        power = np.array(power_pw)/1e12
        return np.sqrt(nep_0**2 +2*h*nu*power*(1+m_calc*eta)/eta)
    def recomb_noise(self, power_pw, nep_0, eta, T_c = 1.2):
        eta_pb = 0.57
        k_b = 1.38*10**(-23)
        super_gap = 1.76*k_b*T_c
        power = np.array(power_pw)/1e12
        return np.sqrt(2 * power * super_gap/(eta_pb*eta))
    def nep_photon_limit(self, power_pw, eta):
        h = 6.62606015E-34
        nu = 256.42614e9
        start_index = len(self.optical_powers)-len(power_pw)
        m_calc = self.m(nu, self.coldload_temps[start_index:])
        power = np.array(power_pw)/1e12
        return np.sqrt(2*h*nu*power*(1+m_calc*eta)/eta)
    
    def optical_efficiency_estimates(self):
        try:
            os.mkdir(self.sweep_dir + "fig/nep_vs_optical_power")
        except:
            print("Directory exists")
    
        h = 6.62606015E-34
        k_b = 1.3806493e-23
        nu = 256.42614e9

        delta = 5.447400556e-23

        matplotlib.rcParams.update({'font.size': 18, "font.family": "serif", "mathtext.fontset": "dejavuserif"})

        dir_name = "nep_vs_optical_power"
        
        for j in range(len(self.readout_powers)):
        #for j in [4]:
            h = 6.62606015E-34
            nu = 256.42614e9
            plt.figure()
            fig, ax = plt.subplots(len(self.drones.iloc[0, 0,].resonators), 1)
            #fig.suptitle("%.1f dB" % (self.readout_powers[j]))
            fig.set_figheight(25)
            fig.set_figwidth(14)
            
            for k in range(len(self.drones.iloc[0, 0,].resonators)):
            #for k in [0]:
                #ax_r = ax[k].twinx()
                #ax_r.scatter(self.optical_powers, avg_noise_sxx[k], marker = '.', color = "red")
                #p0 = [self.nep_avg[k].iloc[0,j], 0.65, 57.67e9]
                p0 = [self.nep_avg[k].iloc[0,j], 0.65]
                
                fit_powers = np.linspace(self.optical_powers[0], self.optical_powers[-1], 100)
                #print(np.sqrt(np.diag(pcov))[1])
                try: 
                    popt_photon, pcov = curve_fit(self.nep_photon, self.optical_powers[1:], self.nep_avg[k].iloc[:, j].tolist()[1:], p0)
                    ax[k].scatter(self.optical_powers, self.nep_avg[k].iloc[:, j], marker = '.')

                    #ax[k].plot(fit_powers, self.nep_photon(fit_powers, popt[0], popt[1]), label="$\eta = $ %.2f $\pm$ %.2f,$nep_0$ = %.3fE-17" %(popt[1], np.sqrt(np.diag(pcov))[1], popt[0]*1e17))
                except:
                    print("curve fit failed")
                try: 
                    p0 = [popt_photon[0], popt_photon[1]]
                    popt, pcov = curve_fit(self.nep_r, self.optical_powers[5:], self.nep_avg[k].iloc[:, j].tolist()[5:], p0)
                    ax[k].plot(self.optical_powers, self.nep_r(self.optical_powers, popt[0], popt[1]), label="$\eta = $ %.2f $\pm$ %.2f, \n $nep_0$ = %.1fE-17 $\pm$ %.1fE-17" %(popt[1], np.sqrt(np.diag(pcov))[1],  popt[0]*1e17, np.sqrt(np.diag(pcov))[0]*1e17))
                    #ax[k].plot(self.optical_powers, self.nep_photon_limit(self.optical_powers, popt[1]), "--")
                    #ax[k].set_ylim(.95*popt[0])
                    print()
                except:
                    print("curve fit failed")
                    
                
                
                ax[k].legend(fontsize = 20, bbox_to_anchor=(1.1, 1.05))
                ax[k].set_yscale("log")
                ax[k].set_xscale("log")
                ax[k].set_title("%.3f MHz, %.1f dB" %(self.zero_power_f0s[k][self.readout_powers[j]], self.readout_powers[j]))
                ax[k].set_xlabel("Optical Power [$pW$]")
                ax[k].set_ylabel("$NEP_m$ [$W/\sqrt{Hz}$]")
                
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(self.sweep_dir + "fig/"+ dir_name + "/%.0f_dB_nep_vs_optical_power.png" %(self.readout_powers[j]))
            #plt.savefig(self.sweep_dir + "fig/"+ dir_name + "/nep_vs_optical_power.png" )
    def off_res_timestreams_psd(self):
        try:
            os.mkdir(self.sweep_dir + "fig/off_res_psds")
            
        except:
            print("Directory already exists.")
        dir_name = "off_res_psds/"
        for k in range(len(self.drones.iloc[0, 0,].resonators)):
            for i in range(len(self.coldload_temps)):
                fig, ax = plt.subplots()
                colormap = plt.cm.coolwarm;
                #norm = matplotlib.colors.BoundaryNorm(self.optical_powers, colormap.N)
                colors = colormap(np.linspace(0, 1, len(self.readout_powers)));
                #print(colors)
                ax.set_prop_cycle('color', colors);
                for j in range(len(self.readout_powers)):
                    stream = self.off_res_timestreams.iloc[j, i].stream_s21s[k]
                    f, psd = signal.welch(np.abs(stream), fs=512e6 / 2 ** 20, nperseg=256);
                    ax.plot(f, psd, label = "%.1f dB" % self.readout_powers[j])
                    bounds = range(len(self.readout_powers)+1)
                ticks = np.array(bounds) + .5
                norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
                labels = ["%.1f dB" % power for power in self.readout_powers]
                labels.append(0)
                cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), ticks = ticks)
                cbar.set_ticklabels(labels)
                ax.set_title("%.3f MHz, %.1f W" %(self.drones.iloc[j, i].resonators[k].f0/1e6, self.optical_powers[i]))
                ax.set_xlabel("f [$Hz$]")
                #ax.legend()
                ax.set_ylabel("S21 PSD")
                plt.savefig(self.sweep_dir + "fig/"+ dir_name + "/%.3f_MHz_%.1f_K_off_res_psd.png" %(self.drones.iloc[j, i].resonators[k].f0/1e6, self.optical_powers[i]))
                
                    
