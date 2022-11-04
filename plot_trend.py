import numpy as np
import matplotlib.pyplot as plt 

# =============================================================================
# functions
# =============================================================================
def extract_temperature(fn):
    idx = fn.find('_T')
    temperature = float(fn[idx+2:idx+9])
    return temperature
  
# =============================================================================
# Main
# =============================================================================
data_list = []
data = {
  "fn": "Bar3s2_thermal_CW03-58_160nm_vac_chp2_15283.7s_T159.169C_th0.150_x-0.300_y0.000_20.00s_010300_saxs.png",
  "a_nm": 12.5,
  "b_nm": 8.9,
  "alpha": 90,
  "beta":  90,
  "gamma": 111,
  #"fit": [52141, 234807, 0.2221]
  #"fit_all": [149328, 2626354, 0.0569],
}
data_list.append(data)

data = {
  "fn": "Bar3s2_thermal_CW03-58_160nm_vac_chp2_15914.3s_T164.458C_th0.150_x-0.200_y0.000_20.00s_010441_saxs.png",
  "a_nm": 11.8,
  "b_nm": 8.9,
  "alpha": 90,
  "beta":  90,
  "gamma": 111,
  #"fit": [50235, 227377, 0.2209],
  # "fit_all": [140486, 2549325, 0.0551]
}
data_list.append(data)

data = {
  "fn": "Bar3s2_thermal_CW03-58_160nm_vac_chp2_16723.6s_T171.150C_th0.150_x-0.000_y0.000_20.00s_010469_saxs.png",
  "a_nm": 11.4,
  "b_nm": 8.6,
  "alpha": 90,
  "beta":  90,
  "gamma": 113.5,
}
data_list.append(data)

data = {
  "fn": "Bar3_offline/Index/Bar3s2_thermal_CW03-58_160nm_vac_chp2_17532.9s_T177.948C_th0.150_x0.150_y0.000_20.00s_010606_saxs.png",
  "a_nm": 10.9,
  "b_nm": 8.45,
  "alpha": 90,
  "beta":  90,
  "gamma": 114,
  # 43212, 204523, 0.2113
  # 109542, 2296006, 0.0477
}
data_list.append(data)

data = {
  "fn": "Bar3_offline/Index/Bar3s2_thermal_CW03-58_160nm_vac_chp2_18831.3s_T188.714C_th0.150_x0.400_y0.000_20.00s_010758_saxs.png",
  "a_nm": 10.3,
  "b_nm": 8.38,
  "alpha": 90,
  "beta":  90,
  "gamma": 116.5,
}
data_list.append(data)


data = {
  "fn": "Bar3s2_thermal_CW03-58_160nm_vac_chp2_19688.8s_T195.894C_th0.150_x0.600_y0.000_20.00s_010786_saxs.png",
  "a_nm": 10.0,
  "b_nm": 8.31,
  "alpha": 90,
  "beta":  90,
  "gamma": 116.5,
  # 34810, 165732, 0.2100
  # 78916, 1868215, 0.0422
}
data_list.append(data)

data = {
  "fn": "Bar3s2_thermal_CW03-58_160nm_vac_chp2_20713.9s_T204.430C_th0.150_x0.800_y0.000_20.00s_010935_saxs.png",
  "a_nm": 9.65,
  "b_nm": 8.3,
  "alpha": 90,
  "beta":  90,
  "gamma": 117,
  # 31058, 150363, 0.2066
  # 67576, 1696752, 0.0398
}
data_list.append(data)


temp_array = []
a_array = []
b_array = []
gamma_array = []
for data in data_list:
    temp_array.append(extract_temperature(data['fn']))
    a_array.append(data['a_nm'])
    b_array.append(data['b_nm'])
    gamma_array.append(data['gamma'])

plt.figure(10); plt.clf()
plt.subplot(211)
plt.plot(temp_array, a_array, 'o-', label='a') 
plt.plot(temp_array, b_array, 'o:', label='b')
plt.legend()
plt.grid()
fn = data['fn']
idx = fn.find('s_')
st = "{}".format(fn[0:idx-8])
plt.title(st)

plt.subplot(212)
plt.plot(temp_array, gamma_array, 'rx-', label='gamma')
plt.legend()
plt.grid()
plt.xlabel('temperature (degC)',fontsize=10)


