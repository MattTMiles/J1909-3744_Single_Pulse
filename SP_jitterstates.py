import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import subprocess as sproc 
import glob
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker


main_dir = '/fred/oz002/users/mmiles/SinglePulse'

toasim1024_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft1024/toasim_all'
toasim1024_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft1024/toasim_strong'
toasim1024_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft1024/toasim_weak'

toasim512_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft512/toasim_all'
toasim512_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft512/toasim_strong'
toasim512_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft512/toasim_weak'

toasim256_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft256/toasim_all'
toasim256_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft256/toasim_strong'
toasim256_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft256/toasim_weak'

toasim128_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft128/toasim_all'
toasim128_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft128/toasim_strong'
toasim128_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft128/toasim_weak'

toasim64_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft64/toasim_all'
toasim64_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft64/toasim_strong'
toasim64_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft64/toasim_weak'

toasim32_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft32/toasim_all'
toasim32_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft32/toasim_strong'
toasim32_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft32/toasim_weak'

toasim16_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft16/toasim_all'
toasim16_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft16/toasim_strong'
toasim16_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft16/toasim_weak'

toasim8_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft8/toasim_all'
toasim8_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft8/toasim_strong'
toasim8_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft8/toasim_weak'

toasim4_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft4/toasim_all'
toasim4_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft4/toasim_strong'
toasim4_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft4/toasim_weak'

toasim2_all = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft2/toasim_all'
toasim2_strong = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft2/toasim_strong'
toasim2_weak = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/Ft2/toasim_weak'

#Long observation data
msp_2 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft2'
msp_4 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft4'
msp_8 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft8'
msp_16 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft16'
msp_32 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft32'
msp_64 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft64'
msp_128 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft128'
msp_256 ='/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/tim_sims/long_obs/Ft256'


#Ft2048 data
#Tsub2048

#Ft1024 data
Tsub1024 = 3.0179

alldat1024 = np.loadtxt(toasim1024_all+'/avg.dat')

#alldat1024_sim = np.loadtxt(toasim1024_all+'/sim_00/avg.dat')

strongdat1024 = np.loadtxt(toasim1024_strong+'/avg.dat')
#strongdat1024_sim = np.loadtxt(toasim1024_strong+'/sim_00/avg.dat')

#weakdat1024 = np.loadtxt(toasim1024_weak+'/avg.dat')
#weakdat1024_sim = np.loadtxt(toasim1024_weak+'/sim_00/avg.dat')

var_alldat1024 = np.var(alldat1024[:,1])
err_alldat1024 = np.std(alldat1024[:,1])/np.sqrt(len(alldat1024[:,1]))

i=0
vars1024all = []
for all1024 in os.listdir(toasim1024_all):
    if all1024.startswith("sim"):
        os.chdir(toasim1024_all+'/'+all1024)
        alldat1024_sim = np.loadtxt('avg.dat')
        temp_alldat1024_sim = np.var(alldat1024_sim[:,1])
        vars1024all.append(temp_alldat1024_sim)
        i=i+1
        print(f'all1024: {i}')

var_alldat1024_sim = np.average(vars1024all)
#var_alldat1024_sim = np.var(alldat1024_sim[:,1])

var_strongdat1024 = np.var(strongdat1024[:,1])
err_strongdat1024 = np.std(strongdat1024[:,1])/np.sqrt(len(strongdat1024[:,1]))

i=0
vars1024strong = []
for strong1024 in os.listdir(toasim1024_strong):
    if strong1024.startswith("sim"):
        os.chdir(toasim1024_strong+'/'+strong1024)
        strongdat1024_sim = np.loadtxt('avg.dat')
        temp_strongdat1024_sim = np.var(strongdat1024_sim[:,1])
        vars1024strong.append(temp_strongdat1024_sim)
        i=i+1
        print(f'strong1024: {i}')

var_strongdat1024_sim = np.average(vars1024strong)

#var_strongdat1024_sim = np.var(strongdat1024_sim[:,1])
'''
var_weakdat1024 = np.var(weakdat1024[:,1])
err_weakdat1024 = np.std(weakdat1024[:,1])/np.sqrt(len(weakdat1024[:,1]))

vars1024weak = []
for weak1024 in os.listdir(toasim1024_weak):
    if weak1024.startswith("sim"):
        os.chdir(toasim1024_weak+'/'+weak1024)
        weakdat1024_sim = np.loadtxt('avg.dat')
        temp_weakdat1024_sim = np.var(weakdat1024_sim[:,1])
        vars1024weak.append(temp_weakdat1024_sim)

var_weakdat1024_sim = np.average(vars1024weak)
#var_weakdat1024_sim = np.var(weakdat1024_sim[:,1])
'''

stage_all1024 = np.sqrt(var_alldat1024 - vars1024all)
err_all1024 = np.std(stage_all1024)/np.sqrt(len(stage_all1024))
jitter_all1024 = np.average(stage_all1024)

stage_strong1024 = np.sqrt(var_strongdat1024 - vars1024strong)
err_strong1024 = np.std(stage_strong1024)/np.sqrt(len(stage_strong1024))
jitter_strong1024 = np.average(stage_strong1024)
#jitter_weak1024 = np.sqrt(var_weakdat1024 - var_weakdat1024_sim)

jall_imp1024 = jitter_all1024/(np.sqrt(3600/Tsub1024))
errall_imp1024 = err_all1024/(np.sqrt(3600/Tsub1024))
jstrong_imp1024 = jitter_strong1024/(np.sqrt(3600/Tsub1024))
errstrong_imp1024 = err_strong1024/(np.sqrt(3600/Tsub1024))

#jweak_imp1024 = jitter_weak1024/(np.sqrt(3600/Tsub1024))

#Ft512 data
Tsub512 = 1.509

alldat512 = np.loadtxt(toasim512_all+'/avg.dat')

#alldat512_sim = np.loadtxt(toasim512_all+'/sim_00/avg.dat')

strongdat512 = np.loadtxt(toasim512_strong+'/avg.dat')
#strongdat512_sim = np.loadtxt(toasim512_strong+'/sim_00/avg.dat')

#weakdat512 = np.loadtxt(toasim512_weak+'/avg.dat')
#weakdat512_sim = np.loadtxt(toasim512_weak+'/sim_00/avg.dat')

var_alldat512 = np.var(alldat512[:,1])
err_alldat512 = np.std(alldat512[:,1])/np.sqrt(len(alldat512[:,1]))
i=0
vars512all = []
for all512 in os.listdir(toasim512_all):
    if all512.startswith("sim"):
        os.chdir(toasim512_all+'/'+all512)
        alldat512_sim = np.loadtxt('avg.dat')
        temp_alldat512_sim = np.var(alldat512_sim[:,1])
        vars512all.append(temp_alldat512_sim)
        i=i+1
        print(f'all512: {i}')

var_alldat512_sim = np.average(vars512all)
#var_alldat512_sim = np.var(alldat512_sim[:,1])

var_strongdat512 = np.var(strongdat512[:,1])
err_strongdat512 = np.std(strongdat512[:,1])/np.sqrt(len(strongdat512[:,1]))

i=0
vars512strong = []
for strong512 in os.listdir(toasim512_strong):
    if strong512.startswith("sim"):
        os.chdir(toasim512_strong+'/'+strong512)
        strongdat512_sim = np.loadtxt('avg.dat')
        temp_strongdat512_sim = np.var(strongdat512_sim[:,1])
        vars512strong.append(temp_strongdat512_sim)
        i=i+1
        print(f'strong1024: {i}')

var_strongdat512_sim = np.average(vars512strong)

#var_strongdat512_sim = np.var(strongdat512_sim[:,1])
'''
var_weakdat512 = np.var(weakdat512[:,1])
err_weakdat512 = np.std(weakdat512[:,1])/np.sqrt(len(weakdat512[:,1]))

vars512weak = []
for weak512 in os.listdir(toasim512_weak):
    if weak512.startswith("sim"):
        os.chdir(toasim512_weak+'/'+weak512)
        weakdat512_sim = np.loadtxt('avg.dat')
        temp_weakdat512_sim = np.var(weakdat512_sim[:,1])
        vars512weak.append(temp_weakdat512_sim)

var_weakdat512_sim = np.average(vars512weak)
#var_weakdat512_sim = np.var(weakdat512_sim[:,1])
'''

stage_all512 = np.sqrt(var_alldat512 - vars512all)
err_all512 = np.std(stage_all512)/np.sqrt(len(stage_all512))
jitter_all512 = np.average(stage_all512)

stage_strong512 = np.sqrt(var_strongdat512 - vars512strong)
err_strong512 = np.std(stage_strong512)/np.sqrt(len(stage_strong512))
jitter_strong512 = np.average(stage_strong512)
#jitter_weak512 = np.sqrt(var_weakdat512 - var_weakdat512_sim)

jall_imp512 = jitter_all512/(np.sqrt(3600/Tsub512))
errall_imp512 = err_all512/(np.sqrt(3600/Tsub512))
jstrong_imp512 = jitter_strong512/(np.sqrt(3600/Tsub512))
errstrong_imp512 = err_strong512/(np.sqrt(3600/Tsub512))

#jweak_imp512 = jitter_weak512/(np.sqrt(3600/Tsub512))

#Ft256 data
Tsub256 = 0.7545

alldat256 = np.loadtxt(toasim256_all+'/avg.dat')

#alldat256_sim = np.loadtxt(toasim256_all+'/sim_00/avg.dat')

strongdat256 = np.loadtxt(toasim256_strong+'/avg.dat')
#strongdat256_sim = np.loadtxt(toasim256_strong+'/sim_00/avg.dat')

#weakdat256 = np.loadtxt(toasim256_weak+'/avg.dat')
#weakdat256_sim = np.loadtxt(toasim256_weak+'/sim_00/avg.dat')

var_alldat256 = np.var(alldat256[:,1])
err_alldat256 = np.std(alldat256[:,1])/np.sqrt(len(alldat256[:,1]))

i=0
vars256all = []
for all256 in os.listdir(toasim256_all):
    if all256.startswith("sim"):
        os.chdir(toasim256_all+'/'+all256)
        alldat256_sim = np.loadtxt('avg.dat')
        temp_alldat256_sim = np.var(alldat256_sim[:,1])
        vars256all.append(temp_alldat256_sim)
        i=i+1
        print(f'all256: {i}')

var_alldat256_sim = np.average(vars256all)
#var_alldat256_sim = np.var(alldat256_sim[:,1])

var_strongdat256 = np.var(strongdat256[:,1])
err_strongdat256 = np.std(strongdat256[:,1])/np.sqrt(len(strongdat256[:,1]))

i=0
vars256strong = []
for strong256 in os.listdir(toasim256_strong):
    if strong256.startswith("sim"):
        os.chdir(toasim256_strong+'/'+strong256)
        strongdat256_sim = np.loadtxt('avg.dat')
        temp_strongdat256_sim = np.var(strongdat256_sim[:,1])
        vars256strong.append(temp_strongdat256_sim)
        i=i+1
        print(f'strong256: {i}')

var_strongdat256_sim = np.average(vars256strong)

#var_strongdat256_sim = np.var(strongdat256_sim[:,1])
'''
var_weakdat256 = np.var(weakdat256[:,1])
err_weakdat256 = np.std(weakdat256[:,1])/np.sqrt(len(weakdat256[:,1]))

vars256weak = []
for weak256 in os.listdir(toasim256_weak):
    if weak256.startswith("sim"):
        os.chdir(toasim256_weak+'/'+weak256)
        weakdat256_sim = np.loadtxt('avg.dat')
        temp_weakdat256_sim = np.var(weakdat256_sim[:,1])
        vars256weak.append(temp_weakdat256_sim)

var_weakdat256_sim = np.average(vars256weak)
#var_weakdat256_sim = np.var(weakdat256_sim[:,1])
'''

stage_all256 = np.sqrt(var_alldat256 - vars256all)
err_all256 = np.std(stage_all256)/np.sqrt(len(stage_all256))
jitter_all256 = np.average(stage_all256)

stage_strong256 = np.sqrt(var_strongdat256 - vars256strong)
err_strong256 = np.std(stage_strong256)/np.sqrt(len(stage_strong256))
jitter_strong256 = np.average(stage_strong256)
#jitter_weak256 = np.sqrt(var_weakdat256 - var_weakdat256_sim)

jall_imp256 = jitter_all256/(np.sqrt(3600/Tsub256))
errall_imp256 = err_all256/(np.sqrt(3600/Tsub256))
jstrong_imp256 = jitter_strong256/(np.sqrt(3600/Tsub256))
errstrong_imp256 = err_strong256/(np.sqrt(3600/Tsub256))

#jweak_imp256 = jitter_weak256/(np.sqrt(3600/Tsub256))

#Ft128 data
Tsub128 = 0.377

alldat128 = np.loadtxt(toasim128_all+'/avg.dat')

#alldat128_sim = np.loadtxt(toasim128_all+'/sim_00/avg.dat')

strongdat128 = np.loadtxt(toasim128_strong+'/avg.dat')
#strongdat128_sim = np.loadtxt(toasim128_strong+'/sim_00/avg.dat')

#weakdat128 = np.loadtxt(toasim128_weak+'/avg.dat')
#weakdat128_sim = np.loadtxt(toasim128_weak+'/sim_00/avg.dat')

var_alldat128 = np.var(alldat128[:,1])
err_alldat128 = np.std(alldat128[:,1])/np.sqrt(len(alldat128[:,1]))
i=0
vars128all = []
for all128 in os.listdir(toasim128_all):
    if all128.startswith("sim"):
        os.chdir(toasim128_all+'/'+all128)
        alldat128_sim = np.loadtxt('avg.dat')
        temp_alldat128_sim = np.var(alldat128_sim[:,1])
        vars128all.append(temp_alldat128_sim)
        i=i+1
        print(f'all128: {i}')

var_alldat128_sim = np.average(vars128all)
#var_alldat128_sim = np.var(alldat128_sim[:,1])

var_strongdat128 = np.var(strongdat128[:,1])
err_strongdat128 = np.std(strongdat128[:,1])/np.sqrt(len(strongdat128[:,1]))

i=0
vars128strong = []
for strong128 in os.listdir(toasim128_strong):
    if strong128.startswith("sim"):
        os.chdir(toasim128_strong+'/'+strong128)
        strongdat128_sim = np.loadtxt('avg.dat')
        temp_strongdat128_sim = np.var(strongdat128_sim[:,1])
        vars128strong.append(temp_strongdat128_sim)
        i=i+1
        print(f'strong128: {i}')

var_strongdat128_sim = np.average(vars128strong)

#var_strongdat128_sim = np.var(strongdat128_sim[:,1])
'''
var_weakdat128 = np.var(weakdat128[:,1])
err_weakdat128 = np.std(weakdat128[:,1])/np.sqrt(len(weakdat128[:,1]))

vars128weak = []
for weak128 in os.listdir(toasim128_weak):
    if weak128.startswith("sim"):
        os.chdir(toasim128_weak+'/'+weak128)
        weakdat128_sim = np.loadtxt('avg.dat')
        temp_weakdat128_sim = np.var(weakdat128_sim[:,1])
        vars128weak.append(temp_weakdat128_sim)

var_weakdat128_sim = np.average(vars128weak)
#var_weakdat128_sim = np.var(weakdat128_sim[:,1])
'''

stage_all128 = np.sqrt(var_alldat128 - vars128all)
err_all128 = np.std(stage_all128)/np.sqrt(len(stage_all128))
jitter_all128 = np.average(stage_all128)

stage_strong128 = np.sqrt(var_strongdat128 - vars128strong)
err_strong128 = np.std(stage_strong128)/np.sqrt(len(stage_strong128))
jitter_strong128 = np.average(stage_strong128)
#jitter_weak128 = np.sqrt(var_weakdat128 - var_weakdat128_sim)

jall_imp128 = jitter_all128/(np.sqrt(3600/Tsub128))
errall_imp128 = err_all128/(np.sqrt(3600/Tsub128))
jstrong_imp128 = jitter_strong128/(np.sqrt(3600/Tsub128))
errstrong_imp128 = err_strong128/(np.sqrt(3600/Tsub128))

#jweak_imp128 = jitter_weak128/(np.sqrt(3600/Tsub128))
#Ft64 data
Tsub64 = 0.188625

alldat64 = np.loadtxt(toasim64_all+'/avg.dat')

#alldat64_sim = np.loadtxt(toasim64_all+'/sim_00/avg.dat')

strongdat64 = np.loadtxt(toasim64_strong+'/avg.dat')
#strongdat64_sim = np.loadtxt(toasim64_strong+'/sim_00/avg.dat')

#weakdat64 = np.loadtxt(toasim64_weak+'/avg.dat')
#weakdat64_sim = np.loadtxt(toasim64_weak+'/sim_00/avg.dat')

var_alldat64 = np.var(alldat64[:,1])
err_alldat64 = np.std(alldat64[:,1])/np.sqrt(len(alldat64[:,1]))

i=0
vars64all = []
for all64 in os.listdir(toasim64_all):
    if all64.startswith("sim"):
        os.chdir(toasim64_all+'/'+all64)
        alldat64_sim = np.loadtxt('avg.dat')
        temp_alldat64_sim = np.var(alldat64_sim[:,1])
        vars64all.append(temp_alldat64_sim)
        i=i+1
        print(f'all64: {i}')

var_alldat64_sim = np.average(vars64all)
#var_alldat64_sim = np.var(alldat64_sim[:,1])

var_strongdat64 = np.var(strongdat64[:,1])
err_strongdat64 = np.std(strongdat64[:,1])/np.sqrt(len(strongdat64[:,1]))

i=0
vars64strong = []
for strong64 in os.listdir(toasim64_strong):
    if strong64.startswith("sim"):
        os.chdir(toasim64_strong+'/'+strong64)
        strongdat64_sim = np.loadtxt('avg.dat')
        temp_strongdat64_sim = np.var(strongdat64_sim[:,1])
        vars64strong.append(temp_strongdat64_sim)
        i=i+1
        print(f'strong64: {i}')

var_strongdat64_sim = np.average(vars64strong)

#var_strongdat64_sim = np.var(strongdat64_sim[:,1])
'''
var_weakdat64 = np.var(weakdat64[:,1])
err_weakdat64 = np.std(weakdat64[:,1])/np.sqrt(len(weakdat64[:,1]))

vars64weak = []
for weak64 in os.listdir(toasim64_weak):
    if weak64.startswith("sim"):
        os.chdir(toasim64_weak+'/'+weak64)
        weakdat64_sim = np.loadtxt('avg.dat')
        temp_weakdat64_sim = np.var(weakdat64_sim[:,1])
        vars64weak.append(temp_weakdat64_sim)

var_weakdat64_sim = np.average(vars64weak)
#var_weakdat64_sim = np.var(weakdat64_sim[:,1])
'''
stage_all64 = np.sqrt(var_alldat64 - vars64all)
err_all64 = np.std(stage_all64)/np.sqrt(len(stage_all64))
jitter_all64 = np.average(stage_all64)

stage_strong64 = np.sqrt(var_strongdat64 - vars64strong)
err_strong64 = np.std(stage_strong64)/np.sqrt(len(stage_strong64))
jitter_strong64 = np.average(stage_strong64)
#jitter_weak64 = np.sqrt(var_weakdat64 - var_weakdat64_sim)

jall_imp64 = jitter_all64/(np.sqrt(3600/Tsub64))
errall_imp64 = err_all64/(np.sqrt(3600/Tsub64))
jstrong_imp64 = jitter_strong64/(np.sqrt(3600/Tsub64))
errstrong_imp64 = err_strong64/(np.sqrt(3600/Tsub64))

#jweak_imp64 = jitter_weak64/(np.sqrt(3600/Tsub64))


#Ft32 data
Tsub32 = 0.0943125

alldat32 = np.loadtxt(toasim32_all+'/avg.dat')

#alldat32_sim = np.loadtxt(toasim32_all+'/sim_00/avg.dat')

strongdat32 = np.loadtxt(toasim32_strong+'/avg.dat')
#strongdat32_sim = np.loadtxt(toasim32_strong+'/sim_00/avg.dat')

#weakdat32 = np.loadtxt(toasim32_weak+'/avg.dat')
#weakdat32_sim = np.loadtxt(toasim32_weak+'/sim_00/avg.dat')

var_alldat32 = np.var(alldat32[:,1])
err_alldat32 = np.std(alldat32[:,1])/np.sqrt(len(alldat32[:,1]))

i=0
vars32all = []
for all32 in os.listdir(toasim32_all):
    if all32.startswith("sim"):
        os.chdir(toasim32_all+'/'+all32)
        alldat32_sim = np.loadtxt('avg.dat')
        temp_alldat32_sim = np.var(alldat32_sim[:,1])
        vars32all.append(temp_alldat32_sim)
        i=i+1
        print(f'all32: {i}')

var_alldat32_sim = np.average(vars32all)
#var_alldat32_sim = np.var(alldat32_sim[:,1])

var_strongdat32 = np.var(strongdat32[:,1])
err_strongdat32 = np.std(strongdat32[:,1])/np.sqrt(len(strongdat32[:,1]))

i=0
vars32strong = []
for strong32 in os.listdir(toasim32_strong):
    if strong32.startswith("sim"):
        os.chdir(toasim32_strong+'/'+strong32)
        strongdat32_sim = np.loadtxt('avg.dat')
        temp_strongdat32_sim = np.var(strongdat32_sim[:,1])
        vars32strong.append(temp_strongdat32_sim)
        i=i+1
        print(f'strong32: {i}')

var_strongdat32_sim = np.average(vars32strong)

#var_strongdat32_sim = np.var(strongdat32_sim[:,1])
'''
var_weakdat32 = np.var(weakdat32[:,1])
err_weakdat32 = np.std(weakdat32[:,1])/np.sqrt(len(weakdat32[:,1]))

vars32weak = []
for weak32 in os.listdir(toasim32_weak):
    if weak32.startswith("sim"):
        os.chdir(toasim32_weak+'/'+weak32)
        weakdat32_sim = np.loadtxt('avg.dat')
        temp_weakdat32_sim = np.var(weakdat32_sim[:,1])
        vars32weak.append(temp_weakdat32_sim)

var_weakdat32_sim = np.average(vars32weak)
#var_weakdat32_sim = np.var(weakdat32_sim[:,1])
'''


stage_all32 = np.sqrt(var_alldat32 - vars32all)
err_all32 = np.std(stage_all32)/np.sqrt(len(stage_all32))
jitter_all32 = np.average(stage_all32)

stage_strong32 = np.sqrt(var_strongdat32 - vars32strong)
err_strong32 = np.std(stage_strong32)/np.sqrt(len(stage_strong32))
jitter_strong32 = np.average(stage_strong32)
#jitter_weak32 = np.sqrt(var_weakdat32 - var_weakdat32_sim)

jall_imp32 = jitter_all32/(np.sqrt(3600/Tsub32))
errall_imp32 = err_all32/(np.sqrt(3600/Tsub32))
jstrong_imp32 = jitter_strong32/(np.sqrt(3600/Tsub32))
errstrong_imp32 = err_strong32/(np.sqrt(3600/Tsub32))

#jweak_imp32 = jitter_weak32/(np.sqrt(3600/Tsub32))


#Ft16 data
Tsub16 = 0.04715625

alldat16 = np.loadtxt(toasim16_all+'/avg.dat')

#alldat16_sim = np.loadtxt(toasim16_all+'/sim_00/avg.dat')

strongdat16 = np.loadtxt(toasim16_strong+'/avg.dat')
#strongdat16_sim = np.loadtxt(toasim16_strong+'/sim_00/avg.dat')

#weakdat16 = np.loadtxt(toasim16_weak+'/avg.dat')
#weakdat16_sim = np.loadtxt(toasim16_weak+'/sim_00/avg.dat')

var_alldat16 = np.var(alldat16[:,1])
err_alldat16 = np.std(alldat16[:,1])/np.sqrt(len(alldat16[:,1]))

i=0
vars16all = []
for all16 in os.listdir(toasim16_all):
    if all16.startswith("sim"):
        os.chdir(toasim16_all+'/'+all16)
        alldat16_sim = np.loadtxt('avg.dat')
        temp_alldat16_sim = np.var(alldat16_sim[:,1])
        vars16all.append(temp_alldat16_sim)
        i=i+1
        print(f'all16: {i}')

var_alldat16_sim = np.average(vars16all)
#var_alldat16_sim = np.var(alldat16_sim[:,1])

var_strongdat16 = np.var(strongdat16[:,1])
err_strongdat16 = np.std(strongdat16[:,1])/np.sqrt(len(strongdat16[:,1]))

i=0
vars16strong = []
for strong16 in os.listdir(toasim16_strong):
    if strong16.startswith("sim"):
        os.chdir(toasim16_strong+'/'+strong16)
        strongdat16_sim = np.loadtxt('avg.dat')
        temp_strongdat16_sim = np.var(strongdat16_sim[:,1])
        vars16strong.append(temp_strongdat16_sim)
        i=i+1
        print(f'strong16: {i}')

var_strongdat16_sim = np.average(vars16strong)

#var_strongdat16_sim = np.var(strongdat16_sim[:,1])
'''
var_weakdat16 = np.var(weakdat16[:,1])
err_weakdat16 = np.std(weakdat16[:,1])/np.sqrt(len(weakdat16[:,1]))

vars16weak = []
for weak16 in os.listdir(toasim16_weak):
    if weak16.startswith("sim"):
        os.chdir(toasim16_weak+'/'+weak16)
        weakdat16_sim = np.loadtxt('avg.dat')
        temp_weakdat16_sim = np.var(weakdat16_sim[:,1])
        vars16weak.append(temp_weakdat16_sim)

var_weakdat16_sim = np.average(vars16weak)
#var_weakdat16_sim = np.var(weakdat16_sim[:,1])
'''


stage_all16 = np.sqrt(var_alldat16 - vars16all)
err_all16 = np.std(stage_all16)/np.sqrt(len(stage_all16))
jitter_all16 = np.average(stage_all16)

stage_strong16 = np.sqrt(var_strongdat16 - vars16strong)
err_strong16 = np.std(stage_strong16)/np.sqrt(len(stage_strong16))
jitter_strong16 = np.average(stage_strong16)
#jitter_weak16 = np.sqrt(var_weakdat16 - var_weakdat16_sim)

jall_imp16 = jitter_all16/(np.sqrt(3600/Tsub16))
errall_imp16 = err_all16/(np.sqrt(3600/Tsub16))
jstrong_imp16 = jitter_strong16/(np.sqrt(3600/Tsub16))
errstrong_imp16 = err_strong16/(np.sqrt(3600/Tsub16))

#jweak_imp16 = jitter_weak16/(np.sqrt(3600/Tsub16))


#Ft8 data
Tsub8 = 0.023574125

alldat8 = np.loadtxt(toasim8_all+'/avg.dat')

#alldat8_sim = np.loadtxt(toasim8_all+'/sim_00/avg.dat')

strongdat8 = np.loadtxt(toasim8_strong+'/avg.dat')
#strongdat8_sim = np.loadtxt(toasim8_strong+'/sim_00/avg.dat')

#weakdat8 = np.loadtxt(toasim8_weak+'/avg.dat')
#weakdat8_sim = np.loadtxt(toasim8_weak+'/sim_00/avg.dat')

var_alldat8 = np.var(alldat8[:,1])
err_alldat8= np.std(alldat8[:,1])/np.sqrt(len(alldat8[:,1]))

i=0
vars8all = []
for all8 in os.listdir(toasim8_all):
    if all8.startswith("sim"):
        os.chdir(toasim8_all+'/'+all8)
        alldat8_sim = np.loadtxt('avg.dat')
        temp_alldat8_sim = np.var(alldat8_sim[:,1])
        vars8all.append(temp_alldat8_sim)
        i=i+1
        print(f'all8: {i}')

var_alldat8_sim = np.average(vars8all)
#var_alldat8_sim = np.var(alldat8_sim[:,1])

var_strongdat8 = np.var(strongdat8[:,1])
err_strongdat8= np.std(strongdat8[:,1])/np.sqrt(len(strongdat8[:,1]))

i=0
vars8strong = []
for strong8 in os.listdir(toasim8_strong):
    if strong8.startswith("sim"):
        os.chdir(toasim8_strong+'/'+strong8)
        strongdat8_sim = np.loadtxt('avg.dat')
        temp_strongdat8_sim = np.var(strongdat8_sim[:,1])
        vars8strong.append(temp_strongdat8_sim)
        i=i+1
        print(f'strong8: {i}')

var_strongdat8_sim = np.average(vars8strong)

#var_strongdat8_sim = np.var(strongdat8_sim[:,1])
'''
var_weakdat8 = np.var(weakdat8[:,1])
err_weakdat8= np.std(weakdat8[:,1])/np.sqrt(len(weakdat8[:,1]))

vars8weak = []
for weak8 in os.listdir(toasim8_weak):
    if weak8.startswith("sim"):
        os.chdir(toasim8_weak+'/'+weak8)
        weakdat8_sim = np.loadtxt('avg.dat')
        temp_weakdat8_sim = np.var(weakdat8_sim[:,1])
        vars8weak.append(temp_weakdat8_sim)

var_weakdat8_sim = np.average(vars8weak)
#var_weakdat8_sim = np.var(weakdat8_sim[:,1])
'''

stage_all8 = np.sqrt(var_alldat8 - vars8all)
err_all8 = np.std(stage_all8)/np.sqrt(len(stage_all8))
jitter_all8 = np.average(stage_all8)

stage_strong8 = np.sqrt(var_strongdat8 - vars8strong)
err_strong8 = np.std(stage_strong8)/np.sqrt(len(stage_strong8))
jitter_strong8 = np.average(stage_strong8)
#jitter_weak8 = np.sqrt(var_weakdat8 - var_weakdat8_sim)

jall_imp8 = jitter_all8/(np.sqrt(3600/Tsub8))
errall_imp8 = err_all8/(np.sqrt(3600/Tsub8))
jstrong_imp8 = jitter_strong8/(np.sqrt(3600/Tsub8))
errstrong_imp8 = err_strong8/(np.sqrt(3600/Tsub8))

#jweak_imp8 = jitter_weak8/(np.sqrt(3600/Tsub8))

#Ft4 data
Tsub4 = 0.0117890625

alldat4 = np.loadtxt(toasim4_all+'/avg.dat')

#alldat4_sim = np.loadtxt(toasim4_all+'/sim_00/avg.dat')

strongdat4 = np.loadtxt(toasim4_strong+'/avg.dat')
#strongdat4_sim = np.loadtxt(toasim4_strong+'/sim_00/avg.dat')

#weakdat4 = np.loadtxt(toasim4_weak+'/avg.dat')
#weakdat4_sim = np.loadtxt(toasim4_weak+'/sim_00/avg.dat')

var_alldat4 = np.var(alldat4[:,1])
err_alldat4= np.std(alldat4[:,1])/np.sqrt(len(alldat4[:,1]))

i=0
vars4all = []
for all4 in os.listdir(toasim4_all):
    if all4.startswith("sim"):
        os.chdir(toasim4_all+'/'+all4)
        alldat4_sim = np.loadtxt('avg.dat')
        temp_alldat4_sim = np.var(alldat4_sim[:,1])
        vars4all.append(temp_alldat4_sim)
        i=i+1
        print(f'all4: {i}')

var_alldat4_sim = np.average(vars4all)
#var_alldat4_sim = np.var(alldat4_sim[:,1])

var_strongdat4 = np.var(strongdat4[:,1])
err_strongdat4= np.std(strongdat4[:,1])/np.sqrt(len(strongdat4[:,1]))

i=0
vars4strong = []
for strong4 in os.listdir(toasim4_strong):
    if strong4.startswith("sim"):
        os.chdir(toasim4_strong+'/'+strong4)
        strongdat4_sim = np.loadtxt('avg.dat')
        temp_strongdat4_sim = np.var(strongdat4_sim[:,1])
        vars4strong.append(temp_strongdat4_sim)
        i=i+1
        print(f'strong4: {i}')

var_strongdat4_sim = np.average(vars4strong)

#var_strongdat4_sim = np.var(strongdat4_sim[:,1])
'''
var_weakdat4 = np.var(weakdat4[:,1])
err_weakdat4= np.std(weakdat4[:,1])/np.sqrt(len(weakdat4[:,1]))

vars4weak = []
for weak4 in os.listdir(toasim4_weak):
    if weak4.startswith("sim"):
        os.chdir(toasim4_weak+'/'+weak4)
        weakdat4_sim = np.loadtxt('avg.dat')
        temp_weakdat4_sim = np.var(weakdat4_sim[:,1])
        vars4weak.append(temp_weakdat4_sim)

var_weakdat4_sim = np.average(vars4weak)
#var_weakdat4_sim = np.var(weakdat4_sim[:,1])
'''

stage_all4 = np.sqrt(var_alldat4 - vars4all)
err_all4 = np.std(stage_all4)/np.sqrt(len(stage_all4))
jitter_all4 = np.average(stage_all4)

stage_strong4 = np.sqrt(var_strongdat4 - vars4strong)
err_strong4 = np.std(stage_strong4)/np.sqrt(len(stage_strong4))
jitter_strong4 = np.average(stage_strong4)
#jitter_weak4 = np.sqrt(var_weakdat4 - var_weakdat4_sim)

jall_imp4 = jitter_all4/(np.sqrt(3600/Tsub4))
errall_imp4 = err_all4/(np.sqrt(3600/Tsub4))
jstrong_imp4 = jitter_strong4/(np.sqrt(3600/Tsub4))
errstrong_imp4 = err_strong4/(np.sqrt(3600/Tsub4))

#jweak_imp4 = jitter_weak4/(np.sqrt(3600/Tsub4))

#Ft2 data
Tsub2 = 0.00589453125

alldat2 = np.loadtxt(toasim2_all+'/avg.dat')

#alldat2_sim = np.loadtxt(toasim2_all+'/sim_00/avg.dat')

strongdat2 = np.loadtxt(toasim2_strong+'/avg.dat')
#strongdat2_sim = np.loadtxt(toasim2_strong+'/sim_00/avg.dat')

#weakdat2 = np.loadtxt(toasim2_weak+'/avg.dat')
#weakdat2_sim = np.loadtxt(toasim2_weak+'/sim_00/avg.dat')

var_alldat2 = np.var(alldat2[:,1])
err_alldat2= np.std(alldat2[:,1])/np.sqrt(len(alldat2[:,1]))

i=0
vars2all = []
for all2 in os.listdir(toasim2_all):
    if all2.startswith("sim"):
        os.chdir(toasim2_all+'/'+all2)
        alldat2_sim = np.loadtxt('avg.dat')
        temp_alldat2_sim = np.var(alldat2_sim[:,1])
        vars2all.append(temp_alldat2_sim)
        i=i+1
        print(f'all2: {i}')

var_alldat2_sim = np.average(vars2all)
#var_alldat2_sim = np.var(alldat2_sim[:,1])

var_strongdat2 = np.var(strongdat2[:,1])
err_strongdat2= np.std(strongdat2[:,1])/np.sqrt(len(strongdat2[:,1]))

i=0
vars2strong = []
for strong2 in os.listdir(toasim2_strong):
    if strong2.startswith("sim"):
        os.chdir(toasim2_strong+'/'+strong2)
        strongdat2_sim = np.loadtxt('avg.dat')
        temp_strongdat2_sim = np.var(strongdat2_sim[:,1])
        vars2strong.append(temp_strongdat2_sim)
        i=i+1
        print(f'strong2: {i}')

var_strongdat2_sim = np.average(vars2strong)

#var_strongdat2_sim = np.var(strongdat2_sim[:,1])
'''
var_weakdat2 = np.var(weakdat2[:,1])
err_weakdat2= np.std(weakdat2[:,1])/np.sqrt(len(weakdat2[:,1]))

vars2weak = []
for weak2 in os.listdir(toasim2_weak):
    if weak2.startswith("sim"):
        os.chdir(toasim2_weak+'/'+weak2)
        weakdat2_sim = np.loadtxt('avg.dat')
        temp_weakdat2_sim = np.var(weakdat2_sim[:,1])
        vars2weak.append(temp_weakdat2_sim)

var_weakdat2_sim = np.average(vars2weak)
#var_weakdat2_sim = np.var(weakdat2_sim[:,1])
'''

stage_all2 = np.sqrt(var_alldat2 - vars2all)
err_all2 = np.std(stage_all2)/np.sqrt(len(stage_all2))
jitter_all2 = np.average(stage_all2)

stage_strong2 = np.sqrt(var_strongdat2 - vars2strong)
err_strong2 = np.std(stage_strong2)/np.sqrt(len(stage_strong2))
jitter_strong2 = np.average(stage_strong2)
#jitter_weak2 = np.sqrt(var_weakdat2 - var_weakdat2_sim)

jall_imp2 = jitter_all2/(np.sqrt(3600/Tsub2))
errall_imp2 = err_all2/(np.sqrt(3600/Tsub2))
jstrong_imp2 = jitter_strong2/(np.sqrt(3600/Tsub2))
errstrong_imp2 = err_strong2/(np.sqrt(3600/Tsub2))
#jweak_imp2 = jitter_weak2/(np.sqrt(3600/Tsub2))

#Long obs data
longtsub = 8
longobs = np.loadtxt(msp_128+'/avg.dat')
longobs_sim = np.loadtxt(msp_128+'/sim_00/avg.dat')

var_longobs = np.var(longobs[:,1])
var_longobs_sim = np.var(longobs_sim[:,1])

jitter_longobs = np.sqrt(var_longobs-var_longobs_sim)

jlongobs_imp = jitter_longobs/(np.sqrt(3600/longtsub))

os.chdir('/fred/oz002/users/mmiles/SinglePulse/snr_normal_window')
dataframe = []
labels = ['all1024','strong1024','all512','strong512','all256','strong256','all128','strong128','all64','strong64','all32','strong32','all16','strong16','all8','strong8','all4','strong4','all2','strong2']
data = [jall_imp1024,jstrong_imp1024,jall_imp512,jstrong_imp512,jall_imp256,jstrong_imp256,jall_imp128,jstrong_imp128,jall_imp64,jstrong_imp64,jall_imp32,jstrong_imp32,jall_imp16,jstrong_imp16,jall_imp8,jstrong_imp8,jall_imp4,jstrong_imp4,jall_imp2,jstrong_imp2]
dataframe.append(data)
df = pd.DataFrame(dataframe,columns=labels)
df.to_pickle("/fred/oz002/users/mmiles/SinglePulse/jitter_data.pkl")

#Graph showing Tsub vs Jitter value
xtsub =[Tsub1024,Tsub512,Tsub256,Tsub128,Tsub64,Tsub32,Tsub16,Tsub8,Tsub4,Tsub2]

jall = [jall_imp1024,jall_imp512,jall_imp256,jall_imp128,jall_imp64,jall_imp32,jall_imp16,jall_imp8,jall_imp4,jall_imp2]
jall_culled = [jall_imp1024,jall_imp512,jall_imp256,jall_imp128,jall_imp64,jall_imp32,jall_imp16,np.nan,np.nan,np.nan]

jstrong = [jstrong_imp1024,jstrong_imp512,jstrong_imp256,jstrong_imp128,jstrong_imp64,jstrong_imp32,jstrong_imp16,jstrong_imp8,jstrong_imp4,jstrong_imp2]
jstrong_culled = [jstrong_imp1024,jstrong_imp512,jstrong_imp256,jstrong_imp128,jstrong_imp64,jstrong_imp32,jstrong_imp16,jstrong_imp8,jstrong_imp4,np.nan]

os.chdir('/fred/oz002/users/mmiles/SinglePulse/')
##New jitter sample error
sampall1024 = jall_imp1024/np.sqrt(2*len(alldat1024[:,1]))
sampall512 = jall_imp512/np.sqrt(2*len(alldat512[:,1]))
sampall256 = jall_imp256/np.sqrt(2*len(alldat256[:,1]))
sampall128 = jall_imp128/np.sqrt(2*len(alldat128[:,1]))
sampall64 = jall_imp64/np.sqrt(2*len(alldat64[:,1]))
sampall32 = jall_imp32/np.sqrt(2*len(alldat32[:,1]))
sampall16 = jall_imp16/np.sqrt(2*len(alldat16[:,1]))
sampall8 = jall_imp1024/np.sqrt(2*len(alldat8[:,1]))
sampall4 = jall_imp1024/np.sqrt(2*len(alldat4[:,1]))

sampstrong1024 = jstrong_imp1024/np.sqrt(2*len(strongdat1024[:,1]))
sampstrong512 = jstrong_imp512/np.sqrt(2*len(strongdat512[:,1]))
sampstrong256 = jstrong_imp256/np.sqrt(2*len(strongdat256[:,1]))
sampstrong128 = jstrong_imp128/np.sqrt(2*len(strongdat128[:,1]))
sampstrong64 = jstrong_imp64/np.sqrt(2*len(strongdat64[:,1]))
sampstrong32 = jstrong_imp32/np.sqrt(2*len(strongdat32[:,1]))
sampstrong16 = jstrong_imp16/np.sqrt(2*len(strongdat16[:,1]))
sampstrong8 = jstrong_imp1024/np.sqrt(2*len(strongdat8[:,1]))
sampstrong4 = jstrong_imp1024/np.sqrt(2*len(strongdat4[:,1]))

sampstrongs = [sampstrong1024,sampstrong512,sampstrong256,sampstrong128,sampstrong64,sampstrong32,sampstrong16,sampstrong8,sampstrong4]
sampalls = [sampall1024,sampall512,sampall256,sampall128,sampall64,sampall32,sampall16,sampall8,sampall4]

np.save('sample_error_strong.npy',sampstrongs)
np.save('sample_error_all.npy',sampalls)

##radiometer error
radall1024 = np.mean(var_alldat1024_sim)/np.sqrt(2*len(alldat1024[:,1]))
radall512 = np.mean(var_alldat512_sim)/np.sqrt(2*len(alldat512[:,1]))
radall256 = np.mean(var_alldat256_sim)/np.sqrt(2*len(alldat256[:,1]))
radall128 = np.mean(var_alldat128_sim)/np.sqrt(2*len(alldat128[:,1]))
radall64 = np.mean(var_alldat64_sim)/np.sqrt(2*len(alldat64[:,1]))
radall32 = np.mean(var_alldat32_sim)/np.sqrt(2*len(alldat32[:,1]))
radall16 = np.mean(var_alldat16_sim)/np.sqrt(2*len(alldat16[:,1]))
radall8 = np.mean(var_alldat8_sim)/np.sqrt(2*len(alldat8[:,1]))
radall4 = np.mean(var_alldat4_sim)/np.sqrt(2*len(alldat4[:,1]))

radstrong1024 = np.mean(var_strongdat1024_sim)/np.sqrt(2*len(strongdat1024[:,1]))
radstrong512 = np.mean(var_strongdat512_sim)/np.sqrt(2*len(strongdat512[:,1]))
radstrong256 = np.mean(var_strongdat256_sim)/np.sqrt(2*len(strongdat256[:,1]))
radstrong128 = np.mean(var_strongdat128_sim)/np.sqrt(2*len(strongdat128[:,1]))
radstrong64 = np.mean(var_strongdat64_sim)/np.sqrt(2*len(strongdat64[:,1]))
radstrong32 = np.mean(var_strongdat32_sim)/np.sqrt(2*len(strongdat32[:,1]))
radstrong16 = np.mean(var_strongdat16_sim)/np.sqrt(2*len(strongdat16[:,1]))
radstrong8 = np.mean(var_strongdat8_sim)/np.sqrt(2*len(strongdat8[:,1]))
radstrong4 = np.mean(var_strongdat4_sim)/np.sqrt(2*len(strongdat4[:,1]))

radstrongs = [radstrong1024,radstrong512,radstrong256,radstrong128,radstrong64,radstrong32,radstrong16,radstrong8,radstrong4]
radalls = [radall1024,radall512,radall256,radall128,radall64,radall32,radall16,radall8,radall4]

np.save('radioerr_strongs.npy',radstrongs)
np.save('radioerr_alls.npy',radalls)

##actual jitter error

actjit_all1024 = np.sqrt((sampall1024**2)+(radall1024**2))
actjit_all512 = np.sqrt((sampall512**2)+(radall512**2))
actjit_all256 = np.sqrt((sampall256**2)+(radall256**2))
actjit_all128 = np.sqrt((sampall128**2)+(radall128**2))
actjit_all64 = np.sqrt((sampall64**2)+(radall64**2))
actjit_all32 = np.sqrt((sampall32**2)+(radall32**2))
actjit_all16 = np.sqrt((sampall16**2)+(radall16**2))
actjit_all8 = np.sqrt((sampall8**2)+(radall8**2))
actjit_all4 = np.sqrt((sampall4**2)+(radall4**2))

actjit_errsall = [actjit_all1024,actjit_all512,actjit_all256,actjit_all128,actjit_all64,actjit_all32,actjit_all16,actjit_all8,actjit_all4]

actjit_strong1024 = np.sqrt((sampstrong1024**2)+(radstrong1024**2))
actjit_strong512 = np.sqrt((sampstrong512**2)+(radstrong512**2))
actjit_strong256 = np.sqrt((sampstrong256**2)+(radstrong256**2))
actjit_strong128 = np.sqrt((sampstrong128**2)+(radstrong128**2))
actjit_strong64 = np.sqrt((sampstrong64**2)+(radstrong64**2))
actjit_strong32 = np.sqrt((sampstrong32**2)+(radstrong32**2))
actjit_strong16 = np.sqrt((sampstrong16**2)+(radstrong16**2))
actjit_strong8 = np.sqrt((sampstrong8**2)+(radstrong8**2))
actjit_strong4 = np.sqrt((sampstrong4**2)+(radstrong4**2))

actjit_errsstrong = [actjit_strong1024,actjit_strong512,actjit_strong256,actjit_strong128,actjit_strong64,actjit_strong32,actjit_strong16,actjit_strong8,actjit_strong4]

np.save('actualjiterr_all.npy',actjit_errsall)
np.save('actualjiterr_strong.npy',actjit_errsstrong)

'''
#jweak = [jweak_imp1024,jweak_imp512,jweak_imp256,jweak_imp128,jweak_imp64,jweak_imp32,jweak_imp16,jweak_imp8,jweak_imp4,jweak_imp2]
dataframe2 = []
labels2 = ['1024','512','256','128','64','32','16','8','4','2']
all_err = [errall_imp1024, errall_imp512, errall_imp256, errall_imp128, errall_imp64, errall_imp32, errall_imp16, errall_imp8, errall_imp4, errall_imp2]
strong_err = [errstrong_imp1024, errstrong_imp512, errstrong_imp256, errstrong_imp128, errstrong_imp64, errstrong_imp32, errstrong_imp16, errstrong_imp8, errstrong_imp4, errstrong_imp2]
dataframe2.append(all_err)
dataframe2.append(strong_err)
dferr = pd.DataFrame(dataframe2,columns=labels2)
dferr.to_pickle("/fred/oz002/users/mmiles/SinglePulse/jitter_error.pkl")


jdata = pd.read_pickle('jitter_data.pkl')
jdata = jdata[0,:]
jstrong = jdata[1::2]
jall = jdata[0::2]
jerror = pd.read_pickle('jitter_error.pkl')

strong_err = np.array(jerror.iloc[1])
all_err = np.array(jerror.iloc[0])


fig, ax = plt.subplots()

ax.set_xlabel('Subintegration time (s)')
ax.set_ylabel('Implied Jitter in 1hr (ns)')
#ax.set_title('Jitter noise vs integration time')
ax.set_xscale('log')
ax.set_yscale('log')

#ax.plot(xtsub,jall_culled,'o-', c='dimgray', linewidth=1,label='All pulses')
#ax.plot(xtsub,jstrong_culled,'o-', c='tab:blue', linewidth=1,label='Strong pulses')

ax.errorbar(xtsub,jall[:6]*10**9, yerr = all_err[:6]*10**9,c='dimgray', marker='o',label='All pulses')
ax.errorbar(xtsub,jstrong[:7]*10**9, yerr = strong_err[:7]*10**9, c='tab:blue', marker='o',label='Strong pulses')

#ax.plot(xtsub,jweak,'o-', c='tab:orange', linewidth=1,label='Weak pulses')
#Adityas values
#plt.axhline(9*10**-9, c='tab:purple', linewidth=1, linestyle='--', label='Previous jitter value')
ax.fill_between(xtsub,6,12, hatch='\\',facecolor='None',edgecolor='orchid',alpha=0.7, label='Parthasarathy et al. (2021)',zorder=-10)
ax.fill_between(xtsub,9.4,7.8, hatch='|',facecolor='None',edgecolor='lightskyblue',alpha=1, label='Shannon et al. (2014)',zorder=-10)
ax.fill_between(xtsub,13.5,14.5, hatch='//',facecolor='None',edgecolor='lightslategrey',alpha=0.7, label='Lam et al. (2019)',zorder=-10)
#ax.yaxis.set_major_locator(ticker.FixedLocator(1*10**-8))

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(10)
#Ryans values
#plt.axhline(8.6*10**-9, c='tab:purple', linewidth=1, linestyle='--', label='Previous jitter value')
#ax.fill_between(xtsub,7.8*10**-9,9.4*10**-9, color='tab:purple', alpha=0.3)
ax.legend(prop={'size':8})
ax.set_ylim(5*10**-9,1.5*10**-8)
ax.set_xlim(0.004,4)
ax.axvline(linewidth=2, x=0.004, color='black')
ax.axvline(linewidth=2, x=4, color='black')
ax.axhline(linewidth=2, y=5, color='black')
ax.axhline(linewidth=2, y=15, color='black')
#fig.tight_layout()
fig.savefig('/fred/oz002/users/mmiles/SinglePulse/paper_plots/jitter_comparison.pdf',dpi=1200)
fig.show()
'''



'''
# Quick jitter getter, run this in the tim_sims directory

Tsub128 = 0.377

simdir = os.getcwd()

strongdat128 = np.loadtxt('avg.dat')

var_strongdat128 = np.var(strongdat128[:,1])
err_strongdat128 = np.std(strongdat128[:,1])/np.sqrt(len(strongdat128[:,1]))

i=0
vars128strong = []
for strong128 in os.listdir(simdir):
    if strong128.startswith("sim"):
        os.chdir(simdir+'/'+strong128)
        strongdat128_sim = np.loadtxt('avg.dat')
        temp_strongdat128_sim = np.var(strongdat128_sim[:,1])
        vars128strong.append(temp_strongdat128_sim)
        i=i+1
        print(f'strong128: {i}')

var_strongdat128_sim = np.average(vars128strong)

stage_strong128 = np.sqrt(var_strongdat128 - vars128strong)
err_strong128 = np.std(stage_strong128)/np.sqrt(len(stage_strong128))
jitter_strong128 = np.average(stage_strong128)


jstrong_imp128 = jitter_strong128/(np.sqrt(3600/Tsub128))
errstrong_imp128 = err_strong128/(np.sqrt(3600/Tsub128))

#Actual Error
sampstrong128 = jstrong_imp128/np.sqrt(2*len(strongdat128[:,1]))
radstrong128 = np.mean(var_strongdat128_sim)/np.sqrt(2*len(strongdat128[:,1]))
actjit_strong128 = np.sqrt((sampstrong128**2)+(radstrong128**2))

print('jitter: {}'.format(jstrong_imp128))
print('error: {}'.format(actjit_strong128))

'''