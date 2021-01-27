import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import subprocess as sproc 

main_dir = '/fred/oz002/users/mmiles/SinglePulse'

toasim128_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft128/toasim_all'
toasim128_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft128/toasim_strong'
toasim128_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft128/toasim_weak'

toasim256_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft256/toasim_all'
toasim256_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft256/toasim_strong'
toasim256_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft256/toasim_weak'

toasim64_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft64/toasim_all'
toasim64_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft64/toasim_strong'
toasim64_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft64/toasim_weak'

toasim32_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft32/toasim_all'
toasim32_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft32/toasim_strong'
toasim32_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft32/toasim_weak'

toasim16_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft16/toasim_all'
toasim16_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft16/toasim_strong'
toasim16_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft16/toasim_weak'

toasim8_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft8/toasim_all'
toasim8_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft8/toasim_strong'
toasim8_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft8/toasim_weak'

toasim4_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft4/toasim_all'
toasim4_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft4/toasim_strong'
toasim4_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft4/toasim_weak'

toasim2_all = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft2/toasim_all'
toasim2_strong = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft2/toasim_strong'
toasim2_weak = '/fred/oz002/users/mmiles/SinglePulse/tim_sims/Ft2/toasim_weak'

#Long observation data
msp_2 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft2'
msp_4 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft4'
msp_8 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft8'
msp_16 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft16'
msp_32 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft32'
msp_64 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft64'
msp_128 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft128'
msp_256 ='/fred/oz002/users/mmiles/SinglePulse/tim_sims/long_obs/Ft256'

#Ft256 data
Tsub256 = 0.7545

alldat256 = np.loadtxt(toasim256_all+'/avg.dat')
alldat256_sim = np.loadtxt(toasim256_all+'/sim_00/avg.dat')

strongdat256 = np.loadtxt(toasim256_strong+'/avg.dat')
strongdat256_sim = np.loadtxt(toasim256_strong+'/sim_00/avg.dat')

weakdat256 = np.loadtxt(toasim256_weak+'/avg.dat')
weakdat256_sim = np.loadtxt(toasim256_weak+'/sim_00/avg.dat')

var_alldat256 = np.var(alldat256[:,1])
var_alldat256_sim = np.var(alldat256_sim[:,1])

var_strongdat256 = np.var(strongdat256[:,1])
var_strongdat256_sim = np.var(strongdat256_sim[:,1])

var_weakdat256 = np.var(weakdat256[:,1])
var_weakdat256_sim = np.var(weakdat256_sim[:,1])

jitter_all256 = np.sqrt(var_alldat256 - var_alldat256_sim)
jitter_strong256 = np.sqrt(var_strongdat256 - var_strongdat256_sim)
jitter_weak256 = np.sqrt(var_weakdat256 - var_weakdat256_sim)

jall_imp256 = jitter_all256/(np.sqrt(3600/Tsub256))
jstrong_imp256 = jitter_strong256/(np.sqrt(3600/Tsub256))
jweak_imp256 = jitter_weak256/(np.sqrt(3600/Tsub256))

#Ft128 data
Tsub128 = 0.377

alldat128 = np.loadtxt(toasim128_all+'/avg.dat')
alldat128_sim = np.loadtxt(toasim128_all+'/sim_00/avg.dat')

strongdat128 = np.loadtxt(toasim128_strong+'/avg.dat')
strongdat128_sim = np.loadtxt(toasim128_strong+'/sim_00/avg.dat')

weakdat128 = np.loadtxt(toasim128_weak+'/avg.dat')
weakdat128_sim = np.loadtxt(toasim128_weak+'/sim_00/avg.dat')

var_alldat128 = np.var(alldat128[:,1])
var_alldat128_sim = np.var(alldat128_sim[:,1])

var_strongdat128 = np.var(strongdat128[:,1])
var_strongdat128_sim = np.var(strongdat128_sim[:,1])

var_weakdat128 = np.var(weakdat128[:,1])
var_weakdat128_sim = np.var(weakdat128_sim[:,1])

jitter_all128 = np.sqrt(var_alldat128 - var_alldat128_sim)
jitter_strong128 = np.sqrt(var_strongdat128 - var_strongdat128_sim)
jitter_weak128 = np.sqrt(var_weakdat128 - var_weakdat128_sim)

jall_imp128 = jitter_all128/(np.sqrt(3600/Tsub128))
jstrong_imp128 = jitter_strong128/(np.sqrt(3600/Tsub128))
jweak_imp128 = jitter_weak128/(np.sqrt(3600/Tsub128))

#Ft64 data
Tsub64 = 0.188625

alldat64 = np.loadtxt(toasim64_all+'/avg.dat')
alldat64_sim = np.loadtxt(toasim64_all+'/sim_00/avg.dat')

strongdat64 = np.loadtxt(toasim64_strong+'/avg.dat')
strongdat64_sim = np.loadtxt(toasim64_strong+'/sim_00/avg.dat')

weakdat64 = np.loadtxt(toasim64_weak+'/avg.dat')
weakdat64_sim = np.loadtxt(toasim64_weak+'/sim_00/avg.dat')

var_alldat64 = np.var(alldat64[:,1])
var_alldat64_sim = np.var(alldat64_sim[:,1])

var_strongdat64 = np.var(strongdat64[:,1])
var_strongdat64_sim = np.var(strongdat64_sim[:,1])

var_weakdat64 = np.var(weakdat64[:,1])
var_weakdat64_sim = np.var(weakdat64_sim[:,1])

jitter_all64 = np.sqrt(var_alldat64 - var_alldat64_sim)
jitter_strong64 = np.sqrt(var_strongdat64 - var_strongdat64_sim)
jitter_weak64 = np.sqrt(var_weakdat64 - var_weakdat64_sim)

jall_imp64 = jitter_all64/(np.sqrt(3600/Tsub64))
jstrong_imp64 = jitter_strong64/(np.sqrt(3600/Tsub64))
jweak_imp64 = jitter_weak64/(np.sqrt(3600/Tsub64))

#Ft32 data
Tsub32 = 0.0943125

alldat32 = np.loadtxt(toasim32_all+'/avg.dat')
alldat32_sim = np.loadtxt(toasim32_all+'/sim_00/avg.dat')

strongdat32 = np.loadtxt(toasim32_strong+'/avg.dat')
strongdat32_sim = np.loadtxt(toasim32_strong+'/sim_00/avg.dat')

weakdat32 = np.loadtxt(toasim32_weak+'/avg.dat')
weakdat32_sim = np.loadtxt(toasim32_weak+'/sim_00/avg.dat')

var_alldat32 = np.var(alldat32[:,1])
var_alldat32_sim = np.var(alldat32_sim[:,1])

var_strongdat32 = np.var(strongdat32[:,1])
var_strongdat32_sim = np.var(strongdat32_sim[:,1])

var_weakdat32 = np.var(weakdat32[:,1])
var_weakdat32_sim = np.var(weakdat32_sim[:,1])

jitter_all32 = np.sqrt(var_alldat32 - var_alldat32_sim)
jitter_strong32 = np.sqrt(var_strongdat32 - var_strongdat32_sim)
jitter_weak32 = np.sqrt(var_weakdat32 - var_weakdat32_sim)

jall_imp32 = jitter_all32/(np.sqrt(3600/Tsub32))
jstrong_imp32 = jitter_strong32/(np.sqrt(3600/Tsub32))
jweak_imp32 = jitter_weak32/(np.sqrt(3600/Tsub32))

#Ft16 data
Tsub16 = 0.04715625

alldat16 = np.loadtxt(toasim16_all+'/avg.dat')
alldat16_sim = np.loadtxt(toasim16_all+'/sim_00/avg.dat')

strongdat16 = np.loadtxt(toasim16_strong+'/avg.dat')
strongdat16_sim = np.loadtxt(toasim16_strong+'/sim_00/avg.dat')

weakdat16 = np.loadtxt(toasim16_weak+'/avg.dat')
weakdat16_sim = np.loadtxt(toasim16_weak+'/sim_00/avg.dat')

var_alldat16 = np.var(alldat16[:,1])
var_alldat16_sim = np.var(alldat16_sim[:,1])

var_strongdat16 = np.var(strongdat16[:,1])
var_strongdat16_sim = np.var(strongdat16_sim[:,1])

var_weakdat16 = np.var(weakdat16[:,1])
var_weakdat16_sim = np.var(weakdat16_sim[:,1])

jitter_all16 = np.sqrt(var_alldat16 - var_alldat16_sim)
jitter_strong16 = np.sqrt(var_strongdat16 - var_strongdat16_sim)
jitter_weak16 = np.sqrt(var_weakdat16 - var_weakdat16_sim)

jall_imp16 = jitter_all16/(np.sqrt(3600/Tsub16))
jstrong_imp16 = jitter_strong16/(np.sqrt(3600/Tsub16))
jweak_imp16 = jitter_weak16/(np.sqrt(3600/Tsub16))

#Ft8 data
Tsub8 = 0.023574125

alldat8 = np.loadtxt(toasim8_all+'/avg.dat')
alldat8_sim = np.loadtxt(toasim8_all+'/sim_00/avg.dat')

strongdat8 = np.loadtxt(toasim8_strong+'/avg.dat')
strongdat8_sim = np.loadtxt(toasim8_strong+'/sim_00/avg.dat')

weakdat8 = np.loadtxt(toasim8_weak+'/avg.dat')
weakdat8_sim = np.loadtxt(toasim8_weak+'/sim_00/avg.dat')

var_alldat8 = np.var(alldat8[:,1])
var_alldat8_sim = np.var(alldat8_sim[:,1])

var_strongdat8 = np.var(strongdat8[:,1])
var_strongdat8_sim = np.var(strongdat8_sim[:,1])

var_weakdat8 = np.var(weakdat8[:,1])
var_weakdat8_sim = np.var(weakdat8_sim[:,1])

jitter_all8 = np.sqrt(var_alldat8 - var_alldat8_sim)
jitter_strong8 = np.sqrt(var_strongdat8 - var_strongdat8_sim)
jitter_weak8 = np.sqrt(var_weakdat8 - var_weakdat8_sim)

jall_imp8 = jitter_all8/(np.sqrt(3600/Tsub8))
jstrong_imp8 = jitter_strong8/(np.sqrt(3600/Tsub8))
jweak_imp8 = jitter_weak8/(np.sqrt(3600/Tsub8))

#Ft4 data
Tsub4 = 0.0117890625

alldat4 = np.loadtxt(toasim4_all+'/avg.dat')
alldat4_sim = np.loadtxt(toasim4_all+'/sim_00/avg.dat')

strongdat4 = np.loadtxt(toasim4_strong+'/avg.dat')
strongdat4_sim = np.loadtxt(toasim4_strong+'/sim_00/avg.dat')

weakdat4 = np.loadtxt(toasim4_weak+'/avg.dat')
weakdat4_sim = np.loadtxt(toasim4_weak+'/sim_00/avg.dat')

var_alldat4 = np.var(alldat4[:,1])
var_alldat4_sim = np.var(alldat4_sim[:,1])

var_strongdat4 = np.var(strongdat4[:,1])
var_strongdat4_sim = np.var(strongdat4_sim[:,1])

var_weakdat4 = np.var(weakdat4[:,1])
var_weakdat4_sim = np.var(weakdat4_sim[:,1])

jitter_all4 = np.sqrt(var_alldat4 - var_alldat4_sim)
jitter_strong4 = np.sqrt(var_strongdat4 - var_strongdat4_sim)
jitter_weak4 = np.sqrt(var_weakdat4 - var_weakdat4_sim)

jall_imp4 = jitter_all4/(np.sqrt(3600/Tsub4))
jstrong_imp4 = jitter_strong4/(np.sqrt(3600/Tsub4))
jweak_imp4 = jitter_weak4/(np.sqrt(3600/Tsub4))

#Ft2 data
Tsub2 = 0.00589453125

alldat2 = np.loadtxt(toasim2_all+'/avg.dat')
alldat2_sim = np.loadtxt(toasim2_all+'/sim_00/avg.dat')

strongdat2 = np.loadtxt(toasim2_strong+'/avg.dat')
strongdat2_sim = np.loadtxt(toasim2_strong+'/sim_00/avg.dat')

weakdat2 = np.loadtxt(toasim2_weak+'/avg.dat')
weakdat2_sim = np.loadtxt(toasim2_weak+'/sim_00/avg.dat')

var_alldat2 = np.var(alldat2[:,1])
var_alldat2_sim = np.var(alldat2_sim[:,1])

var_strongdat2 = np.var(strongdat2[:,1])
var_strongdat2_sim = np.var(strongdat2_sim[:,1])

var_weakdat2 = np.var(weakdat2[:,1])
var_weakdat2_sim = np.var(weakdat2_sim[:,1])

jitter_all2 = np.sqrt(var_alldat2 - var_alldat2_sim)
jitter_strong2 = np.sqrt(var_strongdat2 - var_strongdat2_sim)
jitter_weak2 = np.sqrt(var_weakdat2 - var_weakdat2_sim)

jall_imp2 = jitter_all2/(np.sqrt(3600/Tsub2))
jstrong_imp2 = jitter_strong2/(np.sqrt(3600/Tsub2))
jweak_imp2 = jitter_weak2/(np.sqrt(3600/Tsub2))

#Long obs data
longtsub = 8
longobs = np.loadtxt(msp_128+'/avg.dat')
longobs_sim = np.loadtxt(msp_128+'/sim_00/avg.dat')

var_longobs = np.var(longobs[:,1])
var_longobs_sim = np.var(longobs_sim[:,1])

jitter_longobs = np.sqrt(var_longobs-var_longobs_sim)

jlongobs_imp = jitter_longobs/(np.sqrt(3600/longtsub))

#Graph showing Tsub vs Jitter value
xtsub =[Tsub256,Tsub128,Tsub64,Tsub32,Tsub16,Tsub8,Tsub4,Tsub2]

fig, ax = plt.subplots()

jall = [jall_imp256,jall_imp128,jall_imp64,jall_imp32,jall_imp16,jall_imp8,jall_imp4,jall_imp2]
jstrong = [jstrong_imp256,jstrong_imp128,jstrong_imp64,jstrong_imp32,jstrong_imp16,jstrong_imp8,jstrong_imp4,jstrong_imp2]
jweak = [jweak_imp256,jweak_imp128,jweak_imp64,jweak_imp32,jweak_imp16,jweak_imp8,jweak_imp4,jweak_imp2]

ax.set_xlabel('Subintegration time (s)')
ax.set_ylabel('Implied Jitter in 1hr (s)')
ax.set_title('Jitter subsets vs integration times')
ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(xtsub,jall,'o-', c='tab:blue', linewidth=1,label='All pulses')
ax.plot(xtsub,jstrong,'o-', c='tab:green', linewidth=1,label='Strong pulses')
ax.plot(xtsub,jweak,'o-', c='tab:orange', linewidth=1,label='Weak pulses')

ax.legend()

fig.tight_layout()
plt.show()
