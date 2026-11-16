import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
plt.rcParams['mathtext.fontset'] = 'cm'


# Using rcParams to set default style for all plots
# plt.rcParams.update({
#     'font.size': 16,
#     'font.family': 'serif',
#     'font.weight': 'normal'
# })

plt.rcParams.update({
    'axes.labelsize': 30,   # Label size
    'font.size': 16,  
    'mathtext.fontset': 'cm',  # Use Computer Modern font
    'mathtext.default': 'it' ,  # Default math font style
    'font.family': 'serif',
    'font.weight': 'normal'
})
#%%

def calculate_ustar_h(Ug, z0, lat=45):
    """
    Calculate friction velocity from geostrophic wind and roughness length
    
    Parameters:
    G: geostrophic wind speed (m/s)
    z0: roughness length (m)
    lat: latitude in degrees (default=45)
    
    Returns:
    u_star: friction velocity (m/s)
    """
    # Constants
    k = 0.4  # von Karman constant
    C1 = 4.5  # empirical constant
    omega = 7.2921e-5  # Earth's angular velocity (rad/s)
    
    # Calculate Coriolis parameter
    f = 2 * omega * np.sin(np.deg2rad(lat))
    
    # Calculate Rossby number
    Ro = Ug / (f * z0)
    
    # Calculate friction velocity
    u_star = (k * Ug) / (np.log(Ro) - C1)
    C2 = 0.25
    h = C2 * (u_star / f)
    return u_star , h


#%%
k = 0.4
zR = 10
z0 = np.array([0.0005 , 0.005 , 0.05 , 0.1  , 0.25 , 0.5 , 1.0 , 2 , 3])
Cd = k**2 / (np.log(zR/z0))**2





#%% Plot the Drag coefficient with the surface roughness 
plt.figure(figsize=(8,5))

plt.plot(z0,Cd,'k')
plt.xlabel(r'$z_{0}$')       # Explicit grouping
plt.ylabel(' $C_d$')

plt.tight_layout()
plt.show()



#%% Plot the Drag coefficient with the surface roughness 
Ugs = np.linspace(2, 20,4,dtype=int)




plt.figure(figsize=(16,5.5))

for u_idx , Ug in enumerate(Ugs):
    u_star = np.zeros_like(z0)
    h =  np.zeros_like(z0)
    
    for i in range(0,z0.size):
        u_star[i],h[i] = calculate_ustar_h(Ug, z0[i], lat=45)
    plt.subplot(1,2,1)
    plt.plot(z0,u_star,alpha = 0.25*(u_idx+1), color = 'k', label = rf'Ug = {Ug} m/sec')
    plt.xlabel(r'${z_0} [m]$')
    plt.ylabel('$u_* [m/sec]$')
    
    plt.subplot(1,2,2)
    plt.plot(z0,h,alpha = 0.25*(u_idx+1), color = 'k', label = rf'Ug = {Ug} m/sec')
    plt.xlabel(r'${z_0} [m]$')
    plt.ylabel('$h [m]$')

        
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%%  For the validation cases 
Ugs = np.array([5.85, 6.5, 7.15, 10])
z0 = np.array([0.0001,0.01,0.05, 0.1 , 0.2])

# Print header
print("\n{:<10} {:<10} {:<10} {:<10}".format('Ug (m/s)', 'z0 (m)', 'u* (m/s)','h (m)'))
print("-" * 32)  # Separator line

for u_idx, Ug in enumerate(Ugs):
    u_star = np.zeros_like(z0)
    
    for i in range(0, z0.size):
        u_star[i],h[i] = calculate_ustar_h(Ug, z0[i], lat=45)
        # Format numbers with 3 decimal places
        print("{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(Ug, z0[i], u_star[i], h[i]))




#%% 

