import numpy as np
import matplotlib.pyplot as plt
import propagation as prop
import general_formulas as gf

# Define the parameters
wavelength = 0.589
num_pixels = 2**7
oversamp = 2**3

p = np.linspace(0.1,2,20,endpoint=True)
p_ext = np.linspace(0.0,max(p),len(p)+1,endpoint=True)
throughput = np.zeros(len(p)+1)
z_max = 16
cnms = np.zeros(z_max)

no_pinhole = prop.propagate(cnms,1,0,max_zerns = z_max, wavelength=wavelength,pup_width=num_pixels,fp_oversamp=oversamp, mode_type = 'Zernike')
summed = no_pinhole.sum()
for i,size in enumerate(p):
    l = prop.propagate(cnms,0,size,max_zerns = z_max, wavelength=wavelength,pup_width=num_pixels,fp_oversamp=int(oversamp*max(p)/size), mode_type = 'Zernike')
    throughput[i+1] = l.sum()/summed

# calculate a linear fit to the data
lin_line,vals_1 = gf.calc_fit(p_ext,throughput,gf.linear_line)
quad_line,vals_2 = gf.calc_fit(p_ext,throughput,gf.quadratic_line)
exp_line,vals_3 = gf.calc_fit(p_ext,throughput,gf.exponential_line)

plt.figure()
plt.plot(p_ext,throughput,color='black', linestyle='-', label='Data')
#plt.plot(p_ext,lin_line,color='red', linestyle='--', label='Linear Fit')
#plt.plot(p_ext,quad_line,color='green', linestyle='--', label='Quadratic Fit')
#plt.plot(p_ext,exp_line,color='blue', linestyle='--', label='Exponential Fit')
#plt.plot([min(p_ext),max(p_ext)],[0.5,0.5],color='orange',linestyle=':')
plt.title('Throughput vs Pinhole Size')
#plt.text(0.1,0.8,'Fitting function: y = {:.3f}x + {:.3f}'.format(vals_1[0],vals_1[1]),transform=plt.gca().transAxes,fontsize=12,
#         color='red')
#plt.text(0.1,0.7,'Fitting function: y = {:.3f}$x^2$ + {:.3f}x + {:.3f}'.format(vals_2[0],vals_2[1],vals_2[2]),transform=plt.gca().transAxes,fontsize=12,
#         color='green')
#plt.text(0.1,0.6,'Fitting function: y = {:.3f}e$^{{{:.3f}x}}$'.format(vals_3[0],vals_3[1]),transform=plt.gca().transAxes,fontsize=12,
#            color='blue')
plt.xlabel('Pinhole size ($\lambda/D$)')
plt.ylabel('Throughput')
plt.legend()
plt.show()