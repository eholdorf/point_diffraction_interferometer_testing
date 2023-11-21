import numpy as np
import forward_model as fm
import matplotlib.pyplot as plt
import propagation as prop
import aotools

# need to poke each mode in a small positive and negative direction and see how 
# intensity pattern changes to generate the control matrix

def generate_matrix(max_zerns=20,pup_width=2**6,fp_oversamp=2**2,pinhole_size=1,wavelength=0.589,amp=1e-4,ratio=0.5,mode='Zernike'):
    cnms = np.zeros(max_zerns,dtype=np.float64)

    IM = np.zeros((pup_width**2,max_zerns),dtype=np.float64)
    for Z in range(max_zerns):
        # each time we run the forward model we need to reset the aberrations
        cnms = np.zeros(max_zerns,dtype=np.float64)
        # change the amplitude of the Zernike mode positively and negatively 
        # and run the forward model
        cnms[Z] = amp
        intensity_pos = prop.propagate(cnms,ratio,pinhole_size,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength,mode_type=mode)
        #intensity_pos = fm.forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
        cnms[Z] = -amp
        intensity_neg = prop.propagate(cnms,ratio,pinhole_size,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength,mode_type=mode)
        #intensity_neg = fm.forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
        diff_intensity = intensity_pos - intensity_neg
        # store the difference in the control matrix
        IM[:,Z] = diff_intensity.ravel()/(2*amp)

    IM_inv = np.linalg.pinv(IM)

    return IM_inv

if __name__=="__main__":
    # define the parameters
    max_zerns = 50
    cnms = np.zeros(max_zerns,dtype=np.float64)
    pup_width = 2**7
    fp_oversamp = 2**4
    pinhole_size = 1
    wavelength = 0.589
    piston = True
    amp = 1e-4
    if piston:
        IM = np.zeros((pup_width**2,max_zerns),dtype=np.float64)
        for Z in range(max_zerns):
            # each time we run the forward model we need to reset the aberrations
            cnms = np.zeros(max_zerns,dtype=np.float64)
            # change the amplitude of the Zernike mode positively and negatively 
            # and run the forward model
            cnms[Z] = amp
            intensity_pos = prop.propagate(cnms,0.5,1,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength)
            intensity_pos = fm.forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
            cnms[Z] = -amp
            intensity_neg = prop.propagate(cnms,0.5,1,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength)
            intensity_neg = fm.forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
            diff_intensity = intensity_pos - intensity_neg
            # store the difference in the control matrix
            IM[:,Z] = diff_intensity.ravel()/(2*amp)

        # now we need to find the eigenvalues of the control matrix
        U,S,Vh = np.linalg.svd(IM)
        
        plt.figure()
        plt.plot(list(range(max_zerns)),S,'ko')
        plt.xlabel('Mode Number')
        plt.ylabel('Singular Value')
        plt.title('Singular Values of Interaction Matrix')
        plt.xticks(list(range(max_zerns)))
        plt.show()

    if not piston:
        IM = np.zeros((pup_width**2,max_zerns-1),dtype=np.float64)
        for Z in range(1,max_zerns):
            # each time we run the forward model we need to reset the aberrations
            cnms = np.zeros(max_zerns,dtype=np.float64)
            # change teh amplitude of the Zernike mode positively and negatively 
            # and run the forward model
            cnms[Z] = amp
            intensity_pos = prop.propagate(cnms,0.5,1,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength)
            cnms[Z] = 0
            intensity_neg = prop.propagate(cnms,0.5,1,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength)
            diff_intensity = intensity_pos - intensity_neg
            # store the difference in the control matrix
            IM[:,Z-1] = diff_intensity.ravel()/(1*amp)

        # now we need to find the eigenvalues of the control matrix
        U,S,Vh = np.linalg.svd(IM)

        plt.figure()
        plt.plot(list(range(1,max_zerns)),S,'ko')
        plt.xlabel('Mode Number')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues of Interaction Matrix')
        plt.xticks(list(range(1,max_zerns)))
        plt.show()

    # find the pseudo inverse of the control matrix
    IM_inv = np.linalg.pinv(IM)



    cnms = np.zeros(max_zerns,dtype=np.float64)
    # change teh amplitude of the Zernike mode positively and negatively 
    # and run the forward model
    cnms[4] = 1e-4
    #cnms[7] = -2e-4
    intensity =  prop.propagate(cnms,0.5,1,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength)
    intensity = fm.forward_prop(max_zerns,cnms,pup_width,fp_oversamp,pinhole_size,wavelength)
    intensity_no_aberations = prop.propagate(np.zeros(max_zerns,dtype=np.float64),0.5,1,max_zerns=max_zerns,pup_width=pup_width,fp_oversamp=fp_oversamp,wavelength=wavelength)
    intensity_no_aberations = fm.forward_prop(max_zerns,np.zeros(max_zerns,dtype=np.float64),pup_width,fp_oversamp,pinhole_size,wavelength)

    C = np.dot(IM_inv,(intensity-intensity_no_aberations).ravel())
    print(C)
    Cs = np.zeros(max_zerns,dtype=np.float64)
    if len(C)<len(Cs):
        Cs[len(Cs)-len(C):] = C
    else:
        Cs=C
        Cs[0] = 0
    plt.figure()
    plt.imshow(np.einsum("ijk,i->jk",aotools.zernikeArray(max_zerns,pup_width,norm='rms'),Cs))
    plt.colorbar()
    plt.title('Recovered')

    plt.figure()
    plt.imshow(np.einsum("ijk,i->jk",aotools.zernikeArray(max_zerns,pup_width,norm='rms'),cnms))
    plt.colorbar()
    plt.title('Original')

    plt.figure()
    plt.imshow(np.einsum("ijk,i->jk",aotools.zernikeArray(max_zerns,pup_width,norm='rms'),cnms)-np.einsum("ijk,i->jk",aotools.zernikeArray(max_zerns,pup_width,norm='rms'),Cs))
    plt.colorbar()
    plt.title('Difference')
    plt.show()