# import tensorflow
import numpy as np

# A function to compute potential and eigen energies
def he2D_potientials_energies(n, dz, na, nb, da, db, n_epsilon):
    n_sq = n ** 2   # total 2D size
    z1 = np.zeros(n)
    z2 = np.zeros(n)
    dz2 = dz ** 2
    two_inverse_dz2 = 2 / dz2
    mhalf_inverse_dz2 = -0.5 / dz2
    z_start = -n/2*dz
    a2 = np.zeros(na)
    b2 = np.zeros(nb)
    a2_start = 0.5 - na/2*da
    b2_start = 0.339 - na/2*db
    n_pot = na*nb  # number of potentials
    pot = np.zeros((n_sq, n_pot))
    ham = np.zeros((n_sq, n_sq))
    epsilon = np.zeros((n_epsilon, n_pot))
    au_to_ev = 27.211

    for j in range(n):
        z1[j] = z_start + j*dz
        z2[j] = z_start + j*dz
    for j in range(na):
        a2[j] = a2_start + j*da
    for j in range(nb):
        b2[j] = b2_start + j*db

    for i in range(na):  # loops of parameters a2, b2
        for j in range(nb):
            i_pot = i*nb + j
            i_sq = 0
            for ii in range(n):  # loops of coordinates z1, z2
                for jj in range(n):
                    i_sq = ii*n + jj
                    pot[i_sq, i_pot] = 1.0/np.sqrt((z1[ii]-z2[jj])*(z1[ii]-z2[jj]) + b2[j]) - 2.0/np.sqrt(z1[ii]*z1[ii] + a2[i]) - 2.0/np.sqrt(z2[jj]*z2[jj] + a2[i])  # soft Coulomb potential for 2D Helium atom
                    ham[i_sq, i_sq] = two_inverse_dz2 + pot[i_sq, i_pot]  # diagonal elements of H
                    if ii > 0:
                       ham[i_sq, (ii - 1) * n + jj] = mhalf_inverse_dz2  # ii-1 elements of H
                    if ii < n-1:
                       ham[i_sq, (ii + 1) * n + jj] = mhalf_inverse_dz2  # ii+1 elements of H
                    if jj > 0:
                       ham[i_sq, ii * n + jj - 1] = mhalf_inverse_dz2  # jj-1 elements of H
                    if jj < n-1:
                       ham[i_sq, ii * n + jj + 1] = mhalf_inverse_dz2  # jj+1 elements of H
            eigen_energies = np.sort(np.linalg.eigvals(ham))  # compute ordered eigen energies for each potential
            epsilon[..., i_pot] = eigen_energies[0:n_epsilon]  # select the first sevearl eigen energies
            #print "===== Eigen energies for the", i_pot, "-th potential ====="
            #print epsilon[..., i_pot] * au_to_ev
    return pot, epsilon

# Input parameters
n = 40 #200 # input 1D size: n1, must be multiple of 4.
dz = 0.1
na = 50 #50  
nb = 50 #50
da = 0.01 #0.002
db = 0.01 #0.002
n_epsilon = 5
au_to_ev = 27.211

# Call the function
he2D_pot, he2D_epsilon = he2D_potientials_energies(n, dz, na, nb, da, db, n_epsilon)

#savetxt('submission2.csv', zip(a,b), delimiter=',', fmt='%f')
np.savetxt("he2d_potentials.csv", he2D_pot, delimiter=",")
np.savetxt("he2d_energies.csv", he2D_epsilon, delimiter=",")

# Print results
#for i in range(na):  # loops of parameters a2, b2
#    for j in range(nb):
#        i_pot = i*nb + j
        #print "===== Eigen energies for the", i_pot, "-th potential ====="
        #print he2D_epsilon[..., i_pot] * au_to_ev

