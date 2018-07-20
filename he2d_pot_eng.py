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
                    # Compute Hamiltonian: H = K + V, where K is computed by three-points finite difference
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
    return pot, epsilon

# ====== start main program ====
# Input parameters
n = 40  
dz = 0.1
na = 200 
nb = 200
da = 0.0001
db = 0.0001
n_epsilon = 5
au_to_ev = 27.211

# Call the function compute potentials and energies
he2D_pot, he2D_epsilon = he2D_potientials_energies(n, dz, na, nb, da, db, n_epsilon)

# save potentials and energies
np.savetxt("he2d_potentials.csv", he2D_pot, delimiter=",")
np.savetxt("he2d_energies.csv", he2D_epsilon, delimiter=",")

