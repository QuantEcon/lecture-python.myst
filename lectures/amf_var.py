
import numpy as np
import scipy as sp
import scipy.linalg as la
import quantecon as qe
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm



class AMF_LSS_VAR:
    """
    This class transforms an additive (multipilcative)
    functional into a QuantEcon linear state space system.
    """

    def __init__(self, A, B, D, F=None, nu=None, x_0=None):
        # Unpack required elements
        self.nx, self.nk = B.shape
        self.A, self.B = A, B

        # checking the dimension of D (extended from the scalar case)
        if len(D.shape) > 1 and D.shape[0] != 1:
            self.nm = D.shape[0]
            self.D = D
        elif len(D.shape) > 1 and D.shape[0] == 1:
            self.nm = 1
            self.D = D
        else:
            self.nm = 1
            self.D = np.expand_dims(D, 0)

        # Create space for additive decomposition
        self.add_decomp = None
        self.mult_decomp = None

        # Set F
        if not np.any(F):
            self.F = np.zeros((self.nk, 1))
        else:
            self.F = F

        # Set nu
        if not np.any(nu):
            self.nu = np.zeros((self.nm, 1))
        elif type(nu) == float:
            self.nu = np.asarray([[nu]])
        elif len(nu.shape) == 1:
            self.nu = np.expand_dims(nu, 1)
        else:
            self.nu = nu

        if self.nu.shape[0] != self.D.shape[0]:
            raise ValueError("The dimension of nu is inconsistent with D!")

        # Initialize the simulator
        self.x_0 = x_0

        # Construct BIG state space representation
        self.lss = self.construct_ss()

    def construct_ss(self):
        """
        This creates the state space representation that can be passed
        into the quantecon LSS class.
        """

        # Pull out useful info
        nx, nk, nm = self.nx, self.nk, self.nm
        A, B, D, F, nu = self.A, self.B, self.D, self.F, self.nu

        if self.add_decomp:
            nu, H, g = self.add_decomp
        else:
            nu, H, g = self.additive_decomp()

        # Auxiliary blocks with 0's and 1's to fill out the lss matrices
        nx0c = np.zeros((nx, 1))
        nx0r = np.zeros(nx)
        nx1 = np.ones(nx)
        nk0 = np.zeros(nk)
        ny0c = np.zeros((nm, 1))
        ny0r = np.zeros(nm)
        ny1m = np.eye(nm)
        ny0m = np.zeros((nm, nm))
        nyx0m = np.zeros_like(D)

        # Build A matrix for LSS
        # Order of states is: [1, t, xt, yt, mt]
        A1 = np.hstack([1, 0, nx0r, ny0r, ny0r])            # Transition for 1
        A2 = np.hstack([1, 1, nx0r, ny0r, ny0r])            # Transition for t
        A3 = np.hstack([nx0c, nx0c, A, nyx0m.T, nyx0m.T])   # Transition for x_{t+1}
        A4 = np.hstack([nu, ny0c, D, ny1m, ny0m])           # Transition for y_{t+1}
        A5 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])     # Transition for m_{t+1}
        Abar = np.vstack([A1, A2, A3, A4, A5])

        # Build B matrix for LSS
        Bbar = np.vstack([nk0, nk0, B, F, H])

        # Build G matrix for LSS
        # Order of observation is: [xt, yt, mt, st, tt]
        G1 = np.hstack([nx0c, nx0c, np.eye(nx), nyx0m.T, nyx0m.T])    # Selector for x_{t}
        G2 = np.hstack([ny0c, ny0c, nyx0m, ny1m, ny0m])               # Selector for y_{t}
        G3 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])               # Selector for martingale
        G4 = np.hstack([ny0c, ny0c, -g, ny0m, ny0m])                  # Selector for stationary
        G5 = np.hstack([ny0c, nu, nyx0m, ny0m, ny0m])                 # Selector for trend
        Gbar = np.vstack([G1, G2, G3, G4, G5])

        # Build H matrix for LSS
        Hbar = np.zeros((Gbar.shape[0], nk))

        # Build LSS type
        if not np.any(self.x_0):
            x0 = np.hstack([1, 0, nx0r, ny0r, ny0r])
        else:
            x0 = self.x_0

        S0 = np.zeros((len(x0), len(x0)))
        lss = qe.lss.LinearStateSpace(Abar, Bbar, Gbar, Hbar, mu_0=x0, Sigma_0=S0)

        return lss

    def additive_decomp(self):
        """
        Return values for the martingale decomposition
            - nu        : unconditional mean difference in Y
            - H         : coefficient for the (linear) martingale component (kappa_a)
            - g         : coefficient for the stationary component g(x)
            - Y_0       : it should be the function of X_0 (for now set it to 0.0)
        """
        I = np.eye(self.nx)
        A_res = la.solve(I - self.A, I)
        g = self.D @ A_res
        H = self.F + self.D @ A_res @ self.B

        return self.nu, H, g

    def multiplicative_decomp(self):
        """
        Return values for the multiplicative decomposition (Example 5.4.4.)
            - nu_tilde  : eigenvalue
            - H         : vector for the Jensen term
        """
        nu, H, g = self.additive_decomp()
        nu_tilde = nu + (.5)*np.expand_dims(np.diag(H @ H.T), 1)

        return nu_tilde, H, g





def future_moments(amf_future, N=25):

    nx, nk, nm = amf_future.nx, amf_future.nk, amf_future.nm
    nu_tilde, H, g = amf_future.multiplicative_decomp()
    
    # Allocate space (nm is the number of additive functionals)
    mbounds = np.zeros((3, N))
    sbounds = np.zeros((3, N))
    ybounds = np.zeros((3, N))
    #mbounds_mult = np.zeros((3, N))
    #sbounds_mult = np.zeros((3, N))

    # Simulate for as long as we wanted
    moment_generator = amf_future.lss.moment_sequence()
    tmoms = next(moment_generator)

    # Pull out population moments
    for t in range (N-1):
        tmoms = next(moment_generator)
        ymeans = tmoms[1]
        yvar = tmoms[3]

        # Lower and upper bounds - for each additive functional
        yadd_dist = norm(ymeans[nx], np.sqrt(yvar[nx, nx]))
        ybounds[:2, t+1] = yadd_dist.ppf([0.1, .9])
        ybounds[2, t+1] = yadd_dist.mean()

        madd_dist = norm(ymeans[nx+nm], np.sqrt(yvar[nx+nm, nx+nm]))
        mbounds[:2, t+1] = madd_dist.ppf([0.1, .9])
        mbounds[2, t+1] = madd_dist.mean()

        sadd_dist = norm(ymeans[nx+2*nm], np.sqrt(yvar[nx+2*nm, nx+2*nm]))
        sbounds[:2, t+1] = sadd_dist.ppf([0.1, .9])
        sbounds[2, t+1] = sadd_dist.mean()


        #Mdist = lognorm(np.asscalar(np.sqrt(yvar[nx+nm, nx+nm])), scale=np.asscalar(np.exp(ymeans[nx+nm]- \
        #                                              t*(.5)*np.expand_dims(np.diag(H @ H.T), 1))))
        #Sdist = lognorm(np.asscalar(np.sqrt(yvar[nx+2*nm, nx+2*nm])),
        #                scale = np.asscalar(np.exp(-ymeans[nx+2*nm])))
        #mbounds_mult[:2, t+1] = Mdist.ppf([.01, .99])
        #mbounds_mult[2, t+1] = Mdist.mean()

        #sbounds_mult[:2, t+1] = Sdist.ppf([.01, .99])
        #sbounds_mult[2, t+1] = Sdist.mean()

    ybounds[:, 0] = amf_future.x_0[2+nx]
    mbounds[:, 0] = mbounds[-1, 1]
    sbounds[:, 0] = -g @ amf_future.x_0[2:2+nx]

    #mbounds_mult[:, 0] = mbounds_mult[-1, 1]
    #sbounds_mult[:, 0] = np.exp(-g @ amf_future.x_0[2:2+nx])

    return mbounds, sbounds, ybounds #, mbounds_mult, sbounds_mult
