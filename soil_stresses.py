'''
    class for calculating total- and effective soil stress profiles

    stress model is calculated for each cm between MIN_DEPTH ( <=0m ) and MAX_DEPTH ( >=100m )
    desired depths are interpolated from here when stress functions are called

'''

import numpy as np


class soil_stress_model():
    def __init__( self, gamma=[[0],[19]], u0=[[0,2,60],[0,0,580]], q=0 ):
        # constants
        self.gamma_w = 10 # water unit weight (kN/m³)
        self_delta_d = 0.1 # 1cm const
        self.eps = self_delta_d / 1000

        # save input variables
        self.gamma = gamma # soil unit weight (kN/m³)
        self.u0_def = u0 # in situ pore pressure (kN/m²)
        self.q = q # terrain load (kN/m²)

        # construct depth profile
        self.min_depth = min( 0, min(self.gamma[0]), min(self.u0_def[0]) ) # <=0m (negative: water above terrain)
        self.max_depth = max( 100, max(self.gamma[0]), max(self.u0_def[0]) ) # at least 100m
        self.d = np.arange( self.min_depth, self.max_depth, self_delta_d ) # MISSING: 2 vals for 0 (second for q)

        # update stress profiles
        self.sigma_v0 = self.define_sigma_v0()
        self.u0 = self.define_u0()
        self.sigma_v0_eff = self.define_sigma_v0_eff()


    def define_sigma_v0( self ):
        self.gamma_interp = np.interp( self.d, self.gamma[0], self.gamma[1] ) # interpolate between vals

        self.gamma_interp[self.d < 0-self.eps] = self.gamma_w # shift by eps to ensure gamma_soil at d≈0m
        delta_d = np.diff( self.d )

        delta_sigma_v0 = delta_d  * self.gamma_interp[1:] # calc stress increments
        delta_sigma_v0[ np.abs(self.d[1:]) < self.eps ] += self.q # add terrain load. MISSING:second d0=0! (see init)

        sigma_v0 = np.cumsum( delta_sigma_v0 ) # sum all stress increments with depth
        sigma_v0 = np.insert( sigma_v0, 0, 0 ) # zero stress at top (-> before index 0, add 0)

        return sigma_v0


    def define_u0( self ):
        u0_d = np.array( self.u0_def[0] )
        u0_val = np.array( self.u0_def[1] )

        u0 = np.interp( self.d, u0_d, u0_val ) # interpolate between vals

        # hydrostatic above first definition ( >=0 )
        u0[ self.d < u0_d[0] ] = u0_val[0] + ( self.d[self.d<u0_d[0]]-u0_d[0] ) * self.gamma_w
        u0[ u0<0 ] = 0

        # hydrostatic below last definition
        u0[ self.d > u0_d[-1] ] = u0_val[-1] + ( self.d[self.d>u0_d[-1]]-u0_d[-1] ) * self.gamma_w

        return u0


    def define_sigma_v0_eff( self ):
        return self.sigma_v0-self.u0


    # functions to evaluate stresses at geiven depths
    def calc_sigma_v0( self, depth_profile ):
        return np.interp( depth_profile, self.d, self.sigma_v0 )


    def calc_u0( self, depth_profile ):
        return np.interp( depth_profile, self.d, self.u0 )


    def calc_sigma_v0_eff( self, depth_profile ):
        return np.interp( depth_profile, self.d, self.sigma_v0_eff )


    def plot( self ):
        import matplotlib.pyplot as plt

        fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True, figsize=(6, 6), gridspec_kw={'width_ratios': [3, 1]})

        sigma_label = 'σ' + r'$_v$' + r'$_0$'
        sigma_eff_label = 'σ\'' + r'$_v$' + r'$_0$'
        u0_label = 'u' + r'$_0$'

        ax1.plot( self.u0, self.d, '-', color=(0/255,142/255,194/255), label=u0_label ) # (blue NPRA)
        #ax1.plot( self.u0_def[1], self.u0_def[0], 'x', color=(0,0,0) ) # (68/255,79/255,85/255) - dark grey (NPRA)
        ax1.plot( self.sigma_v0_eff, self.d, '-', color=(93/255,184/255,46/255), label=sigma_eff_label ) # green (NPRA)
        ax1.plot( self.sigma_v0, self.d, '-', color=(237/255,28/255,46/255), label=sigma_label ) # red (NPRA)

        ax2.plot( self.gamma[1], self.gamma[0], 'o', c=(68/255,79/255,85/255), label='lab', zorder=9)
        ax2.plot( self.gamma_interp, self.d, '--', c=(68/255,79/255,85/255), label='interpolated')

        ax1.grid()
        ax2.grid()
        ax1.set_ylim( self.min_depth, 20 )
        ax1.set_xlim( 0, 500 )
        ax1.invert_yaxis()
        ax1.legend()
        ax2.legend(loc='lower right')

        ax1.xaxis.tick_top()
        ax2.xaxis.tick_top()

        ax1.set_xlabel('Stress (kPa)')
        ax1.set_ylabel('Depth (m)')
        titl_2 = 'γ' + ' (kN/m' + r'$^3)$'
        ax2.set_xlabel(titl_2)
        ax1.xaxis.set_label_position('top')
        ax2.xaxis.set_label_position('top')

        plt.show()


if __name__ == '__main__':
    gamma =[ [0], [19] ]
    u0 =[ [1.5], [0] ]
    profile = soil_stress_model( gamma=gamma, u0=u0, q=0 )
    profile.plot()