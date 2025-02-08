import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import _fast_ddtw
np.random.seed(seed=1234) # reproduceability of example


# Example to illustrate that DTW will squish data from a large depth interval
# minimizing DTW(m,n).  windowing  will help, but probably must be learned.
#
# Following this (and similar) experiments, DTW was abandoned for profile
# alignment in this study, and later PCC_warp was developed.
#
# script showcasing DTW and derivative DTW on a simple profile
# to make this example more relevant for geotechnics
#   *  small randomness is added to shifted curves
#   *  one curve has a bulge not present in the others


def simulated_push_resistance( rise=1, const=1.5, rand_fact=0, bulge=0 ): # returns 150 element np.array
    i = np.arange(150) # some depth increments
    
    x = ( (i>=85)*np.sin(20*i*np.pi/180) ) * np.sin( (i-80)*np.pi/180 ) # 4 peaks in last half of curve
    x = (x>0) * x # only positive

    x += rise*i/len(i) # add increasing trend
    x += const # shift by a constant

    y = (i<=40) * 2*np.cos( (i-10)*3*np.pi/180 ) # cosine curve at top
    y = (y>0) * y # only positive
    x += y

    z = np.sin( 20*i*np.pi/180 ) * (i>50) * (i<70) # scale&add bulge
    z = (z>0) * z # only positive
    x += bulge*z

    # add noise
    x += rand_fact*( np.random.rand(len(x))-0.5 )

    x *= 1000 # scale to kPa range

    return x


def distort_d( y, d, scale, length ): # ends intact but data squished towards senter
    delta = d[-1]-d[0] # profile increments
    scaling = delta*scale

    # time distorted with an addition of a sinus bulge from d[0] to d[-1]
    delta_d = ( (np.sin(np.arange(len(y))/len(y)*np.pi)) ) * scaling
    d_dist = d - delta_d # warping

    # resample
    n = int( len(d) * length )
    d_r = np.linspace( d[0], d[-1], n )
    y_r = np.interp( d_r, d_dist, y )

    delta_d_r = np.interp( d_r, d_dist, delta_d ) # resample warp distance

    return y_r, d_r, delta_d_r # resampled


def restore_d( path, d_from, d_to ):
    d_restored = np.copy( d_to )
    for item in path:
        d_restored[item[0]] = d_from[item[1]]
    return d_restored


def dtw_example():
    dtw_penalty = 0

    noise = 0.1
    rise = 1.2
    offset = 0
    d_from = 4
    d_to = 12

    x  = simulated_push_resistance( rise=rise, const=offset, rand_fact=0, bulge=0 )
    y1 = simulated_push_resistance( rise=rise, const=offset, rand_fact=noise, bulge=0 )
    y2 = simulated_push_resistance( rise=rise, const=offset, rand_fact=noise, bulge=1 )

    d = np.linspace( d_from, d_to, len(x) ) # 4-12 m

    # generate distorted curve
    y1_dist, d1_dist, delta_1 = distort_d( y1, d, scale=0.1, length=.85 )
    y2_dist, d2_dist, delta_2 = distort_d( y2, d, scale=0.2, length=1.50 )

    # dwt warp alignment
    path_1 = dtw.warping_path_fast( y1_dist, x, penalty=dtw_penalty ) # red
    path_2 = dtw.warping_path_fast( y2_dist, x, penalty=dtw_penalty ) # green
    _, path_3 = _fast_ddtw.fast_ddtw( y2_dist, x ) # green dashed
    _, path_4 = _fast_ddtw.fast_ddtw( y1_dist, x ) # red dashed
    
    # restored depths from warp alignment
    d1_restored = restore_d( path_1, d, d1_dist )
    d2_restored = restore_d( path_2, d, d2_dist )
    d3_restored = restore_d( path_3, d, d2_dist )
    d4_restored = restore_d( path_4, d, d1_dist )

    # depth shifts
    d_0_shift = d-d
    d_1_shift = d1_restored-d1_dist
    d_2_shift = d2_restored-d2_dist
    d_3_shift = d3_restored-d2_dist
    d_4_shift = d4_restored-d1_dist

    # theoretical depth alignment
    d1_th_shift = d1_dist + delta_1
    d2_th_shift = d2_dist + delta_2

    fig, axs = plt.subplots( 1,4, sharey=True, figsize=[14,8] )

    # theoretical alignment
    axs[0].set_ylabel( 'depth (m)' )
    axs[0].set_xlabel( 'simulated curves (kPa)' )
    axs[0].plot( x, d, lw=2, c=(0/255,142/255,194/255), zorder=-1 )
    axs[0].plot( y1_dist, d1_th_shift, c=(237/255,28/255,46/255), lw=1 )
    axs[0].plot( y2_dist, d2_th_shift, c=(93/255,184/255,46/255), lw=1 )
    axs[0].annotate('extra peak\nonly present\nin green curve', xy=(960,6.97), xycoords='data',
        xytext=(850,6.0), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # simulated curves
    axs[1].set_xlabel( 'depth shifts added (kPa)' )
    axs[1].plot( x, d, lw=2, zorder=-1, c=(0/255,142/255,194/255) ) # blue NPRA
    axs[1].plot( y1_dist, d1_dist, c=(237/255,28/255,46/255), lw=1 ) # red (NPRA)
    axs[1].plot( y2_dist, d2_dist, c=(93/255,184/255,46/255),  lw=1 ) # green (NPRA)

    # depth shifts
    axs[2].set_xlabel( 'shift from unwarped depth (m)' )
    axs[2].plot( d_0_shift, d, lw=2, zorder=-1, c=(0/255,142/255,194/255), label='reference curve' )
    axs[2].plot( d_1_shift, d1_dist, lw=1, c=(237/255,28/255,46/255), label='dtw alignment' )
    axs[2].plot( d_4_shift, d1_dist, ls='--', lw=1, c=(237/255,28/255,46/255), label='ddtw alignment' )
    axs[2].plot( d_2_shift, d2_dist, lw=1, c=(93/255,184/255,46/255), label='dtw a. w/bulge' )
    axs[2].plot( d_3_shift, d2_dist, ls='--', lw=1, c=(93/255,184/255,46/255), label='ddtw a. w/bulge' )
    axs[2].plot( delta_1, d1_dist, ls='--', lw=1, c=(68/255,79/255,85/255), label='true shift' )  # dark grey NPRA
    axs[2].plot( delta_2, d2_dist, ls='--', lw=1, c=(68/255,79/255,85/255) )
    axs[2].annotate('large distortion', xy=(2.11,5.44), xycoords='data',
        xytext=(1.8,4.6), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))    
    axs[2].annotate('', xy=(1.72,8.08), xycoords='data',
        xytext=(2.56,4.84), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    axs[2].annotate('', xy=(2.67,6.65), xycoords='data',
        xytext=(2.56,4.84), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[3].set_xlabel( 'dtw alignment (kPa)' )
    axs[3].plot( x, d, lw=2, c=(0/255,142/255,194/255), zorder=-1 )
    axs[3].plot( y1_dist, d1_restored, c=(237/255,28/255,46/255), lw=1 )
    axs[3].plot( y1_dist, d4_restored, ls='--', c=(237/255,28/255,46/255), lw=1 )
    axs[3].plot( y2_dist, d2_restored, c=(93/255,184/255,46/255), lw=1 )
    axs[3].plot( y2_dist, d3_restored, ls='--', c=(93/255,184/255,46/255), lw=1 )
    axs[3].annotate('large distortion', xy=(460,8.08), xycoords='data',
        xytext=(990,7.5), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    axs[3].annotate('incorrect peak\nalignment', xy=(1175,8.94), xycoords='data',
        xytext=(840,8.2), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[3].annotate('peak squished', xy=(1000,9.75), xycoords='data',
        xytext=(1200,9.4), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[0].set_ylim( d_from, d_to )
    axs[0].invert_yaxis()
    axs[2].legend()

    for ax in axs:
        if ax.get_xlim()[1] > 100: ax.set_xlim( 0, ax.get_xlim()[1] )
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()        
    plt.show()


if __name__ == '__main__':
    dtw_example()