import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


'''
    module presents 
        digitized data from Lehane et al. 2009: Rate effects on penetrometer resistance in kaolin.
        selected range for fitted model are presented as wireframe along with 3D lines for OCR=1, 2, 5

    the log-scale for V is precalculated and plotted on linear axis (ax.set_xscale('log') has a bug).
    ticks are modified accordingly
'''


t_bar_dataset = { # dataset from Lehane et al. 2009 - extracted using WebPlotDigitizer
    1:{ # OCR=1
        'V': (0.02806, 0.0326, 0.03896, 0.04789, 0.05461, 0.06112, 0.06588, 0.08566, 0.09587, 0.1306, 0.1344, 0.1576, 0.1814, 0.2049, 0.2188, 0.2846, 0.3306, 0.3806, 0.4464, 0.4506, 0.902, 0.9105, 1.109, 1.325, 1.428, 1.613, 2.24, 4.57, 5.461, 7.168, 11.24, 12.94, 14.76, 15.61, 20.3, 22.72, 27.41, 33.69, 39.89, 42.99, 48.12, 63.15, 67.44, 98.14, 127.6, 153.9, 192.8, 200.2, 232.6, 302.4, 314, 326, 404.5, 474.4, 640.5, 1247, 1504, 1974, 3067),
        'q_norm': (4.64, 4.648, 4.991, 5.007, 4.432, 5.118, 4.177, 4.117, 4.863, 4.392, 3.937, 4.181, 2.872, 3.977, 2.441, 2.17, 3.108, 2.701, 2.406, 1.967, 1.811, 1.979, 1.851, 2.186, 2.047, 1.935, 2.062, 1.624, 1.568, 1.516, 1.424, 1.839, 1.444, 1.807, 1.847, 1.424, 1.967, 1.919, 1.703, 2.062, 1.839, 1.676, 2.055, 1.552, 2.086, 2.094, 2.222, 2.182, 2.138, 2.043, 2.366, 1.895, 1.991, 2.158, 1.991, 2.23, 2.537, 2.585, 2.445)},
    2:{
        'V': (0.02837, 0.03966, 0.05543, 0.08206, 0.09563, 0.1337, 0.1886, 0.1886, 0.2846, 0.3685, 0.4017, 0.4681, 0.5201, 0.5614, 0.5614, 0.6298, 0.6606, 0.7923, 0.7999, 0.8311, 0.9322, 1.118, 1.303, 1.354, 1.892, 1.892, 2.123, 2.671, 2.801, 3.733, 3.841, 4.147, 4.696, 6.627, 6.627, 9.352, 9.352, 12.58, 13.07, 13.32, 15.53, 18.62, 18.62, 20.89, 27.05, 27.31, 28.92, 36.74, 39.28, 55.43, 55.97, 67.12, 81.28, 87.74, 137.6, 181.6, 284.6, 446.3, 540.4, 648.1, 855.3, 1354, 1504, 1821, 2381, 3769),
        'q_norm': (8.165, 7.844, 6.707, 6.513, 7.166, 6.108, 7.789, 4.547, 6.295, 5.509, 4.928, 5.744, 4.008, 4.287, 3.609, 3.276, 5.629, 3.542, 4.159, 3.234, 3.688, 3.252, 2.798, 3.591, 3.5, 3.01, 3.27, 3.149, 2.804, 2.846, 2.61, 3.215, 3.324, 3.033, 2.852, 2.676, 2.937, 3.281, 2.719, 3.227, 3.124, 2.87, 3.118, 3.112, 3.111, 2.815, 3.021, 2.869, 3.474, 3.559, 2.845, 3.686, 2.803, 3.952, 3.951, 3.855, 3.782, 4.356, 4.259, 4.132, 4.229, 4.592, 4.404, 4.712, 4.452, 4.954)},
    5:{
        'V': (0.04464, 0.06501, 0.0881, 0.1102, 0.1424, 0.1494, 0.2074, 0.2246, 0.2925, 0.2925, 0.2972, 0.3573, 0.4329, 0.447, 0.4803, 0.6057, 0.6254, 0.7518, 0.7578, 0.8544, 0.8822, 0.8822, 1.06, 1.275, 1.327, 1.496, 1.798, 2.093, 2.11, 2.536, 2.929, 2.952, 2.952, 3.549, 4.3, 4.511, 4.81, 6.017, 6.017, 7.588, 8.42, 10.62, 10.62, 14.98, 14.98, 20.79, 20.96, 25.19, 28.4, 29.33, 29.33, 29.33, 34.14, 35.25, 42.71, 44.1, 48.16, 53.87, 62.21, 75.38, 75.38, 85.67, 88.46, 110.7, 128.8, 142.9, 223.7, 293.7, 355.8, 463.3, 627.9, 754.8, 760.8, 871.7, 1065, 1364, 1538, 2240, 2650, 3186, 4115, 6703),
        'q_norm': (12.51, 12.09, 11.59, 12.52, 13.38, 12.05, 11.5, 11.92, 11.27, 10.6, 11.83, 11.24, 8.284, 9.572, 12.28, 9.908, 8.765, 10.6, 9.736, 8.338, 9.999, 7.603, 9.146, 7.794, 8.021, 7.848, 6.152, 7.231, 7.322, 5.889, 7.231, 8.62, 5.979, 8.329, 7.005, 6.279, 6.868, 5.889, 6.36, 6.633, 6.025, 6.487, 6.27, 6.188, 5.898, 6.587, 5.507, 6.633, 6.469, 5.771, 5.707, 7.286, 6.515, 7.259, 6.388, 5.553, 7.694, 6.496, 6.778, 5.97, 8.719, 5.834, 7.776, 7.776, 5.816, 8.883, 9.019, 8.284, 7.948, 8.375, 8.22, 8.629, 7.975, 8.955, 9.046, 8.701, 8.62, 9.572, 9.835, 9.754, 9.191, 10.13)}
}


colors={ #(112/255,48/255,160/255) # '#333'
    0 : (237/255,28/255,46/255), # model
    1 : (0/255,142/255,194/255), # NPRA blue
    2 : (93/255,184/255,46/255), # NPRA green
    5 : (255/255,150/255,0/255), # NPRA orange

    'mesh':(0,0,0,0.4)
}


markers={
    1 : "o",
    2 : "^",
    5 : "s"
}


# returns tick labels to simulate log-scale in linear space for 3D plots
# see in comments here: https://stackoverflow.com/questions/3909794/plotting-mplot3d-axes3d-xyz-surface-plot-with-log-scale
def log_tick_formatter( val, pos=None ):
    return f"$10^{{{val:g}}}$"


def calc_model( OCR, V ):
    # returns values for model fitted to experimental data
    V_0 = 30
    m = 0.4

    # a, b, c & e defined as functions of OCR
    a = 0.08 * OCR + 0.55
    b = -0.05 * OCR + 2
    c = 5.6 / OCR**0.9
    e = 12 * OCR - 6

    return ( a + b/(1+c*V) ) * (e * (1+np.arcsinh( V/V_0 )))**m


def create_figure():
    axis_label_size = 15
    n_xy = 1000
    V_min, V_max = 0.001, 100000
    OCR_min, OCR_max = 1, 8

    # define calculation grid
    grid_V = np.logspace( np.log10(V_min), np.log10(V_max), n_xy )
    grid_OCR = np.linspace( OCR_min, OCR_max, n_xy )
    Vs, OCRs = np.meshgrid( grid_V, grid_OCR )

    q_norms = calc_model( OCRs, Vs )  # calculate model parameters and model values (on grid)

    fig, ax = plt.subplots( subplot_kw={"projection": "3d"}, figsize=(15.04,12.00) )

    # draw surface wireframe,  rstride & cstride are set so that wirenet matches axis major axis
    surf = ax.plot_wireframe( np.log10( Vs ), OCRs, q_norms, rstride=143, cstride=125, color=colors['mesh'], linewidth=1)

    # draw datapoints and selected 3D lines
    first_line = True # used to label a single 3D line
    for ocr in t_bar_dataset:
        line_q_norm = calc_model( ocr, grid_V ) # 3D line data
        line_ocr = grid_V*0 + ocr # generate ocr values of same shape
        
        label='Proposed model' if first_line else ''
        ax.plot( np.log10(grid_V), line_ocr, line_q_norm, lw=3, c=colors[0], label=label )
        first_line = False # only label first line

        # draw test data
        ax.scatter3D(
                    np.log10(t_bar_dataset[ocr]['V']), 
                    ocr, 
                    t_bar_dataset[ocr]['q_norm'],
                    label='Test data OCR=' + str(ocr),
                    s=50,
                    marker=markers[ocr],
                    ec=(0,0,0),
                    fc=colors[ocr],
                    alpha=1, # no fading
                    zorder=10
                    )

    # update tick values to mimic log-scale ( log values drawn on linear scale )
    ax.xaxis.set_major_formatter( mticker.FuncFormatter(log_tick_formatter) )
    ax.xaxis.set_major_locator( mticker.MaxNLocator(integer=True) )

    # set limits
    ax.set_xlim( np.log10(V_min)*0.95, np.log10(V_max) )
    ax.set_ylim( OCR_min, OCR_max*0.98 )
    ax.set_zlim( 0, 15 )

    # annotate axis
    ax.set_xlabel( r'$V=\frac{v \cdot d}{c_h} (-)$', fontsize=axis_label_size )
    ax.set_ylabel('OCR (-)', fontsize=axis_label_size)
    ax.set_zlabel( r"$\frac{q_{T-bar}}{ \sigma  ' _V }$ (-)", fontsize=axis_label_size)

    #ax.legend()
    fig.savefig('lehane_et_al_3d.png', dpi=600)
    plt.show()


# runs in current module. called last -> all functions loaded
if __name__=='__main__':
    create_figure()