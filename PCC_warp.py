from copy import deepcopy
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.backend_bases import MouseButton
from dtw_ex import simulated_push_resistance, distort_d, restore_d # continue with example from DTW
np.random.seed(seed=1234) # for reproduceability of dtw comparison example


class PCC_warp():
    """
    PCC_warp
    ================

    attempts to find a warp for a curve, while striving to maximize pearsons r increment in each step

    The algorithm has two main components
        nudge_warp_points(): modifies current warp by moving existing warp points
        add_warp_points(): adds new warp points between existing points

    both components are run at each iteration, and the one giving a larger increase in r is selected.

    Initialization arguments
    ------------------------
        n - maximum number of alignment points (default: 10)
        threshold - minimum increase in r for a change to be accepted (default: 1e-5)
        n_increments - number of increments to divide range between warp points into (default: 50)
        n_max_iterations - maximum number of iterations allowed (default: 100)
        max_shift - largest warp allowed for a single point (default: 999)
        proximity_limit - minimum distance between warp points (both from and to) (default: 0.2)
        discount - penalize long warps in seelction profess (default: True)

        filter_largest_above_percent - only evaluate (discounted) scores within this % of maximum (default: 0.99)
        increment_divider - nudge recursion control variable, nth recursion divides the interval with x^n (default: 3)
        precision - number of decimals in warp definition (default: 2)

    Functions
    ---------
        fit_path()
            arguments:
                curve - curve to be warped ( [x_value_list, y_value_list] )
                reference - curve to warp to (same format)

            returns:
                path describing warp from curve to reference

            stores:
                curve and reference, warp & score history

        plot_history()
            arguments:
                file_name - string with name of file (ex. 'anim.gif') to save animation. No name -> not saved (default: '')
                frame_ms - number of milliseconds for each frame in animation

            returns: None

            displays: animation

        get_warp()
            arguments: none

            returns:
                warp definition as a list of tuples [ (x_from, x_to)_0, ..., (x_from, x_to)_n ]

        get_warp_history()
            arguments: none

            returns:
                list of warp definitions for each step in the algorithm

        inverse_path()
            arguments: none

            returns path describing warp from reference to curve
    """
    def __init__( 
            self,
            n=10,
            threshold=1e-5,
            n_increments=50,
            n_max_iterations = 100,
            proximity_limit = 0.2,
            max_shift = 999,
            discount = True,
            scale_shift_curve = True,
            filter_largest_above_percent = 0.98,
            increment_divider = 3,
            precision = 3,
            score_visuals=0, # True/False or nr. of iterations to visualize
        ):

        self.n = n
        self.threshold = threshold
        self.n_increments = n_increments
        self.n_max_iterations = n_max_iterations
        self.proximity_limit = proximity_limit
        self.max_shift = max_shift,
        self.filter_largest_above_percent = filter_largest_above_percent # % of r_max discounting keeps
        self.increment_divider = increment_divider # nudge recursion search area reduction
        self.precision = precision # decimals in nudge/new_point
        self.min_incr = 1/10**self.precision # smallest allowed increment
        self.discount = discount
        self.base_discount_x = 0.5
        self.fixed_ends = True # first/last warp fixed by user
        self.scale_shift_curve = scale_shift_curve
        self.scores_visuals = score_visuals

        self.reset()


    def reset( self ):
        self.warp_history=[] # list of warp states (current warp is last)
        self.r_s = [] # list of each acheived r
        self.initial_warp = None
        self.final_warp = None
        self.click_first = True


    def warp_settings( self, curve_settings, reference_settings ):
        self.curve_settings = curve_settings
        self.reference_settings = reference_settings


    def apply_settings( self ):
        if not hasattr(self, 'curve_settings'): return
        c_settings = self.curve_settings
        if 'initial_warp' in c_settings: self.initial_warp = c_settings['initial_warp']
        if 'final_warp' in c_settings: self.final_warp = c_settings['final_warp']


    def trim_to_def( self, curve, definition_range ):
        if len(definition_range)>0:
            d_from = definition_range[0]
            d_to = definition_range[1]

            depths = curve[0]
            vals = curve[1]

            # simple filter, ignores range definition points
            return [ depths[(depths >= d_from) & (depths <= d_to)], vals[(depths >= d_from) & (depths <= d_to)] ]
        return curve


    def fit_path( self, curve, reference, warp=None ):
        self.reset()
        self.curve = curve
        self.reference = reference
        self.apply_settings() # trim to definition, inital warp, ignore depths
        self.scaler()

        self.warp_history.append( [(self.curve[0][0],self.curve[0][0])] )
        self.r_s.append( self.warped_r(self.warp_history[0] ) ) # book-keeping

        if not warp:
            warp = [ (curve[0][0],reference[0][0]), (curve[0][-1],reference[0][-1]) ]
            if hasattr(self, 'initial_warp') and self.initial_warp and len(self.initial_warp)>0:
                warp = self.initial_warp

        self.warp_history.append( warp )
        self.r_s.append( self.warped_r(self.warp_history[-1] ) )
        self.r = self.warped_r( warp )

        center_x_dist = abs( self.center_x(self.reference)-self.center_x(self.curve) )
        #self.base_discount_x = max( center_x_dist, 1 )

        if self.final_warp: #early exit
            print('final warp for curve given as:', self.final_warp)
            return self.final_warp

        for _ in range(self.n_max_iterations):
            nudge_warp, nudge_r_inc = self.nudge_warp_points()
            warp, r_inc = self.add_warp_points()
            if nudge_r_inc>r_inc: # select better warp
                warp, r_inc = nudge_warp, nudge_r_inc

            if r_inc>self.threshold: # accept change
                self.warp_history.append( warp )
                self.r = self.warped_r( warp )
                self.r_s.append( self.r )
            else: # no/little improvement
                break
        return self.generate_path( self.warp_history[-1] )


    def generate_path( self, warp, inversed=False ):
        warped_x, ref_x = self.warp_depth( self.curve[0], warp ), self.reference[0]
        if inversed: # reverse curves
            warped_x, ref_x = self.reference[0], self.warp_depth( self.curve[0], warp )

        path = []
        for i in range(len(warped_x)):
            closest_index = np.argmin( np.absolute(ref_x-warped_x[i]) )
            path.append( ( i, closest_index ) )
        return path


    def inverse_path( self ):
        return self.generate_path( self.warp_history[-1], inversed=True )


    def get_warp( self ):
        return self.warp_history[-1]


    def get_warp_history( self ):
        return self.warp_history


    def apply_warp( self, curve, warp ):
        new_x = self.warp_depth( curve[0], warp )
        curve[0] = new_x

        return curve


    def center_x( self, curve ): # returns center of curve x-definition
        return (curve[0][-1]+curve[0][0])/2


    def scaler( self, q_range=(5.0, 95.0) ):
        if not self.scale_shift_curve: return
        curve_IQR = np.percentile(self.curve[1], q_range[1]) - np.percentile(self.curve[1], q_range[0])
        ref_IQR = np.percentile(self.reference[1], q_range[1]) - np.percentile(self.reference[1], q_range[0])

        curve_median = np.median( self.curve[1] )
        ref_median = curve_median = np.median( self.reference[1] )

        self.curve[1] -= curve_median
        self.curve[1] *= ref_IQR/curve_IQR
        self.curve[1] += ref_median


    def add_warp_points( self ):
        delta_scores = {}
        warps = {}

        warp = self.warp_history[-1]
        best_warp, r_change = warp.copy(), 0 # no change is default return

        # only add up to selected number of points
        if len( warp ) >= self.n: return best_warp, r_change

        for i in range( len(warp)-1 ):
            w_froms = [x[0] for x in warp]
            w_tos   = [x[1] for x in warp]

            x_from = self.get_x_add_range( w_froms, i )
            x_to = self.get_x_add_range( w_tos, i )

            for xf in x_from:
                for xt in x_to:
                    if abs(xf-xt)>self.max_shift: continue
                    w_p = ( xf, xt ) # current warp point
                    tmp_warp = warp[:i+1] + [ w_p ] + warp[i+1:]
                    tmp_r = self.warped_r( tmp_warp )

                    if (tmp_r-self.r) > self.threshold: # improvements above threshold
                        delta_scores[ w_p ] = tmp_r - self.r # track warp, score_change & index
                        warps[ w_p ] = tmp_warp

        if delta_scores: # found improvements
            disc_delta_scores = self.discount_scores( delta_scores ) # discounted by warp distance
            w = max( disc_delta_scores, key=disc_delta_scores.get ) # best discounted improvement
            best_warp = warps[ w ]
            r_change = delta_scores[ w ]

        return best_warp, r_change


    def where_to_warp( self, x_from, curve, reference, warp ):
        x_froms = np.array( [w[0] for w in warp] )
        x_tos = np.array( [w[1] for w in warp] )

        id_to = np.searchsorted(x_froms, x_from, side = 'right')

        self.curve = self.trim_xy( curve, x_froms[id_to-1], x_froms[id_to] )
        self.reference = self.trim_xy( reference, x_tos[id_to-1], x_tos[id_to] )

        delta = (x_tos[id_to]-x_tos[id_to-1])/4
        center = (x_tos[id_to]+x_tos[id_to-1])/2

        x_to_rng = np.round(np.linspace(center-delta,center+delta, 500),3)
        scores = {}
        warps = {}

        for tmp_x_to in x_to_rng:
            w = (x_from,tmp_x_to)
            tmp_warp = warp[:id_to] + [w] + warp[id_to:]
            scores[w] = self.warped_r(tmp_warp)
            warps[w] = tmp_warp

        best = max( scores, key=scores.get )

        print(best, scores[best] )

        return warps[best]


    def get_x_add_range( self, x_list, i ):
        x_low = x_list[i] + self.proximity_limit
        x_high = x_list[i+1] - self.proximity_limit

        if (x_high-x_low)/self.min_incr<self.n_increments:
            tmp_range = np.arange(x_low,x_high, self.min_incr) # use if fewer
        else:
            tmp_range = np.linspace(x_low,x_high, self.n_increments)
        return np.round( tmp_range, self.precision )


    def nudge_warp_points( self, warp=None, r_change=0, j=-2, n=0 ):
        delta_scores = {}
        indexes = {}

        if not warp:
            warp = self.warp_history[ -1 ]
            r_change = 0 # no change for current warp
        best_warp = warp.copy()

        for i in range( len(warp) ):
            if j>-1: i=j # recursion loop

            if self.fixed_ends:
                if i==0: continue # short-circuit first & last
                if i==len(warp)-1: continue

            w_froms = [x[0] for x in warp]
            w_tos   = [x[1] for x in warp]

            x_from = self.get_x_nudge_range( w_froms, i, n )
            x_to = self.get_x_nudge_range( w_tos, i, n )

            # check if either set is empty or negative 
            if x_from.size==0 or x_to.size==0: break
            if ( x_from.max()-x_from.min() ) <= self.min_incr: break 
            if ( x_to.max()-x_to.min() ) <= self.min_incr: break

            for xf in x_from:
                for xt in x_to:
                    if abs(xf-xt)>self.max_shift: continue
                    tmp_warp = warp.copy()
                    w_p = ( xf, xt ) # current warp point
                    tmp_warp[i] = w_p
                    tmp_r = self.warped_r( tmp_warp )

                    if (tmp_r-self.r) > self.threshold: # improvements above threshold
                        delta_scores[ w_p ] = tmp_r - self.r # track warp, score_change & index
                        indexes[ w_p ] = i

            if j>-1: break # recursion loop

        if delta_scores: # found improvements, select best and recurse
            self.visualize_scores( delta_scores, title='Incremental score, ' + r'$\Delta r$' )
            disc_delta_scores = self.discount_scores( delta_scores ) # discounted by warp distance
            self.visualize_scores( disc_delta_scores, title='Discounted incremental score, ' + r'$\Delta r _d$' )

            reduced_disc_delta_scores = self.vals_close_to_max( disc_delta_scores ) # out of the 1-2% top discounted scores
            top_candidates = { k: delta_scores[k] for k in reduced_disc_delta_scores.keys() }
            w = max( top_candidates, key=top_candidates.get ) # find the best un-discounted improvement

            best_warp[ indexes[w] ] = w # construct best nudged warp
            best_warp, r_change = self.nudge_warp_points( warp=best_warp, r_change=delta_scores[w], j=indexes[w], n=n+1 )

        return best_warp, r_change


    def get_x_nudge_range( self, x_list, i, n ):
        x_cen = x_list[i]

        x_under = x_list[i-1] + self.proximity_limit # close to point below
        x_above = x_list[i+1] - self.proximity_limit # close to point above

        r_low = ( x_cen-x_under ) / (self.increment_divider**n) # increments 1 order smaller for each increment
        r_high = ( x_above-x_cen ) / (self.increment_divider**n)

        x_low = x_cen - r_low
        x_high = x_cen + r_high

        if (x_high-x_low)/self.min_incr<self.n_increments:
            tmp_range = np.arange( x_low,x_high, self.min_incr ) # use if fewer
        else:
            tmp_range = np.linspace( x_low,x_high, self.n_increments )
        return np.round( tmp_range, self.precision )


    def vals_close_to_max( self, some_dict ):
        if not some_dict: return some_dict # empty
        threshold = max( some_dict.values() ) * self.filter_largest_above_percent
        res = { k: v for k,v in some_dict.items() if v>threshold }
        return res


    def discount_scores( self, scores ):
        if not self.discount: return scores
        res = { k: v/( abs(k[0]-k[1]) + self.base_discount_x) for k, v in scores.items() }
        return res


    def trim_xy( self, curve, start, end ):
        mask = np.where( np.logical_and(curve[0]>=start, curve[0]<=end) )
        x_mask = curve[0][mask]

        desired_x = np.unique( np.concatenate( ([start], x_mask, [end]) ) )
        desired_y = np.interp( desired_x, curve[0], curve[1] )
        return ( desired_x, desired_y )


    def warp_depth(self, x, warp ):
        x_fr = [ x[0] for x in warp]
        delta = [ x[1]-x[0] for x in warp]
        delta_x = np.interp( x, x_fr, delta)
        x_warped = x + delta_x
        return x_warped


    def warped_r( self, warp):
        x_warped = self.warp_depth( self.curve[0], warp )

        x_min = max( min(x_warped), min(self.reference[0]) )
        x_max = min( max(x_warped), max(self.reference[0]) )

        # adds endpoints from warp definition to curve/reference
        tmp_curve = self.trim_xy( [x_warped, self.curve[1]], x_min, x_max )
        tmp_ref = self.trim_xy( self.reference, x_min, x_max )

        all_x_s = np.unique( np.concatenate( (x_warped, tmp_ref[0]) ) ) # combine all x-s:  curve[0] potentially outside area of interest
        ref_resampled = np.interp( all_x_s, tmp_ref[0], tmp_ref[1] ) # resample reference
        curve_resampled = np.interp( all_x_s, tmp_curve[0], tmp_curve[1] ) # resample curve
        r, _ = scipy.stats.pearsonr( curve_resampled, ref_resampled ) # calculate r
        return r


    def plot_xy( self, x, y ):
        fig,ax = plt.subplots()
        ax.plot(x,y)
        plt.show()


    def visualize_scores( self, scores, title='' ):        
        if not bool( self.scores_visuals ): return
        self.scores_visuals = int( self.scores_visuals ) - 1
        
        fontsize = 20
        tick_l_size = 18
        
        
        x_from = []
        x_to = []
        xx_vals = []
        for k, v in scores.items():
            x_from.append( k[0] )
            x_to.append( k[1] )
            xx_vals.append( v )

        plt.rcParams.update({
            'grid.color': '1',
            'grid.linestyle': '--',
        })

        fig, ax = plt.subplots( figsize=(10, 6) )

        n_int = 100
        xi, yi = np.linspace( min(x_from), max(x_from), n_int), np.linspace(min(x_to), max(x_to), n_int )
        xi, yi = np.meshgrid( xi, yi )

        rbf = scipy.interpolate.Rbf( x_from, x_to, xx_vals, function=[ 'gaussian', 'inverse', 'linear', 'cubic', 'quintic', 'thin_plate', 'multiquadric'][1] ) # 
        fi = rbf( xi, yi )
        cen_map = ax.pcolormesh( xi, yi, fi )
        p = ax.scatter( x_from, x_to, s=50, fc=(1,1,1), ec=(0,0,0) )
        cb = plt.colorbar( cen_map ) # draw legend
        cb.ax.tick_params( labelsize=tick_l_size )
        #cb.ax.set_ylabel( 'Incremental increase in Pearson\'s r', fontsize=fontsize )
        ax.set_xlabel( 'Depth from, s (m)', fontsize=fontsize)
        ax.set_ylabel( 'Depth to, t (m)', fontsize=fontsize )
        if title != '': plt.title( title, fontsize=fontsize )
        ax.tick_params( labelsize=tick_l_size ) # top=True, labeltop=True, bottom=False, labelbottom=False, 
        plt.grid()
        plt.show()


    def plot_warp( self, curve, reference, warp, delta=0, id_curve='', drop=[], filename='' ):
        fig, ax = plt.subplots(2,1, sharex=True, sharey=True,figsize=(20, 12))
        x_warped = self.warp_depth( curve[0], warp )

        marker = '.'
        marker_size = 3

        curve[1] += delta

        # original curves
        ax[0].plot( reference[0], reference[1], ls='-', marker=marker, markersize=marker_size, lw=2, zorder=-1, label='reference', c=(0/255,142/255,194/255)) # NPRA Blue
        ax[0].plot( curve[0], curve[1], ls='-', marker=marker, markersize=marker_size, lw=1, label='curve', c=(93/255,184/255,46/255)) # NPRA green

        # warped curve
        ax[1].plot( reference[0], reference[1], ls='-', marker=marker, markersize=marker_size, lw=2, zorder=-1, label='reference', c=(0/255,142/255,194/255))
        ax[1].plot( x_warped, curve[1], ls='-', marker=marker, markersize=marker_size, lw=1, label='curve', c=(93/255,184/255,46/255))

        # connection points
        x_from = [w[0] for w in warp]
        x_to = [w[1] for w in warp]
        y_vals = [ np.interp( w[0], curve[0], curve[1]) for w in warp ]
        c=[ (0,0,0,1) ] * len(x_from)
        ax[0].scatter( x_from, y_vals, s=20, c=c ) # y_s
        ax[1].scatter( x_to, y_vals, s=20, c=c ) # y_s

        # connection lines
        for w in warp:
            y_val = np.interp( w[0], curve[0], curve[1])
            con = ConnectionPatch( xyA=[ w[0],y_val ], xyB=[ w[1],y_val ], coordsA="data", coordsB="data", 
                                         axesA=ax[0], axesB=ax[1], color=(.4,.4,.4), lw=0.5, ls='-')
            ax[1].add_artist(con) # add hidden connections

        ax[0].set_xlabel( 'original x values' )
        ax[0].set_ylabel( 'y values' )
        ax[1].set_xlabel( 'warped x values' )
        ax[1].set_ylabel( 'y values' )

        plt.legend()

        # add text:
        ax[0].text(.6, .98, 'aligning: ' + id_curve + ' (' + str(len(warp)) + ' points)', ha='left', va='top', transform=ax[0].transAxes)

        # fix y_range
        y_range = ax[0].get_ylim()
        ax[0].set_ylim(y_range)
        ax[1].set_ylim(y_range)

        # add drop regions
        for d in drop:
            ax[1].add_patch( Rectangle( (d[0], y_range[0]), d[1]-d[0], y_range[1]-y_range[0], facecolor = (1,0,0,0.075), fill=True) )


        def on_click( event ): # add mouse_click hook
            if event.button is MouseButton.LEFT and event.inaxes == ax[0]: #only in upper diagram
                #if self.measure_mode:
                #print( f'data coords {np.round(event.xdata, self.precision)} {np.round(event.ydata, self.precision)}' )
                val = str(np.round(event.xdata, self.precision))
                if self.click_first:
                    print( '(' + val, end=',' )
                else:
                    print( val + '),' )
                self.click_first = not self.click_first

        def on_key( event ):
            self.key_press( event.key )
            #print('you pressed', event.key, event.xdata, event.ydata)

        plt.connect('button_press_event', on_click)
        cid = fig.canvas.mpl_connect('key_press_event', on_key)

        if filename:
            plt.savefig( filename )
        else:
            plt.show()


    def key_press( self, key ):
        print(key)
        if key=='m':
            self.measure_mode = not self.measure_mode


    def plot_history( self, file_name='', frame_ms=1000 ):
        fig = plt.figure(layout=None, figsize=(10, 6))
        gs = fig.add_gridspec(nrows=2, ncols=3, left=0.08, right=0.94, hspace=0.35, wspace=0.35 )

        ax = []
        ax.append( fig.add_subplot(gs[:-1, :-1]) )
        ax.append( fig.add_subplot(gs[-1, :-1]) )
        ax.append( fig.add_subplot(gs[:-1, -1]) )
        ax.append( fig.add_subplot(gs[-1, -1]))

        max_n = max([len(x) for x in self.warp_history]) # maximum allowed n_points

        # create data to plot in animations
        X = [] # states (only X-coordinates change)
        X_shift = []
        PF = [] # points from
        PT = [] # points to
        WP = [] # warp points
        A = [] # arrays of alphas (hide/show warps)
        C = [ 0 ] * max_n # black

        for i in range(len(self.warp_history)):
            warp = self.warp_history[i]

            x_warped = self.warp_depth( self.curve[0], warp ) # warped x-curve
            X.append( x_warped )
            X_shift.append( x_warped - self.curve[0] )

            alphas = [ 0 ] * max_n # hidden
            x_from, x_to = [ 0.0 ] * max_n, [ 0.0 ] * max_n
            for j in range(len(warp)): # overwrite used items for each warp
                alphas[j] = 1 # visible
                x_from[j] = warp[j][0]
                x_to[j] = warp[j][1]

            A.append(alphas)

            xf_warped = self.warp_depth( x_from, warp )

            y_s = [ min(min(self.curve[1]),min(self.reference[1])) ] * max_n
            PF.append( np.array([ [x, y] for x,y in zip(x_from,y_s) ]) )
            PT.append( np.array([ [x, y] for x,y in zip(x_to,y_s) ]) )
            WP.append( np.array([ [ xw, xt-xf] for xf,xt,xw in zip(x_from,x_to,xf_warped) ]) )

        # original curves
        ax[0].plot( self.reference[0], self.reference[1], lw=2, zorder=-1, label='reference', c=(0/255,142/255,194/255)) #NPRA Blue
        ax[0].plot( self.curve[0], self.curve[1], lw=1, label='curve', c=(93/255,184/255,46/255)) #NPRA green
        ax[0].set_xlabel( 'original x values' )
        ax[0].set_ylabel( 'y values' )

        y_range = ax[0].get_ylim() # fix boundary for warp points
        ax[0].set_ylim(y_range)

        # animated warp plot
        ax[1].plot( self.reference[0], self.reference[1], lw=2, zorder=-1, label='reference', c=(0/255,142/255,194/255))
        anim_curve = ax[1].plot( X[0], self.curve[1], lw=1, label='curve', c=(93/255,184/255,46/255))[0]
        ax[1].set_xlabel( 'original and warped x values' )
        ax[1].set_ylabel( 'y values' )
        ax[1].set_ylim(y_range)

        # r-and n curves
        steps = np.arange( len(self.warp_history) )
        ns = [ len(w) for w in self.warp_history ]

        ax[2].plot( steps, self.r_s, label='reference', c=(237/255,28/255,46/255)) # NPRA red
        ax[2].tick_params(axis='y', labelcolor=(237/255,28/255,46/255))
        ax[2].set_ylabel( 'Pearson\'s r (-)', c=(237/255,28/255,46/255))
        ax[2].set_xlabel( 'step (-)' )
        ax[2].set_xlim( 0, len(self.warp_history)-1 )
        ax[2].set_ylim( -1, 1 )

        n_ax = ax[2].twinx()
        n_ax.step( steps, ns, where='post', label='reference', c=(0,0,0))
        n_ax.set_ylabel( 'warp points (-)' )
        n_ax.set_ylim( 1, max_n+1 )

        # status curve
        anim_status = ax[2].plot([0,0],[-1,1], ls='--', zorder=-1, c=(.7,.7,.7) )[0]

        # warp curves
        ax[3].plot( self.reference[0], self.reference[0]*0, lw=2, zorder=-1, label='reference', c=(0/255,142/255,194/255) )
        anim_warp = ax[3].plot( self.curve[0]+X_shift[1], X_shift[0], lw=1, label='curve', c=(93/255,184/255,46/255))[0]

        max_s = max([max(x) for x in X_shift])
        min_s = min([min(x) for x in X_shift])
        max_s = max(max_s, -min_s) * 1.15
        max_s = np.round( max_s )
        ax[3].set_ylim( -max_s,max_s )
        ax[3].set_xlim( min(self.reference[0]), max(self.reference[0]) )

        ax[3].set_xlabel( 'warped x' )
        ax[3].set_ylabel( 'depth shift' )

        y_s = [y_range[0]*1.1] * max_n

        # add hidden warp points
        anim_pts_from = ax[0].scatter( PF[0][:,0], PF[0][:,1], s=20, c=C, alpha=0 )
        anim_pts_to = ax[1].scatter( PT[0][:,0], PT[0][:,1], s=20, c=C, alpha=0 )
        anim_pts_warp = ax[3].scatter( WP[0][:,0], WP[0][:,1], s=20, c=C, alpha=0, zorder=999)

        w_min = min([min(w[:,1]) for w in WP])
        w_max = max([max(w[:,1]) for w in WP])

        ax[3].set_ylim( w_min, w_max )

        # warp lines:
        con = []
        for c in range(max_n):
            xy_from, xy_to = [0, y_range[0]-1], [0, y_range[0]-1]
            if c<len(self.warp_history[0]):
                xy_from= [ self.warp_history[0][c][0],y_s[0] ]
                xy_to = [ self.warp_history[0][c][1],y_s[0] ]

            con.append( ConnectionPatch( xyA=xy_from, xyB=xy_to, coordsA="data", coordsB="data", 
                                         axesA=ax[0], axesB=ax[1], color=(.4,.4,.4), lw=0.5, ls='-', alpha=0) )
            ax[1].add_artist(con[-1]) # add hidden connections

        def animate(i): # handles updates for each frame
            anim_curve.set_xdata( X[i] ) # original
            anim_warp.set_xdata( X[i] ) # warped
            anim_warp.set_ydata( X_shift[i] ) # depth shift
            anim_status.set_xdata( [i,i] ) # vertical line

            # warp points in time series graphs
            anim_pts_from.set_offsets( PF[i] )
            anim_pts_from.set_alpha( A[i] )
            anim_pts_to.set_offsets( PT[i] )
            anim_pts_to.set_alpha( A[i] )

            # warp points in depth shift graph
            anim_pts_warp.set_offsets( WP[i] )
            anim_pts_warp.set_alpha( A[i] )

            # warp lines
            for c, pf, pt, a in zip( con, PF[i], PT[i], A[i] ):
                c.set_alpha(a)
                c.xy1 = pf
                c.xy2 = pt

        anim = FuncAnimation( fig, animate, interval=frame_ms, frames=len(X) )

        # add legend, try to keep it out of the way
        if self.curve[0][0]<self.reference[0][0]: # shifted right
            if self.curve[1][0] < (y_range[0]+y_range[1])/2:
                ax[1].legend(loc='upper left')
            else: ax[1].legend(loc='lower left')
        else:
            if self.curve[1][-1] < (y_range[0]+y_range[1])/2:
                ax[1].legend(loc='upper right')
            else: ax[1].legend(loc='lower right')

        if file_name: anim.save( file_name )
        plt.show()


def advanced_example_data( noise=0.1, rise=1.2, offset=0, d_from=4, d_to=12 ):
    # 3 different curves
    x  = simulated_push_resistance( rise=rise, const=offset, rand_fact=0, bulge=0 )
    y1 = simulated_push_resistance( rise=rise, const=offset, rand_fact=noise, bulge=0 )
    y2 = simulated_push_resistance( rise=rise, const=offset, rand_fact=noise, bulge=1 )
    d = np.linspace( d_from, d_to, len(x) ) # 4-12 m

    # distort example curve depths
    y1_dist, d1_dist, delta_1 = distort_d( y1, d, scale=0.1, length=.85 )
    y2_dist, d2_dist, delta_2 = distort_d( y2, d, scale=0.2, length=1.50 )

    return d, x, d1_dist, y1_dist, d2_dist, y2_dist, delta_1, delta_2


def depth_adjustment_example( simple_example=True, score_visuals=False ):
    if simple_example:
        if False: # old
            d = np.array( [ 4, 8, 12 ] )
            x = np.array( [ 0.5, 2, 1 ] )
            d1_dist = np.array( [ 2, 4, 10 ] )
            y1_dist = np.array( [ 0.5, 2, 1 ] )
        else: # new example
            d = np.array( [ 4, 8, 12 ] )
            x = np.array( [ 500.0, 1000, 600 ] )
            d1_dist = np.array( [ 2, 4, 9 ] )
            y1_dist = np.array( [ 520.0, 1100, 680 ] )
            curve = [ d1_dist, y1_dist ]
    else: # more complex curves
        d, x, d1_dist, y1_dist, d2_dist, y2_dist, _, _ = advanced_example_data()

        # create curve in [x,y] format
        curve = [ d1_dist, y1_dist ]
        curve = [ d2_dist, y2_dist ] # overwritten

    reference = [ d, x ]

    adjuster = PCC_warp(
        n=20,
        n_increments=20,
        threshold=1e-9,
        proximity_limit = .5,
        score_visuals= score_visuals,
    )
    path = adjuster.fit_path( curve, reference )
    print('alignment path', path)
    print('warp definition', adjuster.warp_history[-1])

    adjuster.plot_history( file_name='some_animation.gif', frame_ms=1000 )

    if simple_example:
        
        warps = adjuster.warp_history
        print(warps)


def dtw_comparison_example():
    import pickle
    import os
    f_name = 'PCC_warp_example_paths.pkl'

    d, x, d1_dist, y1_dist, d2_dist, y2_dist, delta_1, delta_2 = advanced_example_data()
    reference   = [ d, x ]
    red_curve   = [ d1_dist, y1_dist ]
    green_curve = [ d2_dist, y2_dist ]

    adjuster = PCC_warp(
        n=20,
        n_increments=20,
        threshold=1e-6,
        proximity_limit = 1,
        scale_shift_curve=True
    )

    if False:
        red_path = adjuster.fit_path( red_curve, reference )
        red_warp = adjuster.get_warp()

        green_path = adjuster.fit_path( green_curve, reference )
        green_warp = adjuster.get_warp()

        with open( f_name, 'wb' ) as f:
            pickle.dump( (red_path, red_warp, green_path, green_warp), f )
        print("paths saved to file")
    else: # save time on fits for plot_development
        with open( f_name, 'rb' ) as f:
            red_path, red_warp, green_path, green_warp = pickle.load( f )
        print("paths loaded from file")

    # restored depths from warp alignment
    d1_restored = restore_d( red_path, d, d1_dist )
    d2_restored = restore_d( green_path, d, d2_dist )

    # depth shifts
    d_0_shift = d-d
    d_1_shift = d1_restored-d1_dist
    d_2_shift = d2_restored-d2_dist

    # theoretical depth alignment
    d1_th_shift = d1_dist + delta_1
    d2_th_shift = d2_dist + delta_2

    # warp definitions
    red_p = [ wp[0] for wp in red_warp ]
    red_delta = [ wp[1]-wp[0] for wp in red_warp ]
    green_p = [ wp[0] for wp in green_warp ]
    green_delta = [ wp[1]-wp[0] for wp in green_warp ]

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
    axs[2].plot( d_1_shift, d1_dist, lw=1, c=(237/255,28/255,46/255), label='PCC_warp alignment' )
    axs[2].plot( d_2_shift, d2_dist, lw=1, c=(93/255,184/255,46/255), label='PCC_warp. w/bulge' )
    axs[2].plot( delta_1, d1_dist, ls='--', lw=1, c=(68/255,79/255,85/255), label='true shift' )  # dark grey NPRA
    axs[2].plot( delta_2, d2_dist, ls='--', lw=1, c=(68/255,79/255,85/255) )
    axs[2].plot( red_delta, red_p, ls='none', marker='o', markerfacecolor='none', lw=1, c=(68/255,79/255,85/255), label='warp definition' ) #markeredgecolor
    axs[2].plot( green_delta, green_p, ls='none', marker='o', markerfacecolor='none', lw=1, c=(68/255,79/255,85/255) )
    axs[2].annotate('large distortion', xy=(1.95,5.36), xycoords='data',
        xytext=(1.8,4.6), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[2].annotate('alignment follows\ntrue shift line\nfrom 7.2m depth', xy=(1.56,7.2), xycoords='data',
        xytext=(1.5,8.4), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[3].set_xlabel( 'PCC_warp alignment (kPa)' )
    axs[3].plot( x, d, lw=2, c=(0/255,142/255,194/255), zorder=-1 )
    axs[3].plot( y1_dist, d1_restored, c=(237/255,28/255,46/255), lw=1 )
    axs[3].plot( y2_dist, d2_restored, c=(93/255,184/255,46/255), lw=1 )
    axs[3].annotate('extra peak moved\n0.5m deeper', xy=(1000,7.56), xycoords='data',
        xytext=(750,6.7), textcoords='data', va='top', ha='left',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    axs[0].set_ylim( d[0], d[-1] )
    axs[2].set_xlim( -0.2, 3.8 )
    axs[0].invert_yaxis()
    axs[2].legend()

    for ax in axs:
        if ax.get_xlim()[1] > 100: ax.set_xlim( 0, ax.get_xlim()[1] )
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
    plt.show()


def visualize_simple( curve, reference, warps):
    # example data
    d = np.array( [ 4, 8, 12 ] )
    x = np.array( [ 500, 1000, 600 ] )
    d1_dist = np.array( [ 2, 4, 9 ] )
    y1_dist = np.array( [ 520, 1100, 680 ] )
    curve = [ d1_dist, y1_dist ]
    reference = [ d, x ]
    warps = [[(2, 2)], [(2, 4), (9, 12)], [(2, 4), (3.763, 7.447), (9, 12)], [(2, 4), (3.979, 7.983), (9, 12)]]

    
    # warped_r function adapted from adjuster class code
    def warped_r( curve, reference, warp, adjuster):
        x_warped = adjuster.warp_depth( curve[0], warp )

        x_min = max( min(x_warped), min(reference[0]) )
        x_max = min( max(x_warped), max(reference[0]) )

        # adds endpoints from warp definition to curve/reference
        tmp_curve = adjuster.trim_xy( [x_warped, curve[1]], x_min, x_max )
        tmp_ref = adjuster.trim_xy( reference, x_min, x_max )

        all_x_s = np.unique( np.concatenate( (x_warped, tmp_ref[0]) ) ) # combine all x-s:  curve[0] potentially outside area of interest
        ref_resampled = np.interp( all_x_s, tmp_ref[0], tmp_ref[1] ) # resample reference
        curve_resampled = np.interp( all_x_s, tmp_curve[0], tmp_curve[1] ) # resample curve
        r, _ = scipy.stats.pearsonr( curve_resampled, ref_resampled ) # calculate r
        return r, all_x_s, curve_resampled, ref_resampled

    # drawing settings
    title_fs = 13
    txt_fs = 11
    marker = 'o'
    ms = 6
    mec = (0,0,0)
    mew = 1
    x_lims = (1,16)
    delta_ylim = (-1,11)
    xticks = np.arange(2,15,2)
    ann = ['as measured', 'step 1: endpoint alignment', 'step 2: add warp point', 'step 3: nudge warp point']

    adjuster = PCC_warp(
        n=20,
        n_increments=20,
        threshold=1e-4,
        proximity_limit = 1,
        scale_shift_curve=True
    )

    fig, axs = plt.subplots( 2, len(warps), figsize=(14,6) )

    some_r = 0
    some_label = ''

    for i, some_warp in enumerate(warps):
        ref = deepcopy(reference)
        crv = deepcopy(curve)

        if i>0:
            #crv = adjuster.apply_warp( crv, some_warp )
            r, all_xs, crv_res, ref_res = warped_r( crv, ref, some_warp, adjuster )
            some_label = 'r=%4.3f' %r
            if i>1:
                some_label += '\n' + r'$\Delta$' + 'r=%4.3f' %(r-some_r)

            ref = [ all_xs, ref_res ]
            crv = [ all_xs, crv_res ]

            some_r = r
            # resample

        delta = [ w[1]-w[0] for w in some_warp ]
        s = [w[0] for w in some_warp]

        axs[0][i].plot( ref[0], ref[1], c=(93/255,184/255,46/255), marker=marker, ms=ms, mec=mec, mew=mew, label='reference', zorder=1 )
        axs[0][i].plot( crv[0], crv[1], c=(0/255,142/255,194/255), marker=marker, ms=ms, mec=mec, mew=mew, label='curve', zorder=3 )
        axs[1][i].plot( s, delta, c=(237/255,28/255,46/255), marker=marker, ms=ms, mec=mec, mew=mew, label='current solution' )

        #if i>0: axs[0][i].plot( old_crv[0], old_crv[1], c=( 0.7, 0.7, 0.7 ), ls='--', label='last step', zorder=0 )


        axs[0][i].text(0.03, 0.97, chr(ord('A')+i), fontsize=title_fs, color=(0,0,0), verticalalignment='top', horizontalalignment='left', transform=axs[0][i].transAxes)
        axs[1][i].text(0.03, 0.97, chr(ord('E')+i), fontsize=title_fs, color=(0,0,0), verticalalignment='top', horizontalalignment='left', transform=axs[1][i].transAxes)
        axs[1][i].text(0.5, 0.03, ann[i], fontsize=txt_fs, color=(0,0,0), verticalalignment='bottom', horizontalalignment='center', transform=axs[1][i].transAxes)
        axs[1][i].text(0.97, 0.97, some_label, fontsize=txt_fs, color=(0,0,0), verticalalignment='top', horizontalalignment='right', transform=axs[1][i].transAxes)


        if i==0:
            axs[0][i].set_ylabel( r'$q _c$' + ' (kPa)', fontsize=title_fs )
            axs[1][i].set_ylabel( 'Warp distance, ' + r'$\delta = t-s$' + ' (m)', fontsize=title_fs )
        axs[0][i].set_xlabel( 'Depth, d (m)', fontsize=title_fs )
        axs[1][i].set_xlabel( 'Depth from, s (m)', fontsize=title_fs )
        axs[0][i].set_xlim( x_lims )
        axs[1][i].set_xlim( x_lims )
        axs[0][i].set_xticks( xticks )
        axs[1][i].set_xticks( xticks )

        axs[1][i].set_ylim( delta_ylim )
        
        old_crv = deepcopy( curve )
        old_crv = adjuster.apply_warp( old_crv, some_warp )

    axs[1][i].plot( [4,8.999],[4,9], c=(.7,.7,.7), ls='--', zorder=0, label='alternate solution' ) # alternate warp 1
    axs[1][i].plot( [2.001,4],[4.4,4], c=(.7,.7,.7), ls='--', zorder=0, label='alternate solution' ) # alternate warp 1

    axs[0][i].legend()

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    depth_adjustment_example( simple_example=True, score_visuals=6 )
    visualize_simple( curve=None, reference=None, warps=None )
    #dtw_comparison_example()