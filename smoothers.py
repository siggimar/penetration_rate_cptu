import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
import warnings
import matplotlib.pyplot as plt



class lowess():
    '''
    Lowess class based on methods described in Cleveland (1979): "Robust Locally Weighted Regression and Smoothing Scatterplots"

    Weighted polynomials calculated using numpy.

    Standard parameter f (or frac) exchanged for depth range parameter delta.
    i.e. fits are calculated using points within distance delta and not closest % of total points.

    Implementation inspired by lowess.py code provided by Alexandre Gramfort on github: https://gist.github.com/agramfort/850437
    '''
    def __init__( self, delta, deg=2, iterations=3 ):
        self.short_name = 'lowess'
        self.long_name = 'Robust Locally Weighted Regression'
        self.delta = delta
        self.w_bound = 1e-9 # ensure weights are never exactly 0
        self.deg = deg
        self.iterations = iterations
        warnings.simplefilter('ignore', np.RankWarning)


    def get_short_name( self ):
        return self.short_name + ' (delta=' + str(self.delta) + ', deg=' + str(self.deg) + ', it=' + str(self.iterations) +')'


    def tri_cubic( self, x ): # tricube function
        x_i = np.clip( np.abs(x), 0, 1-self.w_bound ) # ensure tri(n) ϵ [0,1]
        return ( 1 - x_i**3 )**3


    def bi_square( self, x ): # bisquare function
        x_i = np.clip( x, -1+self.w_bound, 1-self.w_bound ) # ensure bi(n) ϵ [0,1]
        return ( 1 - x_i**2 )**2


    def weighted_regression( self, x_k, x, y, weights ): # np function utilized for regression
        coeff = np.polyfit( x, y, deg=self.deg, w=weights )
        return np.polyval( coeff, x_k )


    def max_abs_residuals( self ):
        m_eps = np.zeros( shape=self.eps.shape )
        for i in range( len(self.x) ):
            abs_local_eps = np.abs( self.eps[ self.idx[i] ] )
            m_eps[i] = np.max( abs_local_eps )
        return m_eps


    def std_abs_residuals( self ):
        m_eps = np.zeros( shape=self.eps.shape )
        for i in range( len(self.x) ):
            local_eps = self.eps[ self.idx[i] ]
            abs_local_eps = np.abs( local_eps )
            m_eps[i] = np.std( local_eps )
        return m_eps


    def fit( self, x, y ): # x & y are 1D np.arrays
        self.smooth = True
        self.x, self.y = np.array( x ), np.array( y )

        # remove nans by index from both x&y
        nan_indexes = np.isnan( self.x ) | np.isnan( self.y )
        self.x, self.y = self.x[~nan_indexes], self.y[~nan_indexes]

        if max( self.y )==min( self.y ): 
            self.smooth=False
            return # constant -> no smoothing

        self.idx = []
        self.xi  = []
        self.yi  = []
        self.w   = []
        self.y_pred_0 = []
        h   = [] 

        for i in range(len(self.x)):
            self.idx.append( np.asarray( np.abs(self.x - self.x[i]) < self.delta ).nonzero()[0] ) # indexes of x for each x0 where |x-x0|<d            
            self.xi.append( self.x[ self.idx[-1] ] )
            self.yi.append( self.y[ self.idx[-1] ] )
            if len(self.xi[-1])==1: 
                self.smooth=False # need at least two point
                return

            # calculate x-weights (done once)
            h.append( np.max( np.abs(self.xi[-1] - self.x[i]) ) )
            x_rel = ( self.xi[-1]-self.x[i] ) / h[-1]
            self.w.append( self.tri_cubic(x_rel) )
            
            self.y_pred_0.append( self.weighted_regression(self.x[i], self.xi[-1], self.yi[-1], weights=self.w[-1]) )
        self.y_pred_0 = np.array(self.y_pred_0) # keep locally weighted regression


    def predict( self, _=None ): # predicted on input x
        y_pred = self.y_pred_0.copy() # start with locally weighted

        if not self.smooth: return y_pred # no smoothing -> return same array

        for it in range( self.iterations ):
            eps = self.y - y_pred   # all prediction residuals
            s = np.median( np.abs( eps ) )
            delta = self.bi_square( eps/(6*s) )

            for i in range( len(self.x) ):
                idx = self.idx[i] # local point id
                d_idx = delta[idx]
                if d_idx.max()==0 and d_idx.min()==0:
                    d_idx = np.ones(shape=d_idx.shape) * self.min_delta
                delta_wi = np.multiply( self.w[i], d_idx ) # update local weights
                y_pred[i] = self.weighted_regression( self.x[i], self.xi[i], self.yi[i], weights=delta_wi )

        self.eps = self.y - y_pred # keep residuals for analysis
        return y_pred # return robust locally weighted regression



# wrap statsmodel function implementation in a class for testing/comparison
class statm_loess():
    def __init__( self, frac=2/3, iterations=1 ):
        self.short_name = 'sm-lowess'
        self.long_name = 'Locally Weighted Scatterplot Smoothing'
        self.frac = frac
        self.iterations = iterations


    def get_short_name( self ):
        return self.short_name + ' (frac=' + str(round(self.frac,3)) + ', it=' + str(self.iterations) +')'


    def fit (self, x, y ):
        self.x = x
        self.y = y


    def predict( self, x ):
        return sm_lowess( self.y, self.x, frac=self.frac, it=self.iterations, xvals=x )


def lowess_example():
    deltas = [0.1, 0.5, 1]

    x = np.linspace( 0, 4*np.pi, 1000 )
    x_1 = x/(4*np.pi)*10 # change x_scale to 0-10m with 1cm intervals
    y = np.sin( x ) * 10 * x_1/max(x_1)    
    y = np.maximum(y,0)
    y_rising_trend = x_1 * 0.4
    
    noise = np.random.uniform( -1/1, 1/1, len(x))
    y_n = y+noise+y_rising_trend


    y_lowess = []
    for delta in deltas:
        smoother = lowess( delta=delta )
        smoother.fit( x_1, y_n )

        y_lowess.append( smoother.predict() )

    fig, ax = plt.subplots()
    ax.plot( x_1, y+y_rising_trend, label='base curve', lw=4, zorder=2)
    ax.plot( x_1, y_n, label='noisy curve', lw=1, c=(.8, .8, .8), zorder=1 )
    
    for delta,y_low in zip(deltas, y_lowess):
        ax.plot( x_1, y_low, label='lowess (' + r'$\Delta=$' + str(delta) + ')', zorder=3 )

    ax.set_ylabel('Example resistance curve (any unit)')
    ax.set_ylabel('Depth (m)')

    ax.legend()
    plt.show()


if __name__=='__main__':
    lowess_example()