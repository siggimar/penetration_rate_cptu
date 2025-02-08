from cpt import cpt
import os
import pickle
from smoothers import lowess, statm_loess
from cptu_rate_dataset import dataset
import matplotlib.pyplot as plt


class smoothers_example:
    def __init__(self) -> None:
        self.res = {}
        self.saves_folder = 'saves'
        self.data_file = 'smoothers_example_data.pkl'
        self.get_data()

    def get_data( self ):
        os.makedirs( self.saves_folder, exist_ok=True )
        file_path = os.path.join( self.saves_folder, self.data_file )

        if os.path.isfile( file_path ): # data saved
            with open( file_path, 'rb' ) as f:
                self.cpts = pickle.load( f )
        else: # read cpt_data
            self.cpts = {}
            for location in [ 0, 1, 2 ]:
                my_dataset = dataset()
                my_dataset.load_dataset( location=location, read_logfiles=True, from_file='dataset_test.pkl' ) # location= 0:Tiller-Flotten, 1:Øysand, 2:Halsen-Stjørdal
                self.cpts[location] = my_dataset.get_reference()
            with open( file_path, 'wb' ) as f:
                pickle.dump( self.cpts, f )

    def set_param( self, param ):
        self.data = {}
        self.param = param
        for k,v in self.cpts.items():
            self.data[k] = v.get_data_with_detph( param )


    def calc( self, smoothers, residuals='max' ):
        for location in self.data:
            depth, vals = self.data[location]
            loc_data = {}

            for smoother in smoothers:
                smoother.fit( depth, vals )
                pred = smoother.predict( depth )
                
                if residuals=='max':
                    res = smoother.max_abs_residuals()
                else:
                    res = smoother.std_abs_residuals()

                loc_data[ smoother.get_short_name() ] = {'x': depth, 'y': pred, 'z': res }

            self.res[ location ] = loc_data


    def plot( self ):
        fig, axs = plt.subplots( 1, 3, sharey=True, sharex=True )
        for loc in self.res:
            axs[loc].plot( self.data[loc][1], self.data[loc][0], label=self.cpts[loc].pos_name, zorder=999 )
            for model in self.res[loc]:
                data = self.res[loc][model]
                axs[loc].plot(data['y'], data['x'], label=model)

        axs[0].invert_yaxis()
        plt.legend()
        plt.show()
    

    def residual_plot( self ):
        locs = {0:'Tiller-Flotten', 1:'Øysand', 2:'Halsen'}
        fig, ax = plt.subplots()
        for loc in self.res:
            for m in self.res[loc]:
                ax.plot( self.res[loc][m]['z'], self.res[loc][m]['y'], marker='o', ls='None', label=locs[loc] )
        ax.set_xlabel( 'std_residuals ' + self.param )
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel( self.param )
        #ax.invert_yaxis()
        plt.legend()
        plt.show()



def general_comparison():
    smoother_tester = smoothers_example()
    param = ['qc','qt','fs','ft','u','du','qn','qe','du_sig_v0_eff','Qt','Bq','Fr','Rf'][1]
    smoother_tester.set_param( param )

    deltas = [ 0.05, 0.1, 0.15 ]

    smoothers = [ lowess( delta=d, iterations=20, deg=2) for d in deltas ]

    smoother_tester.calc( smoothers, residuals='std' )
    smoother_tester.plot()
    smoother_tester.residual_plot()




if __name__ == '__main__':
    general_comparison()