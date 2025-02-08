import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
from cptu_classification_models_2d import model_defs

# defines CPTu SBT classification charts
# class used as background in figures 
# showing rate effects for CPTu in 
# selected charts

class general_model(): # defines & presents 2D calssification models
    def __init__( self, model_def, fill_regions=True ):
        self.x_lim = [np.inf,-np.inf] # plotting bound calculations
        self.y_lim = [np.inf,-np.inf]
        self.fill = fill_regions
        self.parse_raw_def( model_def )


    def parse_raw_def( self, model_def ):
        self.edge_color = ( 0, 0, 0 )
        self.edge_width = 1.2
        self.face_alpha = 1
        self.fontsize_titles = 16
        self.fontsize_ticks = 12
        self.fontsize_txt = 14 #14
        self.max_n = 12

        self.name = model_def['desc']['name']
        self.short_name = model_def['desc']['short_name']
        self.author = model_def['desc']['author']
        self.source = model_def['desc']['source']
        (self.x_var,self.log_x,) = model_def['desc']['x_axis']
        (self.y_var,self.log_y,) = model_def['desc']['y_axis']
        self.colors = { k: v for k, v in enumerate(model_def['desc']['colors']) }
        self.x_bounds = model_def['desc']['x_bounds']
        self.y_bounds = model_def['desc']['y_bounds']

        self.color_transform()
        self.log_x = self.log_x.lower()=='log'
        self.log_y = self.log_y.lower()=='log'

        self.regions = []

        for r in model_def['regions']:
            ( x, y ) = model_def['regions'][r]['xy']
            self.regions.append( 
                self.region_2d( 
                    parent=self, 
                    index=r,
                    name=model_def['regions'][r]['name'],
                    id_loc=model_def['regions'][r]['id_loc'],
                    x=x,
                    y=y
                ) 
            )


    def color_transform( self ):
        for c in self.colors:
            if isinstance( self.colors[c], tuple ):
                self.colors[c] = tuple( [some_rgba/255 for some_rgba in self.colors[c]] )

    def set_bounds( self, ax ):
        if len(self.x_bounds)>0:
            ax.set_xlim( self.x_bounds )
        else:
            ax.set_xlim( self.x_lim )
        if len(self.y_bounds)>0:
            ax.set_ylim( self.y_bounds )
        else:
            ax.set_ylim( self.y_lim )


    def show_legend( self ):
        self.legend()
        plt.tight_layout()
        plt.show()


    def legend( self ):
        self.leg_fig, self.leg_ax = plt.subplots( figsize=(4,5))

        dx, dy = 0.35, 0.35
        x = [-dx, dx, dx, -dx, -dx]
        for i, r in enumerate(self.regions):
            short_id = r.id_txt.split('/')[0] # cut long strings
            name = ' '.join(r.name.split(' ')[1:])

            y = [ -i-dy, -i-dy, -i+dy, -i+dy, -i-dy ]

            if self.fill: self.leg_ax.fill( x, y, facecolor=r.color, edgecolor='none', alpha= self.face_alpha, zorder=1)
            self.leg_ax.plot( x, y, c=self.edge_color, linewidth=self.edge_width, zorder=2 )
            t = self.leg_ax.text( 0, -i, short_id, verticalalignment='center', horizontalalignment='center', size=self.fontsize_ticks*.9, zorder=3 )
            label = self.leg_ax.text( 2*dx, -i, name, verticalalignment='center', horizontalalignment='left', size=self.fontsize_ticks, zorder=3 )
        
        Title = self.leg_ax.text( -dx, 1, self.author, verticalalignment='center', horizontalalignment='left', size=self.fontsize_txt, zorder=3 )

        self.leg_ax.set_xlim( [-1.5*dx, 25*dx] )
        self.leg_ax.set_ylim( [-(self.max_n-1)-1.5*dy, 1 + 1.5*dy] )

        self.leg_ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
        self.leg_ax.spines[:].set_visible(False)
        return self.leg_ax


    def plot( self, letter='' ):
        self.prep_figure( letter )
        
        
        plt.show()


    def prep_figure( self, letter='' ):
        self.fig, self.ax = plt.subplots()

        for r in self.regions:
            if self.fill: self.ax.fill( r.x, r.y, facecolor=r.color, edgecolor='none', alpha= self.face_alpha, zorder=1)
            self.ax.plot( r.x, r.y, c=self.edge_color, linewidth=self.edge_width, zorder=2 )

            if isinstance(r.id_loc[0],list):
                (a_x, a_y), (t_x, t_y), = r.id_loc
                t = self.ax.text( t_x, t_y, r.id_txt, verticalalignment='bottom', horizontalalignment='center', size=self.fontsize_txt, zorder=3 )
                self.ax.add_patch( patches.FancyArrowPatch((t_x, t_y), (a_x, a_y), arrowstyle='->', mutation_scale=20, zorder=3) )
            else:
                t_x, t_y = r.id_loc
                t = self.ax.text( t_x, t_y, r.id_txt, verticalalignment='center', horizontalalignment='center', size=self.fontsize_txt, zorder=3 ).set_clip_on(True)
                #t.set_bbox( dict(facecolor=r.color, alpha=1, lw=0, edgecolor=r.color, pad=0 ) )
        
        if letter!='': self.ax.text( -0.17, 1, letter, horizontalalignment='center',verticalalignment='top', fontsize=self.fontsize_titles*1.4, transform=self.ax.transAxes )

        # log formatting
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))        
        if self.log_x: 
            self.ax.set_xscale('log')
            self.ax.xaxis.set_major_formatter(formatter)
        if self.log_y:
            self.ax.set_yscale('log')
            self.ax.yaxis.set_major_formatter(formatter)
        
        self.ax.tick_params( axis='x', labelsize=self.fontsize_ticks )
        self.ax.tick_params( axis='y', labelsize=self.fontsize_ticks )

        self.set_bounds( self.ax )
        if len(self.x_bounds)>0: self.ax.set_xlim( self.x_bounds )
        if len(self.y_bounds)>0: self.ax.set_ylim( self.y_bounds )
        #turn off top and right spine
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)

        self.ax.set_xlabel( self.get_label(self.x_var), fontsize=self.fontsize_titles )
        self.ax.set_ylabel( self.get_label(self.y_var), fontsize=self.fontsize_titles )
        plt.tight_layout()


    def get_label( self, var ):
        ax_labels = {
            'Qt': r"$Q_t=\frac{ q_t - \sigma _v }{ \sigma ' _v }$" + ' (-)',
            'Fr': r'$F_r=\frac{ f_s }{ q_t - \sigma _v }$' +' (%)',
            'Rf': r'$R_f=\frac{ f_s }{ q_t }$' +' (%)',
            'Bq': r'$B_q=\frac{ u_2 - u_0 }{ q_t - \sigma _v }$' + ' (-)',
            'du_n': r"$\frac{u _2 - u _0}{\sigma ' _v }$" + '  (-)',
            'qe': r'$q _e = q_t - u _2$' + ' (kPa)',
            'qt': 'q' + r'$_t$' + ' (kPa)',
            'fs': 'f' + r'$_s$' + ' (kPa)',
        }
        if var in ax_labels: return ax_labels[var]
        return var


    class region_2d():
        def __init__( self, parent, index, name, id_loc, x, y ):
            self.parent = parent

            # save data
            self.index = index
            self.name = name
            self.id_txt = name.split(' ')[0] # id from name
            self.id_loc = id_loc
            if not self.id_loc: self.id_loc=[1,200] # remove when done with model defs
            self.color = self.parent.colors[self.index]
            self.x = np.array(x)
            self.y = np.array(y)

            # optional pre-check shortout
            self.minX = np.min(self.x)
            self.maxX = np.max(self.x)
            self.minY = np.min(self.y)
            self.maxY = np.max(self.y)

            # report back
            self.parent.x_lim[0] = min(self.parent.x_lim[0], self.minX)
            self.parent.x_lim[1] = max(self.parent.x_lim[1], self.maxX)
            self.parent.y_lim[0] = min(self.parent.y_lim[0], self.minY)
            self.parent.y_lim[1] = max(self.parent.y_lim[1], self.maxY)
            
            



        def x_of_y_by_points( self, y, x1, y1, x2, y2 ):
            if self.parent.log_y and y>0 and y1 > 0 and y2 > 0:
                if self.parent.log_x and x1 > 0 and x2 > 0:  # log-log
                    x = 10**( (np.log10(y / y1) * np.log10(x1 / x2) / np.log10(y1 / y2)) + np.log10(x1) )
                else:  # lin-log
                    x = np.log10(y/y1) * ( (x1-x2) / np.log10(y1/y2) ) + x1
            else:
                if self.parent.log_x and x1 > 0 and x2 > 0:  # log-lin
                    x = 10**( ((y-y1) * np.log10(x1/x2) / (y1-y2)) + np.log10(x1) )
                else:  # lin-lin (log(n) where n <= 0 will also default to here)
                    x = ( y-y1 ) * ( (x1-x2) / (y1-y2) ) + x1
            return x


        def contains( self, some_x, some_y ):
            point_inside = False
            
            n = len(self.x)
            j = n-1

            if not ( some_x < self.min_x or some_x > self.max_x or some_y < self.min_y or some_y > self.max_y ): # do point in poly routine
                for i in range(n):
                    if ( ( (self.y[i]>some_y) != (self.y[j]>some_y)) and (some_x<self.xOfyByPoints(some_y, self.x[i], self.y[i], self.x[j], self.y[j])) ):
                        point_inside = not point_inside
                    j=i
            return point_inside


if __name__=='__main__':
    for m in model_defs:
        #m='schneider_08'
        cl=general_model(model_defs[m], fill_regions=True)
        cl.plot( letter='' )
        #cl.show_legend()

    #cl_rob = robertson_90_Bq()
    #cl_rob.plot()
