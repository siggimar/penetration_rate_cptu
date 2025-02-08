import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, LinearNDInterpolator
import numpy as np
import json
import os
from cptu_rate_plotter import rate_plotter


class unit_model():
    """
        Model keeps track of unit definition across a site
        used to query depth intervals at different locations.

        results from rate analysis is attached to each unit in res dictionary
    """

    def __init__( self, dataset ):
        self.dataset = dataset
        self.reference = dataset.get_reference()
        self.set_units()
        self.dark = False

        self.r_plotter = rate_plotter()


    def set_units( self ):
        self.units = []
        self.boreholes = []

        for i, ( d_from, d_to ) in enumerate( self.dataset.unit_defs ):
        #for i in range(len(self.dataset.unit_boundaries)-1):
            x,y,z,z_top,z_bottom = [],[],[],[],[] # unit point definition

            # set reference definition
            #d_from = self.dataset.unit_boundaries[i]
            #d_to   = self.dataset.unit_boundaries[i+1]

            for s in self.dataset.soundings:
                # coordinates and warped depths
                sx,sy,sz = s.get_coordinates()
                sd_from = s.warp_depths( d_from )
                sd_to = s.warp_depths( d_to )

                # absolute warped elevations
                z_from = sz-sd_from
                z_to = sz-sd_to

                # append to coordinate list for unit definition
                x.append( sx )
                y.append( sy )
                z.append( sz )
                z_top.append( z_from )
                z_bottom.append( z_to )

                z_start = sz-s.data['d'][0]
                z_stop = sz-s.data['d'][-1]

                self.boreholes.append(
                    unit_model.borehole( self, s.pos_name, sx, sy, sz, z_start, z_stop )
                )

            # define unit and add to list
            self.units.append( unit_model.unit( self, i, x, y, z, z_top, z_bottom ) )


    def plot( self, interpolate=False, plt_units=True, plt_boreholes=True ):
        fig = plt.figure( figsize=(12,10) )
        ax = fig.add_subplot( 111, projection='3d' )

        if plt_boreholes:
            for borehole in self.boreholes:
                x_i, y_i, z_i, z_i_start, z_i_stop = borehole.x, borehole.y, borehole.z_terrain, borehole.z_start, borehole.z_stop

                ax.scatter3D( x_i, y_i, z_i, color='black', zorder=999 ) # ball
                ax.plot( [x_i,x_i], [y_i,y_i], [z_i,z_i_start], color=(0.5,0.5,0.5), zorder=99) # predrilling
                ax.plot( [x_i,x_i], [y_i,y_i], [z_i_start,z_i_stop], color=(0,0,0) ) # drilling

                ax.text( borehole.x, borehole.y, borehole.z_terrain + borehole.txt_offset, borehole.name, None, zorder=999 )


        if plt_units:
            for unit in self.units:
                # visualize boreholes
                if interpolate:
                    unit.interpolate() # polulate units:X,Y,Z
                    ax.plot_surface(unit.X, unit.Y, unit.Z, alpha=.5)
                else:
                    surf = ax.plot_trisurf( unit.x, unit.y, unit.z_top, alpha=.5)

        ax.set_xlabel('Northing (m)')
        ax.set_ylabel('Easting (m)')
        ax.set_zlabel('Elevation (m)')

        plt.show()


    def res_to_json( self ):
        unit_res = {}
        # combine all unt results to single dict
        for u in self.units: 
            unit_res[ ( 'unit:' + str(u.unit_number) )  ] = u.res

        location_res = { 'location:' + str(self.dataset.location): unit_res } # add location ID to combine locations

        return json.dumps( location_res, indent=4 )

    def res_to_csv( self, keys ):
        csv = ''
        for unit in self.units:
            for r in unit.res:
                # first check if all keys are present in 

                for k in keys:
                    csv += str(unit.res[r][k]) + ','
                csv = csv[0:-1] + '\n'
        return csv


    def get_res_keys( self ):
        all_keys = list( self.units[0].res[0].keys() )
        return all_keys


    def save_res_as_csv( self, folder=''  ):
        csv_summary = self.res_to_csv()

        f_name = 'rate_res.csv'
        f_path = os.path.join( folder, f_name )
        if not os.path.exists( folder ) and folder!='': os.makedirs( folder )
        pass


    def save_res_as_json( self, folder='' ):
        json_summary = self.res_to_json() # all results from analysis

        res = self.units[0].res # want to mark it by parameter
        param = res[ list(res)[0] ]['parameter']

        f_name = 'k_res_' + str(self.dataset.location) + '_' + param + '.json'
        f_name = os.path.join( folder, f_name )

        if not os.path.exists( folder ): os.makedirs( folder )

        self.write_string_to_file( json_summary, f_name )


    def write_string_to_file( self, some_string, filename ):
        with open( filename, 'w', encoding="utf-8") as f:
            f.write(some_string)


    class unit():
        def __init__( self, parent, n, x, y, z, z_top, z_bottom ):
            self.parent = parent
            self.unit_number = n
            self.unit_name = str(self.parent.dataset.location) + '-' + str(self.unit_number)
            self.x = x # lists of coordinates
            self.y = y
            self.z = z
            self.z_top = z_top
            self.z_bottom = z_bottom
            self.delta_xy = 0.25

            self.d_top = [ zi - zt_i for zi, zt_i in zip(self.z, self.z_top) ] # calculate depths as well
            self.d_bottom = [ zi - zb_i for zi, zb_i in zip(self.z, self.z_bottom) ]

            self.d_avg = [ (zt+zb)/2 for zt, zb in zip(self.d_top, self.d_bottom) ]

            cv_d, cv_v = self.parent.dataset.c_v

            self.cv = np.interp( self.d_avg, cv_d, cv_v ) 

            self.thickness = [ z_t-z_b for z_t, z_b in zip(self.z_top, self.z_bottom) ]

            self.res = {}
            self.res_idx = 0
            self.u_ids_ = 0

            self.unit_mgr = {} # hash_table


        def add_res( self, some_res ):
            self.res[self.res_idx] = some_res
            self.res_idx += 1


        def update_res( self, some_res ):
            self.res[self.res_idx].update( some_res )


        def unt_top_bottom_by_coordinate( self, x, y, return_depth=False ):
            # as (x,y) coordinates will match coords in self.x, self.y this would go faster using a dict_lookup

            xy = np.zeros( (len(self.x), 2) )
            xy[:, 0] =self.x
            xy[:, 1] = self.y

            zt = self.z_top
            zb = self.z_bottom
            if return_depth:
                zt=self.d_top
                zb=self.d_bottom

            f_top = LinearNDInterpolator(xy, zt)
            f_bottom = LinearNDInterpolator(xy, zb)

            return f_top(x,y), f_bottom(x,y)


        def interpolate( self ):
            # attempt to extrapolate 
            n=200
            x_lim = [ min(self.x)-self.delta_xy, max(self.x)+self.delta_xy ]
            y_lim = [ min(self.y)-self.delta_xy, max(self.y)+self.delta_xy ]
            xi, yi = np.linspace(x_lim[0], x_lim[1], n), np.linspace(y_lim[0], y_lim[1], n)
            self.X, self.Y = np.meshgrid( xi, yi )
            rbf_func = ['multiquadric', 'inverse', 'gaussian','linear','cubic','quintic','thin_plate'][2]
            rbf = Rbf( self.x, self.y, self.z_top, function=rbf_func )
            self.Z = rbf(self.X, self.Y)


        def draw( self, ax, fontsize, label=None ):
            if label is None: label=self.unit_name

            unit_colors = {
                0: (230/255,230/255,230/255),
                1: (255/255,255/255,245/255)
            }
            x = ax.get_xlim()
            y = ax.get_ylim()

            dx =  (x[1]-x[0])/2

            y_top = [ self.d_top[0] for some_x in x ]
            y_bottom = [ self.d_bottom[0] for some_x in x ]

            dy = -(y_top[0]-y_bottom[0])/2.5

            #unit_color =  unit_colors[int(self.parent.dark)]
            unit_color = self.parent.r_plotter.get_color(self.unit_name)

            self.parent.dark = not self.parent.dark # switch colors

            ax.plot( x, y_top, ls='--', c=(0,0,0), lw=0.5, zorder=-10 )
            ax.plot( x, y_bottom, ls='--', c=(0,0,0), lw=0.5, zorder=-10 )

            ax.fill_between( x, y_bottom, y_top, color=unit_color, zorder=-20)#, alpha=0.6 ) # alpha=0.2

            if label!='': ax.text( x[0] + dx, self.d_bottom[0] - dy, label,horizontalalignment='center', verticalalignment='center', size=fontsize ).set_clip_on(True)


    class borehole():
        def __init__( self, parent, name, x, y, z_terrain, z_start, z_stop ):
            self.parent = parent
            self.name = name
            self.x = x # values
            self.y = y
            self.z_terrain = z_terrain
            self.z_start = z_start
            self.z_stop = z_stop
            self.txt_offset = 0.5