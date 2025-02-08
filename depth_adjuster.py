import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from depth_warps import depth_warps
from PCC_warp import PCC_warp

'''
    Class used to identify depth warps used in this study.
    It utilizes the PCC_warp class to iteratively build a depth warp array.
'''

class depth_adjuster():
    def __init__( self, dataset, unit_model=None ):
        self.dataset = dataset
        self.warps = depth_warps
        self.unit_model = unit_model


    def calc_depth_shifts( self ): # routine used to help generate depth warps

        adjuster = PCC_warp(
            n=20,
            n_increments=20,
            threshold=1e-6,
            proximity_limit = 0.3,
            max_shift = 0.4,
            scale_shift_curve=True
        )

        ref = self.dataset.get_reference()
        s_name = self.dataset.all_pos_names()[12]

        exchange_reference = False
        if exchange_reference: # exchange reference for warped curve
            r_name = self.dataset.all_pos_names()[2]
            ref = self.dataset.sounding_by_name( r_name )

        warp = self.warps[s_name]['final_warp']
        drop = self.warps[s_name]['drop_ranges']

        print( 'working on: ' + s_name )
        sounding = self.dataset.sounding_by_name( s_name )

        if True: # use cpt .log files ( always! )
            curve_depth, curve_data  = sounding.get_data_with_detph( 'qc' )
            ref_depth, ref_data = ref.get_data_with_detph( 'qc' )
            curve = [ curve_depth, curve_data ] #[ sounding.data['d'], sounding.data['qc'] ]
            reference = [ ref_depth, ref_data ] #[ ref.data['d'], ref.data['qc'] ]
        else: # use standard .cpt files
            curve = [ sounding.data['d'], sounding.data['qc'] ]
            reference = [ ref.data['d'], ref.data['qc'] ]


        if exchange_reference: # apply warp to reference
            r_warp = self.warps[r_name]['final_warp']
            reference = adjuster.apply_warp( reference, r_warp )

        # start here:  match regions that are easy to match
        p1 = (18.596,18.334)
        p2 = (19.692,19.46)

        if False: # focus on easy parts first
            curve = adjuster.trim_to_def( curve, [p1[0],p2[0]] )
            reference = adjuster.trim_to_def( reference, [p1[1],p2[1]] )
            shift = adjuster.fit_path( curve=curve, reference=reference)
            warp = adjuster.get_warp()
            print( s_name, str(warp).replace(' ','') )
            file_name = s_name + '.gif'
            adjuster.plot_history( file_name=file_name, frame_ms=500 )

        #new_warp = adjuster.where_to_warp( x_from=12.5, curve=curve, reference=reference, warp=warp )
        adjuster.plot_warp( curve, reference, warp, delta=0, id_curve=s_name, drop=drop)#, filename=s_name+'.png')

        if False: # semi automatic
            new_warp = adjuster.where_to_warp( x_from=12.375, curve=curve, reference=reference, warp=warp )
            print(new_warp)
        if False: # automatic fit
            adjuster.fit_path( curve, reference, warp=warp )
            print( s_name, adjuster.get_warp())
            adjuster.plot_history( file_name=s_name + '.gif', frame_ms=500 )


    def plot_all( self, warped=True, fontsize=None, tick_f_size=None, apply_drops=False, draw_units=True ):

        x_lims = { 
            0:[[400,1400],[0,40],[0,1200],[0,10]], #[[400,1400],[0,40],[0,1200],[0,10]]
            1:[[0, 15000],[0,160],[0,500],[0,10]], #[[0, 15000],[0,160],[0,500],[0,10]]
            2:[[0, 3000],[0,70],[0,750],[0,10]], # [[0, 3000],[0,70],[0,750],[0,10]] # 2:[[0, 2000],[0,25],[0,325],[0,10]],

            }
        y_lims = {
            0:[4,20], # [4,20]
            1:[8,20], # [9,10.4], # 
            2:[3,20]  # [3,20] #[8,12]
            }

        if fontsize==None: fontsize = 10
        if tick_f_size==None: tick_f_size = 10

        if draw_units: fig, ax = plt.subplots(1,4, figsize=(12,9), sharey=True, gridspec_kw={'width_ratios': [ 3, 3, 3, .8 ]} )#, sharex=True)
        else: fig, ax = plt.subplots(1,3, figsize=(11,9), sharey=True, gridspec_kw={'width_ratios': [ 3, 1, 3 ]} )#, sharex=True)

        ax[0].set_ylabel('Depth (m)', fontsize=fontsize)
        plt_vars = ['qt', 'fs', 'u']
        for i, a in enumerate(ax):
            a.set_ylim(y_lims[self.dataset.location])
            a.set_xlim(x_lims[self.dataset.location][i])

            x = x_lims[self.dataset.location][i]
            y = y_lims[self.dataset.location]

            # draw units
            if draw_units:## and False:
                self.unit_model.dark = False # start each graph on a dark field
                for j, unit in enumerate(self.unit_model.units):
                    label = None if i==len(plt_vars) else ''
                    unit.draw( a, label=label, fontsize=fontsize ) #label,

            if i==len(plt_vars): break

            label = plt_vars[i][0]
            if len(plt_vars[i])>1:
                label = f"{plt_vars[i][0]}$_{{{plt_vars[i][1]}}}$"
            label += ' (kPa)'
            a.set_xlabel( label, fontsize=fontsize*1.1 )
            a.xaxis.set_label_position('top')

        if draw_units:
            ax[-1].set_xlabel( 'Units', fontsize=fontsize )
            ax[-1].xaxis.set_label_position('top')

        ax[0].invert_yaxis()
        lw_ref = 0.6

        for s in self.dataset.soundings:
            #if s.target_rate != '20mm/s': continue


            lw = lw_ref if s.target_rate != '20mm/s' else .8
            zorder = 1 if lw==lw_ref else 9
            for i, var in enumerate( plt_vars ):
                curve_depth, curve_data = s.get_data_with_detph( var, warped_depth=warped, apply_drops=apply_drops )
                ax[i].plot(curve_data, curve_depth, c=self.dataset.rate_colors[s.target_rate], label=s.target_rate, lw=lw, zorder=zorder) #
                ax[i].tick_params( labelsize=tick_f_size ) # top=True, labeltop=True, bottom=False, labelbottom=False,

        sm = self.dataset.soundings[0].soil_stress_model

        u_0_color = (0,175/255,240/255,1)
        u_0_fcolor = (0,175/255,240/255,.2)
        #ax[2].plot(sm.u0, sm.d, lw=1.5, ls='--',c=u_0_color, zorder=zorder)

        vertices = [ (x_i, y_i) for (x_i,y_i) in zip( sm.u0, sm.d ) ]
        vertices += [ ( -10000, vertices[-1][1] ), ( -10000, vertices[0][1] ), vertices[0] ]
        
        ax[2].add_patch( Polygon(vertices, closed=True, fc=u_0_fcolor,ec=u_0_color, ls='--', lw=1.5, zorder=-1,label=r'$u_0$') )
        

        handles, labels = ax[2].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # count occurences
        by_label = { (k if 'u' in k else k + ' (' + str(labels.count(k)) + 'x)'): v for k, v in by_label.items() }
        

        legend = ax[1].legend(by_label.values(), by_label.keys(), framealpha=1, fontsize=tick_f_size ).set_zorder( 99)

        # no x-scale on unit definition
        if draw_units:
            ax[-1].xaxis.set_tick_params(labelbottom=False)
            ax[-1].set_xticks([])
        
        for i,some_ax in enumerate(ax): 
            if i ==len(ax)-1: break
            some_ax.grid()

        plt.tight_layout()
        plt.show()


    def apply_drops( self, depth, data, drops ):
        tmp_depth = depth.copy()
        tmp_data = data.copy()

        for d in drops:
            filter_from = depth>=d[0]
            filter_to = depth<=d[1]
            mask = np.multiply(filter_from, filter_to)

            tmp_depth[mask]=np.nan
            tmp_data[mask]=np.nan

        return tmp_depth, tmp_data

    def set_warps( self ):
        tmp_warp = None
        for s in self.dataset.soundings:
            # attach data from warp definition to each cpt
            warp = self.warps[s.pos_name]['final_warp']
            drops = self.warps[s.pos_name]['drop_ranges']

            s.set_warp( warp, drops )


            if tmp_warp == warp:
                a=1
            tmp_warp = warp.copy()