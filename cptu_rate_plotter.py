import os
import pandas as pd
import json
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import matplotlib.ticker as mticker

from cptu_classification_models_2d import model_defs
from cptu_classification_charts import general_model

# This script was used to generate most figures found in study
# parameters for the main conclusion (3D figure with v, B_q and K) 
# were selected editing line # 62.  
# to select figures see [ if __name__=='__main__': ] part of script 
# on line 1404

class rate_plotter():
    '''
    Class to plot results from rate analysis. Takes a CSV with unit analysis results (by cptu_rate_processor) as input
    
    '''

    def __init__( self, result_file=None, split_by='site', fill=True ):
       self.fill = fill
       if result_file is not None: self.data = pd.read_csv( result_file )
       self.split_by = split_by # group points by: 'site', 'unit' or 'position' # ['site','unit','position'][0]



    def calc_model( self, v, Bq ):
        # returns values for surface fitted unit model data
        ap = (-0.14, 13, 15, 1.061)
        dp = (0.08, 13, 25, 0.978)

        # a & d are functions of v        
        a = ap[0]*np.tanh((v-ap[1])/ap[2]) + ap[3]
        d = dp[0]*np.tanh((v-dp[1])/dp[2]) + dp[3]
        b = 3
        c = 0.19
        return d + (a-d)/(1+(np.clip(Bq,0,np.inf)/c)**b)


    def k_3d_plot( self ):
        size = 30
        axis_label_size = 15
        axis_tick_size = 13

        n_xy = 1000

        x_lims = (1.4, -.6)
        y_lims = (0, 70) # v
        z_lims = (0.7, 1.4) # k
        log_x = False

        # this line was edited to check which relations were most promising to define K,  giving a simple q_t model
        (( x, v, z ), labels), var_list = self.vars_3D_figs( 'Bq__value_avg', 'rate_avg', 'qt_k' ) #'Qt__k')#'Bq__value_avg', 'rate_avg', 'qt_k' #'Bq__value_avg', 'rate_avg', 'Qt__k'

        fig, ax = plt.subplots( subplot_kw={"projection": "3d"}, figsize=(13.04,10.00) )

        # define calculation grid
        Bq_min, Bq_max = ( -0.2, 1.4 )
        v_min, v_max  = ( 5, 65 )
        grid_Bq = np.linspace( Bq_min, Bq_max, n_xy )
        grid_v = np.linspace( v_min, v_max, n_xy )
        Bqs, vs = np.meshgrid( grid_Bq , grid_v )

        model_grid_vals = self.calc_model( vs, Bqs )  # calculate model parameters and model values (on grid)

        # draw surface wireframe,  rstride & cstride are set so that wirenet matches axis major axis
        surf = ax.plot_wireframe( Bqs, vs, model_grid_vals, rstride=83, cstride=125, color=self.get_color('mesh')[0], linewidth=1, label='Model')
        #surf2 = ax.contour3D( Bqs, vs, model_grid_vals, 150, cmap='binary')

        first_line = True
        for some_line_v in [15,25]:
            line_q_norm = self.calc_model( some_line_v, grid_Bq  ) # 3D line data
            line_v = grid_v*0 + some_line_v # generate v values of same shape

            label='v = 20±5 (mm/s)' if first_line else ''
            ax.plot( grid_Bq, line_v, line_q_norm, lw=2.5, ls='--', c=self.get_color('model')[0], label=label, zorder=20 )
            first_line = False # only label first line

        # standard rate
        line_q_norm = self.calc_model( 20, grid_Bq  ) # 3D line data
        ax.plot( grid_Bq, (grid_v*0+20), line_q_norm, lw=3, c=self.get_color('model')[0], label='v = 20 (mm/s)', zorder=20 )        

        # draw datapoints
        for i, label in enumerate(labels):
            m, msf = self.marker_style( label )
            mc, mc_edge = self.get_color( label, ax )
            
            tmp_x = np.log10( x[i] ) if log_x else x[i]
            plot_mask = v[i]<y_lims[1]
            ax.scatter3D(
                tmp_x[plot_mask], v[i][plot_mask], z[i][plot_mask],
                label=label, # in legend
                s=size*msf,
                marker=m, # type
                ec=mc_edge, fc=mc, # colors
                alpha=1, # no fading
                zorder=10
            )

        # update tick values to mimic log-scale ( log values drawn on linear scale )
        if log_x:
            ax.xaxis.set_major_formatter( mticker.FuncFormatter(self.log_tick_formatter) )
            ax.xaxis.set_major_locator( mticker.MaxNLocator(integer=True) )

        if log_x: x_lims=(np.log10(x_lims[0]), np.log10(x_lims[1]))
        ax.set_xlim( x_lims )
        ax.set_ylim( (y_lims[1], y_lims[0]) )
        ax.set_zlim( z_lims )

        # annotate axis
        offset = 20

        ax.set_xlabel( self.get_label( var_list[0] ), fontsize=axis_label_size*1.2 )
        ax.set_ylabel( self.get_label( var_list[1] ), fontsize=axis_label_size)
        ax.set_zlabel( self.get_label( var_list[2] ), fontsize=axis_label_size*1.3)
        ax.xaxis.labelpad = offset
        ax.yaxis.labelpad = offset/2
        ax.zaxis.labelpad = offset/2

        ax.tick_params( axis='x', labelsize=axis_tick_size )
        ax.tick_params( axis='y', labelsize=axis_tick_size )
        ax.tick_params( axis='z', labelsize=axis_tick_size )

        ax.view_init(elev=25, azim=160)

        plt.legend( fontsize=axis_tick_size )
        plt.show()


    def k_flat_plots( self, split_val=0.3 ):
        # settings
        size = 60
        axis_label_size = 15
        axis_tick_size = 13
        frac = 95

        x_lims = (0,70) #(5,50)
        y_lims = (0.7,1.4)

        # data
        (( Bq, v, k ), labels), var_list = self.vars_3D_figs('Bq__value_avg', 'rate_avg', 'qt_k') #'Bq__value_avg', 'rate_avg', 'qt_k'

        for i, label in enumerate( labels ):

            v_s = np.linspace( 4, 70, 1000)
            tbqs, tvs, tzs = self.split_by_x( Bq[i], v[i], k[i], split_val )

            for tbq, tv, tz in zip( tbqs, tvs, tzs ):
                #if 'alsen' not in label:continue
                # use drained/undrained plot
                model_Bq = np.average(tbq)

                model_k = self.calc_model( v_s, model_Bq )
                offs = self.calc_offset( v_s, model_k, tv, tz, frac/100 )
                model_lb = model_k-offs
                model_ub = model_k+offs

                fig, ax = plt.subplots( figsize=(10,4.00) ) #silt_figures: (6,4.00)

                # markers
                m, msf = self.marker_style( label )
                mc, mc_edge = self.get_color( label, ax )

                ax.plot(v_s, model_k, lw=2, c=mc_edge, ls='-', label='Model at ' r'$B_q$' + '=' + str(np.round(model_Bq,2)), zorder=10)
                ax.plot(v_s,model_ub, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)
                ax.plot(v_s, model_lb, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)

                ax.fill_between( v_s, model_lb, model_ub, facecolor=self.get_color('light_grey')[0], alpha=1, label=str(frac) + '% of points within ±' + str(round(offs,3)), zorder=1 )


                ax.plot(x_lims,[1]*len(x_lims), lw=1, c=mc_edge, ls='-', zorder=2)

                for r in [ 10, 15, 20, 25, 30 ]:
                    x_offset = -.7 if r in [ 10, 25 ] else .7
                    ls = '-' if r==20 else '--'
                    y_vals = [y_lims[0], self.calc_model( r, model_Bq ) ]
                    ax.plot( [r]*2, y_vals, lw=1, ls=ls, c=(0,0,0), zorder=2)

                    if r!=20:
                        t = ax.text( r+x_offset, y_vals[1]+0.1, np.round(y_vals[1],3), horizontalalignment='center', verticalalignment='bottom', zorder=11, fontsize=axis_tick_size )
                        t.set_bbox( dict(facecolor=(1,1,1), alpha=.7, edgecolor=(0,0,0), zorder=11) )
                        ax.plot( r, y_vals[1], marker='o', ms=10, mfc=self.get_color('model')[0], mec=( 0, 0, 0 ),mew=1.8, zorder=11 )

                if len(tbqs)==1:
                    t_label = label
                else:
                    l_ins = '>'
                    if np.average(tbq)<split_val: l_ins = '<'
                    t_label = label + ' (' + r'$B_q$' + l_ins + str(split_val) + ')'



                print('x')
                print(tv)
                print('y')
                print(tz)
                ax.scatter( # datapoints
                    tv, tz, # xy (Qt log transformed)
                    label=t_label, # in legend
                    s=size*msf,
                    marker=m, # type
                    ec=mc_edge, fc=mc, # colors
                    alpha=.3, # no fading
                    zorder=3
                )

                ax.set_xticks([0,10,15,20,25,30,40,50,60,70])
                ax.set_xlim( x_lims )
                ax.set_ylim( y_lims )
                ax.set_xlabel( self.get_label( var_list[1] ), fontsize=axis_label_size )
                ax.set_ylabel( self.get_label( var_list[2] ), fontsize=axis_label_size*1.2 )
                ax.tick_params( axis='x', labelsize=axis_tick_size )
                ax.tick_params( axis='y', labelsize=axis_tick_size )


                ax.legend( fontsize=axis_tick_size ) #, loc='upper right' )
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.tight_layout()
                plt.show()


    def k_flat_plots2( self, split_val=99 ):
        from scipy.optimize import minimize
        def best_fit_line( x, y, x_vals ):

            # calculate coefficients of least squaresline
            A = np.vstack([x, np.ones_like(x)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            # eval line at specified x_list
            y_vals = m * x_vals + c

            return y_vals, m, c

        # settings
        size = 60
        axis_label_size = 15
        axis_tick_size = 13
        frac = 95

        x_lims = (0,70) #(5,50)
        y_lims = (0.0,1.6)


        # data
        (( Bq, v, k ), labels), var_list = self.vars_3D_figs('Bq__value_avg', 'rate_avg', 'fs_k') #'Bq__value_avg', 'rate_avg', 'qt_k''fs_sigma_v0_eff'


        fig, axs = plt.subplots( 1, 3, figsize=(12,4.50) ) #silt_figures: (6,4.00)
        for i, label in enumerate( labels ):

            v_s = np.linspace( 4, 70, 1000)
            tbqs, tvs, tzs = self.split_by_x( Bq[i], v[i], k[i], split_val )

            for tbq, tv, tz in zip( tbqs, tvs, tzs ):
                if label=='Stjørdal: Halsen':
                    print(tbq)
                    print(tv)
                    print(tz)
                    print(label)
                #continue
                #if 'alsen' not in label:continue
                # use drained/undrained plot

                some_mask = tv<70
                tbq, tv, tz = tbq[some_mask], tv[some_mask], tz[some_mask]
                

                model_k, a ,b = best_fit_line( tv, tz, v_s )
                a, b = [ [0.00278, 0.94449], [-0.00384448, 1.076889594], [-0.005058586, 1.101171715] ][i] # RMSE minimized for data with constraint: f(20)~= 1
                a_, b_ = a, b
                model_k = b + v_s * a

                #model_k = sm.nonparametric.lowess( tz,tv, frac=1,xvals=v_s)#[0]

                offs = self.calc_offset( v_s, model_k, tv, tz, frac/100 )
                model_lb = model_k-offs
                model_ub = model_k+offs

                # markers
                m, msf = self.marker_style( label )
                mc, mc_edge = self.get_color( label, axs[i] )

                a, b = str(round(a,4)), str(round(b,2))

                axs[i].plot( v_s, model_k, lw=2, c=mc_edge, ls='-', zorder=10, label=r'$k_{f_s}=$' + a + r'$ v $ + ' + b )
                axs[i].plot( v_s, model_ub, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)
                axs[i].plot( v_s, model_lb, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)
                
                axs[i].fill_between( v_s, model_lb, model_ub, facecolor=self.get_color('light_grey')[0], alpha=1, label=str(frac) + '% within ±' + str(round(offs,3)), zorder=1 )

                axs[i].plot(x_lims,[1]*len(x_lims), lw=1, c=mc_edge, ls='-', zorder=2)


                if len(tbqs)==1:
                    t_label = label
                else:
                    l_ins = '>'
                    if np.average(tbq)<split_val: l_ins = '<'
                    t_label = label + ' (' + r'$B_q$' + l_ins + str(split_val) + ')'

                axs[i].scatter( # datapoints
                    tv, tz, # xy (Qt log transformed)
                    label=t_label, # in legend
                    s=size*msf,
                    marker=m, # type
                    ec=mc_edge, fc=mc, # colors
                    alpha=.3, # no fading
                    zorder=3
                )

                rs = [ 10, 20, 30 ]
                for r in rs:
                    axs[i].plot( [20]*2, [0,1], lw=1, ls='-', c=(0,0,0), zorder=2)

                    if r!=20:
                        y = a_*r+b_
                        t = axs[i].text( r+0, y+0.1, np.round(y,3), horizontalalignment='center', verticalalignment='bottom', zorder=11, fontsize=axis_tick_size )
                        t.set_bbox( dict(facecolor=(1,1,1), alpha=.7, edgecolor=(0,0,0), zorder=11) )
                        axs[i].plot( r, y, marker='o', ms=10, mfc=self.get_color('model')[0], mec=( 0, 0, 0 ),mew=1.8, zorder=11 )
                        axs[i].plot( [r]*2, [0,y], ls='--', lw=1, c=(0,0,0), zorder=2 )
                        
                
                x_ticks = list(set(np.append( axs[i].get_xticks(), rs )))
                axs[i].set_xticks( np.sort(x_ticks) )

                axs[i].set_xlim( x_lims )
                axs[i].set_ylim( y_lims )
                axs[i].set_xlabel( self.get_label( var_list[1] ), fontsize=axis_label_size )
                axs[i].set_ylabel( self.get_label( var_list[2] ), fontsize=axis_label_size*1.2 )
                axs[i].tick_params( axis='x', labelsize=axis_tick_size )
                axs[i].tick_params( axis='y', labelsize=axis_tick_size )

                axs[i].legend( fontsize=axis_tick_size ) #, loc='upper right' )
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)

        fig.text(0.03, 0.94, 'A', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
        fig.text(0.36, 0.94, 'B', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
        fig.text(0.69, 0.94, 'C', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
        plt.tight_layout()
        plt.show()


    def k_flat_plots3( self, split_val=99 ):
        from scipy.optimize import minimize
        def best_fit_line( x, y, x_vals ):

            # calculate coefficients of least squaresline
            A = np.vstack([x, np.ones_like(x)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            # eval line at specified x_list
            y_vals = m * x_vals + c

            return y_vals, m, c

        # settings
        size = 60
        axis_label_size = 15
        axis_tick_size = 13
        frac = 95

        x_lims = (0,70) #(5,50)
        y_lims = (-0.4,1)

        # data
        (( Bq, v, k ), labels), var_list = self.vars_3D_figs('Bq__value_avg', 'rate_avg', 'du_k_star') #'Bq__value_avg', 'rate_avg', 'qt_k'

        fig, axs = plt.subplots( 1, 3, figsize=(12,4.50) ) #silt_figures: (6,4.00)
        for i, label in enumerate( labels ):

            v_s = np.linspace( 4, 70, 1000)
            tbqs, tvs, tzs = self.split_by_x( Bq[i], v[i], k[i], split_val )

            for tbq, tv, tz in zip( tbqs, tvs, tzs ):
                #if 'alsen' not in label:continue
                # use drained/undrained plot
                
                some_mask = (tz>np.percentile(tz, 1)) & (tz<np.percentile(tz, 99))

                fit_v, fit_z = tv[some_mask], tz[some_mask]

                #some_mask = tv<70
                #tbq, tv, tz = tbq[some_mask], tv[some_mask], tz[some_mask]
                model_k, a ,b = best_fit_line( fit_v, fit_z, v_s )
                #model_k = sm.nonparametric.lowess( tz,tv, frac=1,xvals=v_s)#[0]

                offs = self.calc_offset( v_s, model_k, tv, tz, frac/100 )
                model_lb = model_k-offs
                model_ub = model_k+offs

                # markers
                m, msf = self.marker_style( label )
                mc, mc_edge = self.get_color( label, axs[i] )

                a, b = str(round(a,4)), str(round(b,2))

                axs[i].plot( v_s, model_k, lw=2, c=mc_edge, ls='-', zorder=10, label=r'$k ^* _{\Delta u _2}=$' + a + r'$ v $ + ' + b )
                axs[i].plot( v_s,model_ub, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)
                axs[i].plot( v_s, model_lb, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)
                
                axs[i].fill_between( v_s, model_lb, model_ub, facecolor=self.get_color('light_grey')[0], alpha=1, label=str(frac) + '% within ±' + str(round(offs,3)), zorder=1 )

                #axs[i].plot(x_lims,[1]*len(x_lims), lw=1, c=mc_edge, ls='-', zorder=2)


                if len(tbqs)==1:
                    t_label = label
                else:
                    l_ins = '>'
                    if np.average(tbq)<split_val: l_ins = '<'
                    t_label = label + ' (' + r'$B_q$' + l_ins + str(split_val) + ')'

                axs[i].scatter( # datapoints
                    tv, tz, # xy (Qt log transformed)
                    label=t_label, # in legend
                    s=size*msf,
                    marker=m, # type
                    ec=mc_edge, fc=mc, # colors
                    alpha=.3, # no fading
                    zorder=3
                )

                axs[i].set_xlim( x_lims )
                axs[i].set_ylim( y_lims )
                axs[i].set_xlabel( self.get_label( var_list[1] ), fontsize=axis_label_size )
                axs[i].set_ylabel( self.get_label( var_list[2] ), fontsize=axis_label_size*1.2 )
                axs[i].tick_params( axis='x', labelsize=axis_tick_size )
                axs[i].tick_params( axis='y', labelsize=axis_tick_size )

                axs[i].legend( fontsize=axis_tick_size ) #, loc='upper right' )
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)

        fig.text(0.022, 0.94, 'A', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
        fig.text(0.35, 0.94, 'B', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
        fig.text(0.68, 0.94, 'C', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
        plt.tight_layout()
        plt.show()



    def du_flat_plot( self, split_val=0.3 ):
        import statsmodels.api as sm # for V
        def du_model_clay( v, v_0=5, a=0.07, b=16.75, c=0.95 ):
            return a*np.tanh((v-v_0)/b)+c

        def duV_model_clay( V, a=0.07528, b=0.7741, V0=3000 ):
            return a*np.log10( np.clip(V,-np.inf,V0) ) + b
        # settings
        size = 60
        axis_label_size = 15
        axis_tick_size = 13
        frac = 95

        x_lims = (0,70) #(5,50)
        
        y_lims = (0.8,1.5)

        # data
        (( du_k, v, Bq ), labels), var_list = self.vars_3D_figs( 'du_k', 'rate_avg', 'Bq__value_avg' )
        (( du, unit, V ), labels_2), var_list_2 = self.vars_3D_figs( 'du_value_avg', 'unit', 'V' )




        for i, label in enumerate( labels ):
            xV_lims = [(1e2, 1e4),(1e-1, 1e1),(1e1, 1e3)][i]

            if 'iller' not in label: continue

            unique_units = list( set(list(unit[i])) )

            fig, axs = plt.subplots( 1,2, figsize=(10,4.00) ) #silt_figures: (10,4.00) # (6,4.00)
            # markers
            m, msf = self.marker_style( label )
            mc, mc_edge = self.get_color( label, axs[0] )


            axs[0].scatter( # datapoints
                v[i], du_k[i], # xy (Qt log transformed)
                label=label, # in legend
                s=size*msf,
                marker=m, # type
                ec=mc_edge, fc=mc, # colors
                alpha=.3, # no fading
                zorder=3
            )

            resV = []
            resduk = []
            duV_k = np.array([])
            q = 0
            for u_unit in unique_units:
                q += 1
                idx = unit[i]==u_unit
                du_unit = du[i][idx]
                V_unit = V[i][idx]

                ref_estimate = sm.nonparametric.lowess( 
                    du_unit,
                    V_unit, 
                    frac=.8, # use 50-80% of points
                    xvals=1000 #20mm/s
                )[0]

                du_unit_k = du_unit/ref_estimate
                duV_k = np.append(duV_k, du_unit_k)

                t_label=label
                if q>1:t_label = ''
                axs[1].scatter( # datapoints
                    V_unit, du_unit_k, # xy (Qt log transformed)
                    label=t_label, # in legend
                    s=size*msf,
                    marker=m, # type
                    ec=mc_edge, fc=mc, # colors
                    alpha=.3, # no fading
                    zorder=3
                )
                axs[1].set_xscale('log')

                resV += V_unit.astype(str).tolist()
                resduk += du_unit_k.astype(str).tolist()


            if 'iller' in label:
                break_point = 3000
                v_s = np.linspace( 4, 70, 1000)
                V_s = np.logspace(np.log10(1e2), np.log10(1e4))
                model_k = du_model_clay(v_s)
                model_Vk = duV_model_clay( V_s )
                offs = self.calc_offset( v_s, model_k, v[i], du_k[i], frac/100 )

                model_lb = model_k-offs
                model_ub = model_k+offs

                model_label= r'$k_{\Delta u_2}=0.07$'+ '·tanh' + r'$\left( \frac{ \frac{4 \: v}{v_0}-1}{3.35} \right) + 0.95$'
                axs[0].plot(v_s, model_k, lw=2, c=mc_edge, ls='-', label=model_label, zorder=10) #Model at ' r'$B_q$' + '=' + str(np.round(model_Bq,2))
                axs[0].plot(v_s,model_ub, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)
                axs[0].plot(v_s, model_lb, lw=1, c=self.get_color('dark_grey')[0], ls='-', zorder=8)
                axs[0].fill_between( v_s, model_lb, model_ub, facecolor=self.get_color('light_grey')[0], alpha=1, label=str(frac) + '% of points within ±' + str(round(offs,3)), zorder=1 )
                axs[0].plot(x_lims,[1]*len(x_lims), lw=1, c=mc_edge, ls='-', zorder=2)

                # V: vertical horizontal lina at k=1, V=1e3
                axs[1].plot((1e2, 1e4),[1]*len(x_lims), lw=1, c=mc_edge, ls='-', zorder=2)
                axs[1].plot([1e3]*2,[0.7,1], lw=1.5, c=mc_edge, ls='-', zorder=2)
                axs[1].plot([3e3]*2,y_lims, lw=1.5, c=mc_edge, ls='--', zorder=2)

                # V: combined model
                du_max = duV_model_clay( break_point )
                axs[1].plot( V_s[V_s<break_point], model_Vk[V_s<break_point], label='Best fit line ' + r'$V<3000$', lw=2, ls='-', c=(0,0,0), zorder=4)
                axs[1].plot( V_s[V_s>break_point], model_Vk[V_s>break_point], label='Best fit ' + r'$V=3000$', lw=2, ls='--', c=self.get_color('model')[0], zorder=4)
                

                axs[1].plot( (1e2, 1e4), [du_max*0.9]*2, label='90% of fit at ' + r'$V=3000$', lw=2, ls='--', c=self.get_color('Stjørdal: Halsen')[0], zorder=10) #s
                V_undr = np.power(10,(du_max*0.9-0.7741)/0.07528)
                V_undr = np.around(V_undr/5, decimals=0)*5 # to nearest 5

                last_px = None
                for px, py in zip( [break_point, break_point, V_undr, V_undr], [du_max, du_max*0.9, du_max*0.9, y_lims[0] ]):                
                    axs[1].plot(px, py, marker='o', ms=7, mfc=(1,1,1), mec=( 0, 0, 0 ),mew=1.8, zorder=12 ) #s
                    if last_px is not None:
                        axs[1].add_patch( patches.FancyArrowPatch((last_px, last_py), (px, py), arrowstyle='->', mutation_scale=20, zorder=11) )
                    last_px, last_py = px, py

                axs[1].plot(px, py, marker='o',linestyle='None', ms=10, mfc=self.get_color('model')[0], label=r'$V_{undr.}=$' + str(int(round(V_undr,0))), mec=( 0, 0, 0 ),mew=1.8, zorder=14 ) #s


                for r in [ 5, 20, 60 ]:
                    x_offset = -.7 if r in [ 10, 25 ] else .7
                    ls = '-' if r==20 else '--'
                    y_vals = [y_lims[0], du_model_clay(r) ]
                    axs[0].plot( [r]*2, y_vals, lw=1.5, ls=ls, c=(0,0,0), zorder=2)

                    y_offs = 0

                    t = axs[0].text( r+x_offset, y_vals[1]+0.085+y_offs, np.round(y_vals[1],3), horizontalalignment='center', verticalalignment='bottom', zorder=11, fontsize=axis_tick_size )
                    t.set_bbox( dict(facecolor=(1,1,1), alpha=1, edgecolor=(0,0,0), zorder=11) )
                    axs[0].plot( r, y_vals[1], marker='o', ms=10, mfc=self.get_color('model')[0], mec=( 0, 0, 0 ),mew=1.8, zorder=11 )

            axs[0].set_xticks([0,5,10,20,30,40,50,60,70])
            axs[1].set_xticks([1e2, 1e3, 3e3, 1e4])
            
            axs[1].get_xaxis().set_major_formatter(ScalarFormatter())

            fig.text(0.03, 0.94, 'A', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
            fig.text(0.51, 0.94, 'B', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')


            #ax.set_xscale('log')
            axs[0].set_xlim( x_lims )
            axs[0].set_ylim( y_lims )

            axs[1].set_xlim( xV_lims )
            axs[1].set_ylim( y_lims )#(0.7, 1.2) )            

            axs[0].set_xlabel( self.get_label( var_list[1] ), fontsize=axis_label_size )
            axs[0].set_ylabel( self.get_label( var_list[0] ), fontsize=axis_label_size*1.2 )

            axs[1].set_xlabel( self.get_label( var_list_2[2] ), fontsize=axis_label_size )
            axs[1].set_ylabel( r'$k_{\Delta u_{2.V}} = \frac{\Delta u_2}{\Delta u_{2.V.ref}}$', fontsize=axis_label_size*1.2 )
            axs[0].tick_params( axis='x', labelsize=axis_tick_size )
            axs[0].tick_params( axis='y', labelsize=axis_tick_size )
            axs[1].tick_params( axis='x', labelsize=axis_tick_size )
            axs[1].tick_params( axis='y', labelsize=axis_tick_size )

            axs[0].legend( fontsize=axis_tick_size ) #, loc='upper right' )
            axs[1].legend( fontsize=axis_tick_size ) #, loc='upper right' )
            axs[1].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[0].spines['top'].set_visible(False)
            plt.tight_layout()
            plt.show()



    def du_flat_plot2( self, split_val=0.3 ):
        import statsmodels.api as sm # for V
        def du_model_clay( v, v_0=5, a=0.07, b=16.75, c=0.95 ):
            return a*np.tanh((v-v_0)/b)+c

        def duV_model_clay( V, a=0.07528, b=0.7741, V0=3000 ):
            return a*np.log10( np.clip(V,-np.inf,V0) ) + b
        # settings
        size = 60
        axis_label_size = 15
        axis_tick_size = 13
        frac = 95

        x_lims = (0,70) #(5,50)
        y_lims = (0.8,1.5)

        # data
        (( du_k, v, Bq ), labels), var_list = self.vars_3D_figs( 'du_sigma_v0_eff', 'rate_avg', 'Bq__value_avg' )
        (( du, unit, V ), labels_2), var_list_2 = self.vars_3D_figs( 'du_sigma_v0_eff', 'unit', 'V' )

        fig, axs = plt.subplots( 1,2, figsize=(10,4.00) ) #silt_figures: (10,4.00) # (6,4.00)
        for i, label in enumerate( labels ):
            #xV_lims = [(1e2, 1e4),(1e-1, 1e1),(1e1, 1e3)][i]

            if '2-' not in label: continue #'sand'

            unique_units = list( set(list(unit[i])) )


            # markers
            m, msf = self.marker_style( label )
            mc, mc_edge = self.get_color( label, axs[0] )

            axs[0].scatter( # datapoints
                v[i], du_k[i], # xy (Qt log transformed)
                label=label, # in legend
                s=size*msf,
                marker=m, # type
                ec=mc_edge, fc=mc, # colors
                alpha=.3, # no fading
                zorder=3
            )

            axs[1].scatter( # datapoints
                v[i], du_k[i], # xy (Qt log transformed)
                label=label, # in legend
                s=size*msf,
                marker=m, # type
                ec=mc_edge, fc=mc, # colors
                alpha=.3, # no fading
                zorder=3
            )

        axs[0].set_xticks([0,5,10,20,30,40,50,60,70])
        #axs[1].set_xticks([1e2, 1e3, 3e3, 1e4])
        axs[1].get_xaxis().set_major_formatter(ScalarFormatter())

        fig.text(0.03, 0.94, 'A', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')
        fig.text(0.51, 0.94, 'B', fontsize=axis_label_size*1.8, color=(0,0,0), verticalalignment='center', horizontalalignment='center')

        #ax.set_xscale('log')
        #axs[0].set_xlim( x_lims )
        #axs[0].set_ylim( y_lims )

        #axs[1].set_xlim( xV_lims )
        #axs[1].set_ylim( y_lims )#(0.7, 1.2) )            

        axs[0].set_xlabel( self.get_label( var_list[1] ), fontsize=axis_label_size )
        axs[0].set_ylabel( self.get_label( var_list[0] ), fontsize=axis_label_size*1.2 )

        axs[1].set_xlabel( self.get_label( var_list_2[2] ), fontsize=axis_label_size )
        axs[1].set_ylabel( r'$k_{\Delta u_{2.V}} = \frac{\Delta u_2}{\Delta u_{2.V.ref}}$', fontsize=axis_label_size*1.2 )
        axs[0].tick_params( axis='x', labelsize=axis_tick_size )
        axs[0].tick_params( axis='y', labelsize=axis_tick_size )
        axs[1].tick_params( axis='x', labelsize=axis_tick_size )
        axs[1].tick_params( axis='y', labelsize=axis_tick_size )

        axs[0].legend( fontsize=axis_tick_size ) #, loc='upper right' )
        axs[1].legend( fontsize=axis_tick_size ) #, loc='upper right' )
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        plt.tight_layout()
        plt.show()



    def split_by_x( self, x, y, z, split_val ):
        res_x, res_y, res_z = [], [], []

        idx = x<split_val
        tmp_x, tmp_y, tmp_z = x[idx], y[idx], z[idx]
        if tmp_x.size!=0: # append if not empty
            res_x.append(tmp_x), res_y.append(tmp_y), res_z.append(tmp_z)

        tmp_x, tmp_y, tmp_z = x[~idx], y[~idx], z[~idx]
        if tmp_x.size!=0:
            res_x.append(tmp_x), res_y.append(tmp_y), res_z.append(tmp_z)

        return res_x, res_y, res_z

    def classification_charts( self, letter='' ):
        size = 7
        dl_perc = 0.6
        var_id = { # links var_names in chart to index in all_vars
            'Qt': 49,
            'Fr': 58,
            'Rf': 62,
            'Bq': 53,
            'du_n': 69,
            'qe': 41,
            'qt': 16,
            'fs': 20,
        }

        all_vars = self.data_vars()
        k = 0
        for m in model_defs:
            k += 1
            if k<1: continue

            clf = general_model( model_def=model_defs[m], fill_regions=self.fill )
            clf.prep_figure( letter=letter )

            # get data
            var_list = [ all_vars[var_id[clf.x_var]],all_vars[var_id[clf.y_var]], all_vars[8] ]
            (x,y,v), labels = self.get_vars( var_list )

            for i, label in enumerate(labels):
                m, msf = self.marker_style( label )
                mc, mc_edge = self.get_color( label, clf.ax )
                if False: 
                    if '2-' not in label: mc=(1,1,1) # help with rate analysis
                if clf.name == 'D_u2/sigma_v0_eff-Q_t  (Schneider 2008)':
                    qts = np.logspace(np.log10(1), np.log10(1000))
                    dus = 0.3* qts
                    clf.ax.plot( dus, qts, ls='--', lw=1.5, c=(0,0,0), zorder=4 )

                # separate by rates
                base_label = label.replace('NGTS: ', '').replace('Stjørdal: ', '').replace('-Flotten', '')
                slow_mc = tuple( min(rgba*(1+dl_perc),1) for rgba in mc )
                fast_mc = tuple( max(rgba*(1-dl_perc),0) for rgba in mc )

                v_i = np.array( v[i] )
                slow_mask = (v_i<15)
                normal_mask = (v_i>=15) & (v_i<=25)
                fast_mask = (v_i>25)

                slow_x = np.array(x[i])[slow_mask]
                slow_y = np.array(y[i])[slow_mask]
                slow_label = base_label + ' v<15mm/s'
                clf.ax.plot( slow_x, slow_y, ls='none', markersize=size*msf, marker=m, mfc=slow_mc, mec=mc_edge, label=slow_label, zorder=5 )

                normal_x = np.array(x[i])[normal_mask]
                normal_y = np.array(y[i])[normal_mask]
                normal_label = base_label + ' 15≥v≤25mm/s'
                clf.ax.plot( normal_x, normal_y, ls='none', markersize=size*msf, marker=m, mfc=mc, mec=mc_edge, label=normal_label, zorder=10 )

                fast_x = np.array(x[i])[fast_mask]
                fast_y = np.array(y[i])[fast_mask]
                fast_label = base_label + ' v>25mm/s'
                clf.ax.plot( fast_x, fast_y, ls='none', markersize=size*msf, marker=m, mfc=fast_mc, mec=mc_edge, label=fast_label, zorder=5 )

            #legend = clf.ax.legend( framealpha=1, edgecolor=(0,0,0))
            
            plt.show()


    def field_decision_chart( self ):
        hatch_color = {
            'partially' : (0.9, 0.9, 0.9),
            'drained' : (1,1,1),
            'undrained' : (1,1,1),
            'rate_range': (237/255,28/255,46/255,.3)
        }
        size = 10
        self.size_ticks = 12
        self.size_small_ticks = 10
        self.size_titles = 16

        d = 2*np.sqrt(1000/np.pi) # cone diameter in mm^2
        cv_lims = ( 1e-1, 1e5 )
        v_lims = ( 1e-2, 1e3 )#( 1e-3, 1e4 ) # 

        fig, ax = plt.subplots()

        all_vars = self.data_vars() # 7:cv(mm2/s), 8:v (mm/s)
        var_list = [all_vars[7], all_vars[8]]
        (x,y), labels = self.get_vars( var_list )

        # draw undrained- partially drained and drained penetration regions and V-isolines
        V_vals = [0.1, 0.3, 1, 3, 10, 30, 100, 125]
        t_ang = 43
        cv_s = np.array(cv_lims)
        v_i = {}
        cv_i = {}
        for V in V_vals:
            prefix, ls, lw = '', '--', 1
            if V==0.1 or V==100: prefix, ls, lw = 'V=', '-', 1.5                        
            v_i[V] = V*cv_s/d
            cv_i[V] = v_lims[1]*d/V
            ax.plot( cv_s, v_i[V], c=(0,0,0), zorder=2, ls=ls, lw=lw ) # isolines
            if V!=V_vals[-1]:
                if v_i[V][-1]>v_lims[1]:
                    ax.text( cv_i[V], v_lims[1] *1.15, prefix+str(V), verticalalignment='bottom', horizontalalignment='center', size=self.size_small_ticks )
                else:
                    ax.text( cv_lims[1] *1.1, v_i[V][-1], prefix+str(V), verticalalignment='center', size=self.size_small_ticks )
            else:
                ax.text( cv_lims[0] * .9, v_i[V][0], 'V='+str(V), horizontalalignment='right', verticalalignment='center', size=self.size_small_ticks )
        # shade partially drained region
        ax.fill_between(cv_lims, v_i[0.3], v_i[100], facecolor=hatch_color['partially'], zorder=1)
        # annotate regions
        t = ax.text( 1.26, 0.20, 'partially\ndrained', rotation=t_ang, verticalalignment='center', horizontalalignment='center', size=self.size_ticks*.9, zorder=4 )
        t.set_bbox( dict(facecolor=hatch_color['partially'], alpha=1, edgecolor=hatch_color['partially']) )
        t = ax.text( 23, 151, 'undrained', rotation=t_ang, verticalalignment='center', horizontalalignment='center', size=self.size_ticks*.9, zorder=4 )
        t.set_bbox( dict(facecolor=hatch_color['undrained'], alpha=1, edgecolor=hatch_color['undrained']) )
        t = ax.text( 2300, 0.8, 'drained', rotation=t_ang, verticalalignment='center', horizontalalignment='center', size=self.size_ticks*.9, zorder=4 )
        t.set_bbox( dict(facecolor=hatch_color['drained'], alpha=1, edgecolor=hatch_color['drained']) )
        ax.add_patch( patches.FancyArrowPatch((0.3,0.84),(5,0.045) , arrowstyle='<->', mutation_scale=30, lw=1.2, zorder=2) )# partially
        ax.add_patch( patches.FancyArrowPatch((30,85),(9,300) , arrowstyle='->', mutation_scale=30, lw=1.2, zorder=2) )# undrained
        ax.add_patch( patches.FancyArrowPatch((500,4.1),(2000,1) , arrowstyle='->', mutation_scale=30, lw=1.2, zorder=2) )# drained


        # draw datapoints
        for i, label in enumerate(labels):
            m, msf = self.marker_style( label )
            mc, mc_edge = self.get_color( label, ax )
            ax.plot( x[i], y[i], ls='none', markersize=size*msf, marker=m, mfc=mc, mec=mc_edge, label=label, zorder=5 )


        # draw penetration rate range
        for ls, lw, v_val in zip(['-','--','-'],[.9,2,.9],[15,20,25]):
            ax.plot( cv_lims, [v_val]*2, c=(0,0,0), lw=lw, ls=ls, zorder=4 )
        ax.fill_between(cv_lims, [15,15], [25,25], facecolor=hatch_color['rate_range'], zorder=2, label='standard rate range')


        # draw & annotate material boundaries
        cv_bounds = [ cv_lims[0], 4e0/3, 1.42e3/3, cv_lims[1] ]
        v_txt = 600
        ch_txt = ['clays', 'silts', 'sands']
        ch_txt_mask = [ hatch_color['drained'],hatch_color['drained'], hatch_color['partially'] ]
        for i, an in enumerate(ch_txt):
            ax.add_patch( patches.FancyArrowPatch((cv_bounds[i],v_txt), (cv_bounds[i+1],v_txt), arrowstyle='<->', mutation_scale=20, zorder=2) )
            #text
            txt_x = 10**((np.log10(cv_bounds[i])+np.log10(cv_bounds[i+1]))/2)
            t = ax.text( txt_x,v_txt,an, horizontalalignment='center',verticalalignment='center', fontsize=self.size_ticks )
            t.set_bbox( dict(facecolor=ch_txt_mask[i], alpha=1, edgecolor=ch_txt_mask[i]) )
            # vertical lines
            ax.plot( [cv_bounds[i]]*2, v_lims, c=(0,0,0), lw=1.2, ls='-', zorder=3 )


        # logscale formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.set_xlabel( self.get_label(var_list[0]), fontsize=self.size_titles )
        ax.set_ylabel( self.get_label(var_list[1]), fontsize=self.size_titles )
        ax.tick_params( axis='x', labelsize=self.size_ticks )
        ax.tick_params( axis='y', labelsize=self.size_ticks )

        ax.set_xlim( cv_lims )
        ax.set_ylim( v_lims )

        plt.tight_layout()
        ax.grid( which='major', color=(.4,.4,.4), lw=0.8 )
        ax.grid( which='minor', color=(.9,.9,.9), lw=0.4 )        

        plt.legend(loc='lower right')
        plt.show()


    def V_2d_Nplot( self, vars, lims=None, logs=None, legend=0, letter='A' ):
        fontsize_titles, fontsize_ticks = 16, 12
        hatch = {
            'gray': (0.9, 0.9, 0.9),
            'white': (1, 1, 1),
        }

        V_lims = ( 1e-3, 3e4 )
        V_boundaries = [ 0.3, 100 ]
        V_arrow_bounds= [V_lims[0]] + V_boundaries + [V_lims[1]]
        V_txt = ['drained', 'partially\ndrained', '  undrained']
        V_txt_hatch = [ hatch['white'], hatch['gray'], hatch['white'] ]
        V_txt_alpha = [ 0, 1, 0 ]

        if logs is None: logs=[False]*len(vars)

        all_vars = self.data_vars()
        fig, axs = plt.subplots(1,len(vars), figsize=(10,5))

        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

        x_label = self.get_label( all_vars[11] ) # V
        V_s = np.logspace( np.log10(V_lims[0]), np.log10(V_lims[1]), 200 )

        for i, some_var in enumerate(vars):
            ( x, y ), labels = self.get_vars( [all_vars[11],all_vars[some_var]] )
            # draw datapoints
            for j, label in enumerate(labels):
                m, msf = self.marker_style( label )
                mc, mc_edge = self.get_color( label, axs[i] )
                axs[i].plot( x[j], y[j], ls='none', markersize=8*msf, marker=m, mfc=mc, mec=mc_edge, label=label, zorder=10 )
                if lims is not None: axs[i].set_ylim( lims[i] )
                axs[i].set_xlim( V_lims )

                # log_scales
                axs[i].set_xscale('log')
                axs[i].xaxis.set_major_formatter(formatter)
                if logs[i]:
                    axs[i].set_yscale('log')
                    axs[i].yaxis.set_major_formatter(formatter)

                # V-lines & zones
                y_lims = axs[i].get_ylim()
                axs[i].plot( [V_boundaries[0]]*2, y_lims, ls='--', c=(0,0,0), zorder=0)
                axs[i].plot( [V_boundaries[1]]*2, y_lims, ls='--', c=(0,0,0), zorder=0)
                axs[i].fill_between( V_s, y_lims[0], y_lims[1], where=np.logical_and(V_s>V_boundaries[0],V_s<V_boundaries[1]), facecolor=hatch['gray'], zorder=-1)

                for k, an in enumerate(V_txt):                    
                    y_txt = y_lims[1] - (y_lims[1]-y_lims[0])/15
                    y_arr = (y_lims[1] + y_txt)/2
                    if logs[i]: 
                        y_txt = 10**(np.log10(y_lims[1]) - (np.log10(y_lims[1])-np.log10(y_lims[0]))/15)
                        y_arr = 10**( (np.log10(y_lims[1])+np.log10(y_txt))/2 )

                    axs[i].add_patch( patches.FancyArrowPatch((V_arrow_bounds[k],y_arr), (V_arrow_bounds[k+1],y_arr), arrowstyle='<->', mutation_scale=20, zorder=1) )
                    txt_x = 10**((np.log10(V_arrow_bounds[k])+np.log10(V_arrow_bounds[k+1]))/2)
                    t = axs[i].text( txt_x,y_txt, an, horizontalalignment='center',verticalalignment='top', fontsize=fontsize_ticks*.8 )
                    t.set_bbox( dict(facecolor=V_txt_hatch[k], alpha=V_txt_alpha[k], edgecolor=V_txt_hatch[k]) )

                if letter!='':
                    t = axs[i].text( -0.32, 1, chr( ord(letter)+i ), horizontalalignment='center',verticalalignment='top', fontsize=fontsize_titles*1.4, transform=axs[i].transAxes )

                # axis labels
                axs[i].set_xlabel( x_label, fontsize=fontsize_titles )
                axs[i].set_ylabel( self.get_label(all_vars[some_var]), fontsize=fontsize_titles )
                axs[i].tick_params( axis='x', labelsize=fontsize_ticks )
                axs[i].tick_params( axis='y', labelsize=fontsize_ticks )
                if i==legend: axs[i].legend( loc='best' )
                #turn off top and right spine
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)

        plt.tight_layout()
        plt.show()


    def V_2d_6plot( self, vars, lims=None, logs=None, letter='A' ):
        fontsize_titles, fontsize_ticks = 16, 12
        hatch = {
            'gray': (0.9, 0.9, 0.9),
            'white': (1, 1, 1),
        }

        init_spliter = self.split_by

        V_lims = ( 1e-3, 3e4 )
        V_boundaries = [ 0.3, 100 ]
        V_arrow_bounds= [V_lims[0]] + V_boundaries + [V_lims[1]]
        V_txt = ['drained', 'partially\ndrained', '  undrained']
        V_txt_hatch = [ hatch['white'], hatch['gray'], hatch['white'] ]
        V_txt_alpha = [ 0, 1, 0 ]

        if logs is None: logs=[False]*len(vars)

        all_vars = self.data_vars()
        fig, axs = plt.subplots(2,3, figsize=(10,11))

        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

        x_label = self.get_label( all_vars[11] ) # V
        V_s = np.logspace( np.log10(V_lims[0]), np.log10(V_lims[1]), 200 )

        for i, some_var in enumerate(vars):
            p, q = int(i>2), i%3
            
            self.split_by = ['site','unit','position'][0]
            ( _, _ ), site_labels = self.get_vars( [all_vars[11],all_vars[some_var]] )
            self.split_by = ['site','unit','position'][1]
            ( x, y ), labels = self.get_vars( [all_vars[11],all_vars[some_var]] )
            
            # draw datapoints
            for j, label in enumerate(labels):
                m, msf = self.marker_style( label )
                mc, mc_edge = self.get_color( label, axs[p][q] )                
                axs[p][q].plot( x[j], y[j], ls='none', markersize=7*msf, marker=m, mfc=mc, mec=mc_edge, label=label, zorder=10 )
                if lims is not None: axs[p][q].set_ylim( lims[i] )
                axs[p][q].set_xlim( V_lims )

                # log_scales
                axs[p][q].set_xscale('log')
                axs[p][q].xaxis.set_major_formatter(formatter)
                if logs[i]:
                    axs[p][q].set_yscale('log')
                    axs[p][q].yaxis.set_major_formatter(formatter)

                # V-lines & zones
                y_lims = axs[p][q].get_ylim()
                axs[p][q].plot( [V_boundaries[0]]*2, y_lims, ls='--', c=(0,0,0), zorder=0)
                axs[p][q].plot( [V_boundaries[1]]*2, y_lims, ls='--', c=(0,0,0), zorder=0)
                axs[p][q].fill_between( V_s, y_lims[0], y_lims[1], where=np.logical_and(V_s>V_boundaries[0],V_s<V_boundaries[1]), facecolor=hatch['gray'], zorder=-1)

            for k, an in enumerate(V_txt):                
                y_txt = y_lims[1] - (y_lims[1]-y_lims[0])/15
                y_arr = (y_lims[1] + y_txt)/2
                if logs[i]: 
                    y_txt = 10**(np.log10(y_lims[1]) - (np.log10(y_lims[1])-np.log10(y_lims[0]))/15)
                    y_arr = 10**( (np.log10(y_lims[1])+np.log10(y_txt))/2 )

                axs[p][q].add_patch( patches.FancyArrowPatch((V_arrow_bounds[k],y_arr), (V_arrow_bounds[k+1],y_arr), arrowstyle='<->', mutation_scale=20, zorder=1) )
                txt_x = 10**((np.log10(V_arrow_bounds[k])+np.log10(V_arrow_bounds[k+1]))/2)
                t = axs[p][q].text( txt_x,y_txt, an, horizontalalignment='center',verticalalignment='top', fontsize=fontsize_ticks*.8 )
                t.set_bbox( dict(facecolor=V_txt_hatch[k], alpha=V_txt_alpha[k], edgecolor=V_txt_hatch[k]) )

            if letter!='':
                t = axs[p][q].text( -0.32, 1, chr( ord(letter)+i ), horizontalalignment='center',verticalalignment='top', fontsize=fontsize_titles*1.4, transform=axs[p][q].transAxes )

            # axis labels
            axs[p][q].set_xlabel( x_label, fontsize=fontsize_titles )
            axs[p][q].set_ylabel( self.get_label(all_vars[some_var]), fontsize=fontsize_titles )
            axs[p][q].tick_params( axis='x', labelsize=fontsize_ticks )
            axs[p][q].tick_params( axis='y', labelsize=fontsize_ticks )
            #if i==legend: axs[i].legend( loc='best' )
            #turn off top and right spine
            axs[p][q].spines['right'].set_visible(False)
            axs[p][q].spines['top'].set_visible(False)

        self.split_by = init_spliter # return settings

        plt.tight_layout()
        plt.subplots_adjust( bottom=0.17 )

        # generate site labels
        site_labels = list(site_labels)
        handles, labels = axs[1][0].get_legend_handles_labels()
        site_label = { k[1:]:v for k, v in zip(labels, handles) if k in ['0-0', '1-1', '2-2']  }
        for k,v in {'-0':site_labels[0], '-1':site_labels[1], '-2':site_labels[2]}.items():
            some_line = site_label.pop(k)
            some_new_line = axs[1][0].plot([], ls='', marker=some_line.get_marker(), mfc=(1,1,1), mec=some_line.get_mec(), ms=some_line.get_ms()*1.25, label=some_line.get_label() )[0]
            site_label[v + ' (N=' + k[1:] + ')'] = some_new_line



        handles, labels = axs[1][2].get_legend_handles_labels()
        unit_label = { k[1:]:v for k, v in zip(labels, handles) }
        for k,v in unit_label.copy().items():
            some_line = unit_label.pop( k )
            some_new_line = axs[1][2].plot([], ls='', marker='s', mfc=some_line.get_mfc(), mec=(0,0,0), ms=some_line.get_ms()*1.25, label=some_line.get_label() )[0]
            unit_label['N' + k] = some_new_line



        legend_a = axs[1][0].legend( site_label.values(), site_label.keys(), frameon=False, loc='upper left', bbox_to_anchor=(0, -0.17), title='Site', title_fontproperties={'size': 12, 'weight':'bold'}, fontsize=12 )
        legend_b = axs[1][2].legend( unit_label.values(), unit_label.keys(), frameon=False, loc='upper right', bbox_to_anchor=(.8, -0.17), ncol=5, title='Unit name', title_fontproperties={'size': 12, 'weight':'bold'}, fontsize=12  )

        legend_a._legend_box.align = 'left'
        legend_b._legend_box.align = 'left'

        plt.show()


    def V_2d_6plot_2( self, vars, lims=None, logs=None, letter='' ):
        fontsize_titles, fontsize_ticks = 16, 12
        fs_scale = .8
        hatch = {
            'gray': (0.9, 0.9, 0.9),
            'white': (1, 1, 1),
        }

        init_spliter = self.split_by

        V_lims = ( 1e-3, 3e4 )
        V_boundaries = [ 0.3, 100 ]
        V_arrow_bounds= [V_lims[0]] + V_boundaries + [V_lims[1]]
        V_txt = ['drained', 'partially\ndrained', '  undrained']
        V_txt_hatch = [ hatch['white'], hatch['gray'], hatch['white'] ]
        V_txt_alpha = [ 0, 1, 0 ]

        if logs is None: logs=[False]*len(vars)

        all_vars = self.data_vars()
        fig, axs = plt.subplots(2,3, figsize=(10,7))

        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

        x_label = self.get_label( all_vars[11] ) # V
        V_s = np.logspace( np.log10(V_lims[0]), np.log10(V_lims[1]), 200 )

        for i, some_var in enumerate(vars):
            p, q = int(i>2), i%3
            
            self.split_by = ['site','unit','position'][0]
            ( _, _ ), site_labels = self.get_vars( [all_vars[11],all_vars[some_var]] )
            self.split_by = ['site','unit','position'][1]
            ( x, y ), labels = self.get_vars( [all_vars[11],all_vars[some_var]] )
            
            # draw datapoints
            for j, label in enumerate(labels):
                m, msf = self.marker_style( label )
                mc, mc_edge = self.get_color( label, axs[p][q] )                
                axs[p][q].plot( x[j], y[j], ls='none', markersize=7*msf, marker=m, mfc=mc, mec=mc_edge, label=label, zorder=10 )
                if lims is not None: axs[p][q].set_ylim( lims[i] )
                axs[p][q].set_xlim( V_lims )

                # log_scales
                axs[p][q].set_xscale('log')
                axs[p][q].xaxis.set_major_formatter(formatter)
                if logs[i]:
                    axs[p][q].set_yscale('log')
                    axs[p][q].yaxis.set_major_formatter(formatter)

                # V-lines & zones
                y_lims = axs[p][q].get_ylim()
                axs[p][q].plot( [V_boundaries[0]]*2, y_lims, ls='--', c=(0,0,0), zorder=0)
                axs[p][q].plot( [V_boundaries[1]]*2, y_lims, ls='--', c=(0,0,0), zorder=0)
                axs[p][q].fill_between( V_s, y_lims[0], y_lims[1], where=np.logical_and(V_s>V_boundaries[0],V_s<V_boundaries[1]), facecolor=hatch['gray'], zorder=-1)

            for k, an in enumerate(V_txt):                
                y_txt = y_lims[1] - (y_lims[1]-y_lims[0])/15
                y_arr = (y_lims[1] + y_txt)/2
                if logs[i]: 
                    y_txt = 10**(np.log10(y_lims[1]) - (np.log10(y_lims[1])-np.log10(y_lims[0]))/15)
                    y_arr = 10**( (np.log10(y_lims[1])+np.log10(y_txt))/2 )

                axs[p][q].add_patch( patches.FancyArrowPatch((V_arrow_bounds[k],y_arr), (V_arrow_bounds[k+1],y_arr), arrowstyle='<->', mutation_scale=20, zorder=5) )
                txt_x = 10**((np.log10(V_arrow_bounds[k])+np.log10(V_arrow_bounds[k+1]))/2)
                t = axs[p][q].text( txt_x,y_txt, an, horizontalalignment='center',verticalalignment='top', fontsize=fontsize_ticks*.8 )
                t.set_bbox( dict(facecolor=V_txt_hatch[k], alpha=V_txt_alpha[k], edgecolor=V_txt_hatch[k]) )

            if letter!='':
                t = axs[p][q].text( -0.32, 1, chr( ord(letter)+i ), horizontalalignment='center',verticalalignment='top', fontsize=fontsize_titles*1.4*fs_scale, transform=axs[p][q].transAxes )

            # axis labels
            axs[p][q].set_xlabel( x_label, fontsize=fontsize_titles*fs_scale )
            axs[p][q].set_ylabel( self.get_label(all_vars[some_var]), fontsize=fontsize_titles*fs_scale )                        
            axs[p][q].tick_params( axis='x', labelsize=fontsize_ticks )
            axs[p][q].tick_params( axis='y', labelsize=fontsize_ticks )
            #if i==legend: axs[i].legend( loc='best' )
            #turn off top and right spine
            axs[p][q].spines['right'].set_visible(False)
            axs[p][q].spines['top'].set_visible(False)


        self.split_by = init_spliter # return settings

        plt.tight_layout()
        plt.subplots_adjust( bottom=0.17 )

        # generate site labels
        site_labels = list(site_labels)
        handles, labels = axs[1][0].get_legend_handles_labels()
        site_label = { k[1:]:v for k, v in zip(labels, handles) if k in ['0-0', '1-1', '2-2']  }
        for k,v in {'-0':site_labels[0], '-1':site_labels[1], '-2':site_labels[2]}.items():
            some_line = site_label.pop(k)
            some_new_line = axs[1][0].plot([], ls='', marker=some_line.get_marker(), mfc=(1,1,1), mec=some_line.get_mec(), ms=some_line.get_ms()*1.25, label=some_line.get_label() )[0]
            site_label[v + ' (N=' + k[1:] + ')'] = some_new_line



        handles, labels = axs[1][2].get_legend_handles_labels()
        unit_label = { k[1:]:v for k, v in zip(labels, handles) }
        for k,v in unit_label.copy().items():
            some_line = unit_label.pop( k )
            some_new_line = axs[1][2].plot([], ls='', marker='s', mfc=some_line.get_mfc(), mec=(0,0,0), ms=some_line.get_ms()*1.25, label=some_line.get_label() )[0]
            unit_label['N' + k] = some_new_line



        legend_a = axs[1][0].legend( site_label.values(), site_label.keys(), frameon=False, loc='upper left', bbox_to_anchor=(0, -0.17), title='Site', title_fontproperties={'size': 12, 'weight':'bold'}, fontsize=12 )
        legend_b = axs[1][2].legend( unit_label.values(), unit_label.keys(), frameon=False, loc='upper right', bbox_to_anchor=(.8, -0.17), ncol=5, title='Unit name', title_fontproperties={'size': 12, 'weight':'bold'}, fontsize=12  )

        legend_a._legend_box.align = 'left'
        legend_b._legend_box.align = 'left'

        fig.savefig('var_overview.png', dpi=600)
        plt.show()

    def data_vars( self ):
        return self.data.columns.values.tolist()


    def get_vars( self, var_list ):   
        result = []

        # Group the DataFrame by unique values in the splitter column
        grouped = self.data.groupby( self.split_by )
        keys = grouped.groups.keys()

        # Iterate over the groups and extract the desired columns
        for var in var_list:
            var_data = []
            for group_name, group_data in grouped:
                var_values = group_data[var].values.tolist()
                var_data.append(np.array(var_values))
            result.append(var_data)

        return result, keys


    def vars_3D_figs( self, x='Bq__value_avg', y='rate_avg', z='qt_k' ):        
        
        if any ([ i=='' for i in [ x, y, z] ]):
            vars = self.data_vars()
            print('select 3 variables:')
            for i, var in enumerate(vars):
                print( i, var )
            exit()
        #all_vars = self.data_vars() # 19:qt.k, 51:Qt.k, 48: Qt, 8:v, 11:V 52:Bq        
        var_list = [ x, y, z ]
        return self.get_vars( var_list ), var_list


    def log_tick_formatter( self, val, pos=None ):
        return f"$10^{{{val:g}}}$"


    def plot_2D( self, var_list, figsize=(7,5), size=12, logx=False, logy=False ):
        (x, y), labels = self.get_vars( var_list )

        fig, ax = plt.subplots(figsize=figsize)

        for i, label in enumerate(labels):
            m, msf = self.marker_style( label )
            mc, mc_edge = self.get_color( label, ax )
            ax.plot( x[i], y[i], ls='none', markersize=size*msf, marker=m, mfc=mc, mec=mc_edge, label=label )

        ax.set_xlabel( self.get_label(var_list[0]), fontsize=self.size_titles )
        ax.set_ylabel( self.get_label(var_list[1]), fontsize=self.size_titles )

        ax.tick_params(axis='x', labelsize=self.size_ticks)
        ax.tick_params(axis='y', labelsize=self.size_ticks)

        if logx:ax.set_xscale('log')
        if logy:ax.set_yscale('log')
        
        #turn off top and right spine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        return fig, ax


    def get_label( self, var ):
        ax_labels = {
            'du_k_star':r'$K^*_{\Delta u_2}=\frac{\Delta u_2}{\Delta u_{2.ref} + 300}$',
            'Bq__k1':r'$K_{B_{q.1}}=\frac{B_q + 1}{B_{q.ref} + 1}$',
            'fs_k': r'$k_{f_s}=\frac{f_s}{f_{s.ref}}$' + '  (-)',
            'qt_k': r'$k_{q_t}=\frac{q_t}{q_{t.ref}}$' + '  (-)',
            'du_k': r'$k_{\Delta u_2}=\frac{\Delta u_2}{\Delta u_{2.ref}}$' + '  (-)',
            'Qt__k': r'$k_{Q_t}=\frac{Q_t}{Q_{t.ref}}$' + '  (-)',
            'V' : r'$V=\frac{v \cdot d}{c_v}$' + '  (-)',
            'qt_sigma_v0_eff': r"$\frac{q_t}{\sigma'_{v}}$" + '  (-)',
            'rate_avg': 'Penetration rate, ' + r'$v$' +  ' (mm/s)',
            'cv': r'$c_v$' + ' (' + r'$mm^2/s)$',
            'qt_value_avg': 'q' + r'$_t$' + ' (kPa)',
            'fs_value_avg': 'f' + r'$_s$' + ' (kPa)',
            'u_value_avg': 'u' + r'$_2$' + ' (kPa)',
            'du_value_avg': r'$\Delta$' + 'u' + r'$_2$' + ' (kPa)',
            'du_value_avg': r'$\Delta$' + 'u' + r'$_2$' + '=' + 'u' + r'$_2$' + '-u' + r'$_0$' +  ' (kPa)',
            'Qt__value_avg': r"$Q_t=\frac{ q_t - \sigma _v }{ \sigma ' _v }$" + ' (-)',
            'Fr__value_avg': r'$F_r=\frac{ f_s }{ q_t - \sigma _v }$' +' (%)',
            'Bq__value_avg': r'$B_q=\frac{ u_2 - u_0 }{ q_t - \sigma _v }$' + ' (-)',
            'fs_sigma_v0_eff': r"$f_{sn}=\frac{ f_s }{ \sigma ' _v }$" + ' (-)',

            'qc': 'Measured cone resistance, q' + r'$_c$' + ' [kPa]',#',r'$_a$' + r'$_\phi=$'
            'qt': 'Corrected cone resistance, q' + r'$_t$' + ' [kPa]',
            'fs': 'Sleeve frictional resistance, f' + r'$_s$' + ' [kPa]',
            'ft': 'Corrected sleeve frictional resistance, f' + r'$_t$' + ' [kPa]',
            'u': 'Pore pressure, u' + r'$_2$' + ' [kPa]',
            'du': 'Excess pore water pressure, '+ r'$\Delta$' + 'u' + ' [kPa]',
            'qn': 'Net cone resistance q' + r'$_n$' + ' [kPa]',
            'qe': 'Effective cone resistance, q' + r'$_e$' + ' [kPa]',
            'du_sig_v0_eff': r'$\Delta$' + 'u / ' + r'$\sigma$' + '\'' + r'$_v$' + r'$_0$' + '[-]',
            'Qt': 'Normalized cone resistance, Q' + r'$_t$' + ' [-]',            
            'Bq': 'Pore pressure parameter, B' + r'$_q$' + ' [-]',
            'Fr': 'Normalized friction ratio, F' + r'$_r$' + ' [-]',
            'Rf': 'Friction ratio, R' + r'$_f$' + ' [-]',

        }
        if var in ax_labels: return ax_labels[var]
        return var


    def marker_style( self, label ):
        markers = {
            'NGTS: Tiller-Flotten': ( 'o', 1 ),
            'NGTS: Øysand': ( 'v', 1 ),
            'Stjørdal: Halsen': ( 'X', 1.3 )
        }
        for i in range(20):
            markers[ '0-' + str(i) ] = ( 'o', 1 )
            markers[ '1-' + str(i) ] = ( 'v', 1 )
            markers[ '2-' + str(i) ] = ( 'X', 1.3 )
        if label in markers: return markers[label]
        return 'o', 1


    def get_color( self, label, ax=None ):
        edgecolor = ( .1, .1, .1 )
        if not self.fill: return ((1,1,1,1), edgecolor) # still fill with white

        colors={
            'mesh' : (0,0,0,0.7), #grey
            'model': (237/255,28/255,46/255), # NPRA red
            'dark_grey': (.6,.6,.6),
            'light_grey': (.9,.9,.9),
            'NGTS: Tiller-Flotten': (93/255,184/255,46/255), # NPRA green
            'NGTS: Øysand': (0/255,142/255,194/255), # NPRA blue
            'Stjørdal: Halsen': (255/255,150/255,0/255), # NPRA orange
            '-0':(120/255,170/255,210/255),
            '-1':(255/255,180/255,110/255),
            '-2':(130/255,200/255,130/255),
            '-3':(230/255,125/255,125/255),
            '-4':(190/255,165/255,215/255),
            '-5':(185/255,155/255,150/255),
            '-6':(240/255,170/255,220/255),
            '-7':(180/255,180/255,180/255),
            '-8':(215/255,215/255,120/255),
        }

        # generate unis color palette
        #n=10
        #cmap = mpl.colors.LinearSegmentedColormap.from_list( "", ['red','orange','green','blue'] )
        #cmap = mpl.colormaps['tab20']
        #c_list = [ cmap(k) for k in np.arange(0,1,1/n) ]
        ##np.random.shuffle(c_list)
        #for i, c in enumerate( c_list ):
        #    colors[ '-' + str(i) ] = c


        if label in colors: return ( colors[label], edgecolor ) # complete match
        if label[1:] in colors: return ( colors[ label[1:] ], edgecolor ) # complete match

        if ax is not None: # access built in color cycler
            c = next(ax._get_lines.prop_cycler)['color'].lstrip('#')
            c_rgb = tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4))
            return ( c_rgb, edgecolor )

        # B/W
        return ( ( 1, 1, 1 ), edgecolor )


    def calc_bf_params( self, x, y, logx=False, logy=False ):
        res = []
        for some_xs, some_ys in zip( x, y ):
            if logy: pass # sorry
            if logx: some_xs = np.log10(some_xs)
            res.append( np.polyfit(some_xs, some_ys, 1) )           
        return res


    def calc_offset( self, v_s, model_k, tv, tz, perc, logX=False ):
        m_k = np.interp( tv, v_s, model_k )
        if logX: m_k = np.interp(np.log(tv), np.log(v_s), model_k)
        
        offset = np.max( np.abs( m_k-tz ) )
        d_offs = offset/len( tv )

        offset += d_offs
        while offset-d_offs>0:
            under = sum((m_k-offset)>tz)
            above = sum((m_k+offset)<tz)
            if (1-perc) < (above + under)/len(tz): break

            offset -= d_offs

        return offset+d_offs
#offs = self.calc_offset( v_s, model_k, tv, tz, 0.95 )


if __name__=='__main__':
    results = ['results.csv', 'results_du_k_star.csv', 'results_Bq__k1.csv', 'results_warped_no_logs.csv', 'results_unwarped.csv'][1] # best/worse/worst
    split_by = ['site','unit','position'][0]

    plotter = rate_plotter( result_file=results, split_by=split_by, fill=True ) #,  split_by= # 'results.csv'
    all_vars = plotter.data_vars()
    #print(all_vars)
    #plotter.field_decision_chart() 
    #plotter.V_2d_Nplot( vars=[16,20,28], lims=[(0,1.4e4),(0,1e2),(-4e2,1.2e3)], logs=[True,True,False], legend=2 ) #qt:16, fs:20, u2:24, du:28 :: 
    #plotter.V_2d_Nplot( vars=[48,56,52], lims=[(1e-1,1e3),(2e-2,1e1),(-5e-1,1.5e0)], logs=[True,True,False], legend=1 ) #Qt:48, Fr:56, Bq:52
    #plotter.V_2d_Nplot( vars=[49,67,53], lims=[(1e0,1e3),(1e-2,2e0),(-8e-1,1.5e0)], logs=[True,True,False], legend=2 ) #Qt:48, fs/sigv0_eff:65, Bq:52
    #[(100,1.4e4),(1,1e2),(-4e2,1.2e3), (1e0,1e3),(1e-2,2e0),(-8e-1,1.5e0)]
    #plotter.V_2d_6plot( vars=[16,20,28,49,67,53], lims=[(100,1.4e4),(1,1e2),(-200,1000), (1e0,1e3),(1e-2,2e0),(-8e-1,1.5e0)], logs=[True, True, False, True,True,False] ) #qt:16, fs:20, u2:24, du:28 :: 
    #plotter.V_2d_6plot_2( vars=[16,20,28,49,67,53], lims=[(400,3e4),(2,2e2),(-200,1200), (2e0,2e2),(1e-2,2e0),(-2.5e-1,1.4)], logs=[True, True, False, True,True,False] ) #qt:16, fs:20, u2:24, du:28 :: 
    #plotter.classification_charts( letter='' )
    plotter.k_3d_plot()
    # plotter.k_flat_plots()
    #plotter.du_flat_plot()
    
    #plotter.k_flat_plots2()
    #plotter.k_flat_plots3()

    #plotter.du_flat_plot2() - no good