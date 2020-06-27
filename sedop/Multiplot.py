"""
Multiplot.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-27.

Description: Make multipanel plots with shared axes, with or without AxesGrid.
     
"""

import pylab as pl
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

# Matplotlibrc defaults
figsize = pl.rcParams['figure.figsize']
wspace = pl.rcParams['figure.subplot.wspace']
hspace = pl.rcParams['figure.subplot.hspace']
left = pl.rcParams['figure.subplot.left']
right = pl.rcParams['figure.subplot.right']
bottom = pl.rcParams['figure.subplot.bottom']
top = pl.rcParams['figure.subplot.top']
dtop = 1 - top
dbottom = bottom
dright = 1 - right
dleft = left

def AxisConstructor(nr, nc, panel): 
    return pl.subplot(nr, nc, panel)

class multiplot(object):
    def __init__(self, dims = (2, 2), padding = 0, panel_size = None, useAxesGrid = True,
        share_all = True, aspect = False):
        """
        dims = (rows, columns)
        padding = separation of subplots in inches
        panel_size = fraction of default figsize in each dimension
        nd = dimensionality of data
        """
        
        self.dims = dims
        self.padding = padding
        
        if panel_size is None: 
            self.panel_size = np.array([figsize[0], figsize[1]])
        else: 
            
            xsize = figsize[0]
            if panel_size[0] > 1:
                
                for i in xrange(panel_size[0] - 1):
                    xsize += (right - left) * figsize[0]
            
            ysize = figsize[1]    
            if panel_size[1] > 1:
                for i in xrange(panel_size[1] - 1):
                    ysize += (top - bottom) * figsize[1]
            
            self.panel_size = np.array([xsize, ysize])
                    
        if share_all and not useAxesGrid:
            pl.rcParams['figure.subplot.wspace'] = self.padding
            pl.rcParams['figure.subplot.hspace'] = self.padding        
                
        self.fig = pl.figure(1, self.panel_size)
        
        self.Naxes = np.prod(self.dims)
        self.elements = np.reshape(xrange(self.Naxes), (self.dims[0], self.dims[1]))
        self.xaxes = self.elements[-1]
        self.yaxes = zip(*self.elements)[0]                  
        self.lowerleft = self.elements[-1][0]
        self.lowerright = self.elements[-1][-1]
        self.upperleft = self.elements[0][0]
        self.upperright = self.elements[0][-1]
                
        # Set up grid
        if useAxesGrid:        
            self.grid = AxesGrid(self.fig, 111, nrows_ncols = self.dims, axes_pad = self.padding, 
                aspect = aspect, share_all = share_all)
                                
        else:            
            self.grid = {}
            for i in xrange(np.prod(self.dims)):
                self.grid[i] = AxisConstructor(self.dims[0], self.dims[1], i + 1)                        
                                
    def fix_ticks(self, noxticks = False, noyticks = False, style = None):
        """
        Call once all plotting is done, will eliminate redundant tick marks and what not.
        """
                
        # Make sure we don't double up on xticks
        for i in self.xaxes:
            
            xticks = list(self.grid[i].get_xticks())
            dx = np.diff(xticks)[0]
            limits = self.grid[i].get_xlim()
            
            if xticks[-1] > limits[1]: xticks.pop(-1)
            
            if (limits[1] - xticks[-1]) < dx / 2.: loc = -1
            else: loc = None
                      
            if i == self.lowerright:
                self.grid[i].set_xticks(xticks)                                
            else:                 
                self.grid[i].set_xticks(xticks[0:loc])
                
            if style is not None: self.grid[i].ticklabel_format(style = style)   
                
        # Make sure we don't double up on yticks
        for i in self.yaxes:
            
            yticks = list(self.grid[i].get_yticks())
            dy = np.diff(yticks)[0]
            limits = self.grid[i].get_ylim()
            
            if yticks[-1] > limits[1]: yticks.pop(-1)
                        
            if (limits[1] - yticks[-1]) < dy / 2.: loc = -1
            else: loc = None
                                    
            if i == self.upperleft:
                self.grid[i].set_yticks(yticks)                        
            else:                 
                self.grid[i].set_yticks(yticks[0:loc])        
    
            if style is not None: self.grid[i].ticklabel_format(style = style)
            
        # Remove ticks from non-edge axes     
        for i in xrange(self.Naxes):
            if i not in self.xaxes: 
                self.grid[i].set_xticklabels([])
            elif i not in self.yaxes: 
                self.grid[i].set_yticklabels([])
            else:
                continue    
        
        if noxticks:
            for i, element in enumerate(self.grid):
                self.grid[i].set_xticks([])

        if noyticks:
            for i, element in enumerate(self.grid):
                self.grid[i].set_yticks([])
    
    def global_xlabel(self, label, xy = (0.5, 0.025), size = 'x-large'):
        """
        Set shared xlabel.
        """        
        
        self.fig.text(xy[0], xy[1], label, 
            ha = 'center', va = 'center', size = size)
            
    def global_ylabel(self, label, xy = (0.025, 0.5), size = 'x-large'):
        """
        Set shared ylabel.
        """        
        
        self.fig.text(xy[0], xy[1], label, 
            ha = 'center', va = 'center', rotation = 'vertical', size = size)