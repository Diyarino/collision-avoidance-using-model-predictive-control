# -*- coding: utf-8 -*-
"""
Plot Configuration Module

This module provides standard configuration settings for Matplotlib to 
ensure consistent, publication-ready plot formatting across the project.

Created on Tue Jul 15 09:39:53 2025
@author: Altinses
"""

import shutil
import matplotlib.pyplot as plt


def configure_plt(check_latex: bool = True) -> None:
    """
    Configures Matplotlib's global settings for font sizes, LaTeX rendering,
    and general aesthetics.

    Args:
        check_latex (bool, optional): If True, checks if LaTeX is installed on 
            the system and enables it for Matplotlib text rendering. 
            Defaults to True.
    """
    if check_latex:
        # Check for LaTeX using the modern shutil library
        if shutil.which('latex'):
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
            
    plt.rc('font', family='Times New Roman')
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    # Define standard sizes
    small_size = 13
    small_medium = 14
    medium_size = 16
    big_medium = 18
    big_size = 20
    
    # Apply configurations
    plt.rc('font', size=small_size)          # Default text sizes
    plt.rc('axes', titlesize=big_medium)     # Fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # Fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # Fontsize of the x tick labels
    plt.rc('ytick', labelsize=small_size)    # Fontsize of the y tick labels
    plt.rc('legend', fontsize=small_medium)  # Legend fontsize
    plt.rc('figure', titlesize=big_size)     # Fontsize of the figure title
    
    plt.rc('grid', c='0.5', ls='-', lw=0.5)
    
    # Apply global grid and layout configurations
    plt.grid(True)
    plt.tight_layout()
    plt.close()


if __name__ == '__main__':
    configure_plt()