# -*- coding: utf-8 -*-
"""
Experiment Folder Generation Module

This module provides utilities to automatically generate, structure, and 
manage directories for training machine learning models, ensuring organized 
storage for setups, data, images, and model weights.

Created on Tue Aug  9 12:15:06 2022
@author: Diyar Altinses, M.Sc.
"""

import os
import time
from typing import Tuple, Optional


class GenerateFolder:
    """
    Generates and manages a structured directory hierarchy for training trials.
    """

    def __init__(self, GenerateAll: bool = False):
        """
        Initializes the folder generation class.

        Args:
            GenerateAll (bool, optional): If True, instantly generates the 
                train, setup, and data folders upon instantiation. Defaults to False.
        """
        self.trainfolder: Optional[str] = None
        self.setupfolder: Optional[str] = None
        self.imgfolder: Optional[str] = None
        self.datafolder: Optional[str] = None
        self.netfolder: Optional[str] = None
        self.tablefolder: Optional[str] = None
        
        self.setup_idx: int = 1
        
        if GenerateAll:
            self.GenerateTrainFolder(generate=True)
            self.GenerateSetupFolder(generate=True)
            self.GenerateDataFolder(generate=True)
            
    def GenerateTrainFolder(self, generate: bool = False, location: str = None, name: str = '') -> 'GenerateFolder':
        """
        Defines and optionally creates the main training directory based on the current time.

        Args:
            generate (bool, optional): Directly creates the folder on the disk. Defaults to False.
            location (str, optional): Alternative base path. Defaults to None (uses CWD).
            name (str, optional): Suffix string to append to the folder name. Defaults to ''.

        Returns:
            GenerateFolder: Returns self for method chaining.
        """
        time_str = time.strftime("%y_%m_%d__%H_%M_%S") + name
        
        if location is None:
            location = os.getcwd()
            
        self.trainfolder = os.path.join(location, "training", time_str)
        
        if generate:
            os.makedirs(self.trainfolder, exist_ok=True)
            
        return self
    
    def GenerateSetupFolder(self, generate: bool = False, location: str = None) -> 'GenerateFolder':
        """
        Defines and optionally creates a specific setup sub-directory.

        Args:
            generate (bool, optional): Directly creates the folder on the disk. Defaults to False.
            location (str, optional): Alternative base path. Defaults to None (uses trainfolder).

        Returns:
            GenerateFolder: Returns self for method chaining.
        """        
        if location is None:
            location = self.trainfolder
            
        self.setupfolder = os.path.join(location, str(self.setup_idx).zfill(3) + "_setup")
        
        if generate:
            os.makedirs(self.setupfolder, exist_ok=True)
            
        return self
    
    def GenerateDataFolder(self, generate: bool = False, location: str = None) -> Tuple[str, str, str, str]:
        """
        Defines and optionally creates the data, image, model, and table sub-directories.

        Args:
            generate (bool, optional): Directly creates the folders on the disk. Defaults to False.
            location (str, optional): Alternative base path. Defaults to None (uses setupfolder).

        Returns:
            Tuple[str, str, str, str]: Paths for (datafolder, imgfolder, netfolder, tablefolder).
        """
        if location is None:
            location = self.setupfolder
            
        self.datafolder = os.path.join(location, "data")
        self.imgfolder = os.path.join(location, "img")
        self.netfolder = os.path.join(location, "model")
        self.tablefolder = os.path.join(location, "table")
        
        if generate:
            os.makedirs(self.datafolder, exist_ok=True)
            os.makedirs(self.imgfolder, exist_ok=True)
            os.makedirs(self.netfolder, exist_ok=True)
            os.makedirs(self.tablefolder, exist_ok=True)
            
        return self.datafolder, self.imgfolder, self.netfolder, self.tablefolder

    def GetFolder(self) -> Tuple[str, str, str, str, str, str]:
        """
        Outputs all current folder paths tracked by the instance.

        Returns:
            Tuple[str, str, str, str, str, str]: Paths for (trainfolder, setupfolder, 
                datafolder, imgfolder, netfolder, tablefolder).
        """
        return (self.trainfolder, self.setupfolder, self.datafolder, 
                self.imgfolder, self.netfolder, self.tablefolder)
    
    def __call__(self, setup_step: bool = False) -> Tuple[str, str, str, str, str, str]:
        """
        Updates the internal setup index and returns all active folders.

        Args:
            setup_step (bool, optional): If True, increments the setup index and 
                creates a new setup folder. Defaults to False.

        Returns:
            Tuple[str, str, str, str, str, str]: All tracked folder paths.
        """
        if setup_step:
            self.setup_idx += 1
            self.GenerateSetupFolder(generate=True)
            
        return (self.trainfolder, self.setupfolder, self.datafolder, 
                self.imgfolder, self.netfolder, self.tablefolder)


def generate_trainfolder(generate: bool = False, location: str = None) -> Tuple[str, str, str, str, str]:
    """
    Standalone function to quickly generate a standard training directory structure.

    Args:
        generate (bool, optional): Directly creates the folders on the disk. Defaults to False.
        location (str, optional): Alternative base path. Defaults to None (uses CWD).

    Returns:
        Tuple[str, str, str, str, str]: Paths for (train_folder, img_folder, 
            config_folder, model_folder, data_folder).
    """
    time_str = time.strftime("%y_%m_%d__%H_%M_%S")
    
    if location is None:
        location = os.getcwd()
        
    train_folder = os.path.join(location, "training", time_str)
    img_folder = os.path.join(train_folder, "img")
    config_folder = os.path.join(train_folder, "config")
    model_folder = os.path.join(train_folder, "model")
    data_folder = os.path.join(train_folder, "data")
    
    if generate:
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(data_folder, exist_ok=True)
        
    return train_folder, img_folder, config_folder, model_folder, data_folder