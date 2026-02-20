# -*- coding: utf-8 -*-
"""
Data Storage and Tracking Module

This module provides the DataStorage class, designed to track, store, and 
display training metrics (like Loss and Accuracy) in real-time during 
model training loops.

Created on Tue Aug  9 12:33:43 2022
@author: Diyar Altinses, M.Sc.
"""

import time
from typing import List, Union, Any

import torch


class DataStorage:
    """
    Stores training data while offering customizable console prints.

    This class automatically tracks time and computes moving averages 
    for 'Loss' and 'Acc' if they are included in the initialization names.
    """

    def __init__(
        self, 
        names: List[str], 
        average_window: int = 100, 
        show: int = 2, 
        line: int = 50, 
        header: int = 500, 
        step: int = 1, 
        precision: int = 3, 
        name: str = "", 
        auto_show: bool = True
    ):
        """
        Initializes the DataStorage instance.

        Args:
            names (List[str]): List of metric names to store.
            average_window (int, optional): Window size for moving average. Defaults to 100.
            show (int, optional): Batches between each new print. Defaults to 2.
            line (int, optional): Batches before printing on a new line. Defaults to 50.
            header (int, optional): Batches before reprinting column headers. Defaults to 500.
            step (int, optional): Storage step size (e.g., step=2 halves memory usage). Defaults to 1.
            precision (int, optional): Decimal places for printed values. Defaults to 3.
            name (str, optional): Optional identifier name for the storage instance. Defaults to "".
            auto_show (bool, optional): Automatically display values. Defaults to True.
        """
        self.Name = name
        self.Names = ["Time"] + names
        
        self.AverageWindow = average_window
        self.Show = show
        self.Line = line
        self.Header = header
        self.Step = step
        self.Precision = precision
        self.Batch = 0
        self.autoshow = auto_show

        # Automatically append moving average trackers
        if "Loss" in self.Names:
            self.Names.append("avg. Loss")
        if "Acc" in self.Names:
            self.Names.append("avg. Acc")

        # Formatting lengths for console display
        self.Lens = [len(n) + 5 for n in self.Names]
        
        # Initialize empty lists for all tracked metrics
        self.StoredValues = {n: [] for n in self.Names}
        self.Columns = len(self.Names)
        self.DumpValues = {}

    def Store(self, vals: List[Union[int, float, torch.Tensor]], force: int = 0) -> None:
        """
        Stores a new row of data in the internal StoredValues dictionary.

        Args:
            vals (List[Union[int, float, torch.Tensor]]): Metrics to store. Order 
                must match the `names` provided during initialization.
            force (int, optional): If > 0, forces storage and display for the 
                given batch number regardless of the `step` parameter. Defaults to 0.
        """
        # Save absolute start time on the first batch
        if self.Batch == 0:
            self.DumpValues["TimeStart"] = time.time()
            
        if self.Batch % self.Step == 0 or force > 0:
            
            # Calculate elapsed time in minutes
            elapsed_time = (time.time() - self.DumpValues["TimeStart"]) / 60.0
            self.StoredValues["Time"].append(elapsed_time)
            
            for col in range(1, self.Columns):
                current_name = self.Names[col]
                
                if current_name == "avg. Loss":
                    # Calculate moving average for Loss
                    recent_loss = self.StoredValues["Loss"][-self.AverageWindow:]
                    avg_loss = torch.sum(torch.tensor(recent_loss)) / self.AverageWindow
                    self.StoredValues[current_name].append(avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss)
                    
                elif current_name == "avg. Acc":
                    # Calculate moving average for Accuracy
                    recent_acc = self.StoredValues["Acc"][-self.AverageWindow:]
                    avg_acc = torch.sum(torch.tensor(recent_acc)) / self.AverageWindow
                    self.StoredValues[current_name].append(avg_acc.item() if isinstance(avg_acc, torch.Tensor) else avg_acc)
                    
                else:
                    # Store standard values, extracting floats from tensors if necessary
                    val = vals[col - 1]
                    if isinstance(val, torch.Tensor):
                        self.StoredValues[current_name].append(val.cpu().detach().item())
                    else:
                        self.StoredValues[current_name].append(val)
            
            # Handle console display logic
            if self.autoshow:
                if self.Batch == 0:
                    self._GetHead()
                    self._Display()
                    print("")
                else:
                    if self.Batch % self.Show == 0 or force > 0:
                        self._Display()
                    if self.Batch % self.Line == 0:
                        print("")
                    if self.Batch % self.Header == 0:
                        self._GetHead()
                        
        self.Batch += 1

    def _Display(self) -> None:
        """Formats and prints the most recently stored values to the console."""
        outstr = "\r"
        args: List[str] = []
        
        for col in range(self.Columns):
            val = self.StoredValues[self.Names[col]][-1]
            outstr += "{:s}"

            if isinstance(val, float):
                val_str = str(round(val, self.Precision))
            elif isinstance(val, torch.Tensor):
                val_str = str(round(val.item(), self.Precision))
            else:
                val_str = str(val)
                
            # Apply padding based on column length
            args.append(val_str + (self.Lens[col] - len(val_str)) * " ")
            
        print(outstr.format(*args), end="")

    def _GetHead(self) -> None:
        """Prints the column headers formatted to match value widths."""
        print("")
        header_string = ""
        
        for col in range(self.Columns):
            name = self.Names[col]
            header_string += name + (self.Lens[col] - len(name)) * " "
            
        print(header_string)