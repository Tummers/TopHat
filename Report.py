# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:17:16 2019

@author: User
"""
import sys
from Course import Course

def main():
    #read in data
    sys.stdout.write("Progress: %.1f%%\r"
                              %(0))
    sys.stdout.flush()
    
    if(len(sys.argv) < 2):
        print("Command should include <title of input file> <title of output file>")
        exit()
    
    infile = sys.argv[1] + ".xlsx"
    report_file = sys.argv[2]
    
    #clearing file
    f = open(report_file + ".txt", "w") 
    f.close()   

    a = Course(infile, report_file, participation_points=True)

    a.missed_score_correlation()
    
main()