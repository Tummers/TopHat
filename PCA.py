# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:34:02 2019

@author: User
"""

from Course import Course
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklPCA

class PrincipleComponents:
    def __init__(self, course, savename):
        """
        creates an instance of PCA, taking course (an n x m array of 1s and 0s) as an argument
        """
        self.course = course
        self.answers = course.answers
        self.lecture_indices = course.lecture_indices
        self.savename = str(savename)
        self.q_no = len(self.answers[0])
    
    def qxq(self, mark_l=True, remove_no_response=True):
        """
        produces a grid of q by q squares, where each has a value based on correlation of correct scores
        """
        self.grid = np.zeros([self.q_no, self.q_no])
        indices = []
        
        for i in range(self.q_no): 
            for j in range(i + 1): # symmetric about x = y so only need to go to i in j 
                q2_ans = self.answers[:, j]
                q1_ans = self.answers[:, i]
                
                if(remove_no_response == True): #if removing no response answers, make a list of students answering both
                    for k in range(len(self.answers)):
                    
                        if(q1_ans[k] >= 0 and q2_ans[k] >= 0):
                            indices.append(k)
                else:
                    indices = np.arange(len(self.answers))
                    
                q1_ans = q1_ans[indices]
                q2_ans = q2_ans[indices]
                
                diff = q1_ans - q2_ans 
                
                matching = diff[diff==0] # only need to check how many are zero, both qs right/wrong means q1 - q2 = 0
                unmatching = diff[diff!=0]
                score = len(matching) - len(unmatching) # for anti correlation subtract non matching from matching
                
                self.grid[i, j] = score / len(diff)
                self.grid[j, i] = self.grid[i, j]
                
                indices = []
                
            progress = (i / self.q_no) * 100
            sys.stdout.write("Plotting Correlations: %.2f%%\r" %(progress))
            sys.stdout.flush()
                
        self.plotWindow = plt.figure(figsize=[12, 12])
        self.plotWindow.clear()
        
        
        plt.title("Correlation of Correctness by Question for " + self.savename)
        plt.imshow(self.grid, cmap=plt.get_cmap("coolwarm"), vmin=-1, vmax=1, origin="upper")
        if(mark_l == True):
            lines = np.copy(self.lecture_indices)
            lines = lines.astype(float)
            lines -= .5
            plt.vlines(lines, ymin=0, ymax=self.q_no)
            plt.hlines(lines, xmin=0, xmax=self.q_no)
        figuresave = self.savename + "_correctness_correlation.png"
        self.plotWindow.savefig(figuresave)
        
        return self.grid                
        
    def components(self, components):
        sys.stdout.write("Finding principle components            \r")
        sys.stdout.flush()
        
        #producing eigenvalues/vectors from qxq matrix
        correlation = self.qxq()
        
        eigen_vals, eigen_vecs = np.linalg.eig(correlation)
        
        eigen_pair = []
        for i in range(len(eigen_vals)):
            eigen_pair.append([np.abs(eigen_vals[i]), eigen_vecs[:,1]])
            
        #identifying no of PCs
        full = np.sum(eigen_vals)
        tot = 0
        k = 0
        eigen_vals_prop = []
        while tot < 0.8:
            prop = eigen_vals[k] / full
            eigen_vals_prop.append(prop)
            tot += prop
            k += 1
            
        #splitting qs into principle components
        pca = sklPCA(n_components=k)
        x = StandardScaler().fit_transform(np.transpose(self.answers))
        pcs = pca.fit_transform(x)
        
        #finding loadings of questions on components
        loadings = np.zeros([len(pcs[0]), len(pcs)])
        loading_eigen_values = np.zeros(len(pcs[0]))
        loading_eigen_vectors = np.transpose(pcs)          
        
        eigen_squares = np.transpose(pcs ** 2) #square of each value
        
        for i in range(len(loadings)):
            loading_eigen_values[i] = np.sum(eigen_squares[i]) #sum of squares
            
            for j in range(len(loadings[0])):
                load_value =  loading_eigen_vectors[i, j] * np.sqrt(loading_eigen_values[i])
                loadings[i, j] = load_value
         
        #first dimension is PCs, second is Qs
        
        #plotting heatmap of loadings and questions
        fig = plt.figure(figsize=(12, 8))
        figuresave = self.savename + "_Loadings"
        plt.clf()
        plt.imshow(loadings, cmap="coolwarm", vmin=-1, vmax=1) #assume stat between -1 and 1
        plt.title("Principle Component Loadings " + self.savename)
        plt.xlabel("Principle Component")
        plt.ylabel("Question Number")
        fig.savefig(figuresave)
        
        #establishing lower bounds for contribution to PC
        lower_bound = np.zeros(len(pcs[0])) # number of pcs
        
        for i in range(len(lower_bound)):
            lower_bound[i] = np.max(loadings[i]) * 0.8 # 80% of the maximum loading 
            
        #selecting questions above lower bound
        pc_conts = []
        for i in range(len(loadings)):
            contributions_i = []
            for j in range(len(loadings[i])):
                if(loadings[i, j] >= lower_bound[i]):
                    contributions_i.append(j)
            pc_conts.append([])
            pc_conts[i] = contributions_i
            
        #writing contributing questions to each pc to a txt
        txt_name = self.savename + "_contributions.txt"
        f = open(txt_name, "w+")
        for i in range(k):
            f.write("PC " + str(i + 1) + ": (Lower Limit: " + str(lower_bound[i]) + ")")
            for j in range(len(pc_conts[i])):
                f.write("\n     " + self.course.q_dict[pc_conts[i][j]]) #str(pc_conts[i][j]))
            f.write("\n\n")
        f.close()
        
        #producing scree plot
        indices = np.arange(k)
        indices += 1
        fig = plt.figure(figsize=(12, 8))
        figuresave = self.savename + "_" + str(k) + "_PCs.png"
        plt.clf()
        plt.plot(indices, np.abs(eigen_vals_prop))
        plt.title("Principle Components " + self.savename)
        plt.xlabel("Principle Components")
        plt.ylabel("Explained Variance")
                
        fig.savefig(figuresave)
        
    
def main():
    
    infile = sys.argv[1] + ".xlsx"
    savename = sys.argv[2] 
    
    a = Course(infile, savename +".txt", participation_points = False)
    
    b = PrincipleComponents(a, savename)
    #b.qxq(mark_l=False, remove_no_response=True)
    b.components(2)
main()