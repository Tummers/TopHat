# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:44:24 2019

@author: Tom Summers
"""

import numpy as np
import xlrd as xl
import matplotlib.pyplot as plt
import sys

class Course:
    
    def __init__(self, infile, report_file=None, participation_points=False):
        """
        takes a 2d array (data) and forms students and a dict of questions
        """
        sys.stdout.write("Initialising Course Object.\r")
        sys.stdout.flush()
        self.part_point = participation_points # are there points for participation
        self.infile = infile
        self.report_file = report_file + ".txt"
        
        titles, q_indices = self.readin_titles() # getting titles and positions in spreadsheet of questions
        self.q_dict = {}
        for i in range(len(titles)):
            self.q_dict[i] = titles[i]  
        
        self.q_indices = np.array(q_indices)
        
        data = self.readin_data() # getting responses to questions from spreadsheet
        self.answers = data
        self.student_no = len(data)
        self.q_no = len(data[0])
        
        if(self.part_point == True):
            start = 0
            stop = self.search_dict("Remove Dielectric from Capacitor")
            self.remove_part_point(start, stop)
            start2 = self.search_dict("Charge and Magnet")
            stop2 = self.search_dict("Two Rings")
            self.remove_part_point(start2, stop2)
            start3 = self.search_dict("Ideal Window Pane")
            stop3 = self.search_dict("Ideal Window Pane")
            self.remove_part_point(start3, stop3)
            start4 = self.search_dict("Evasive maneuver")
            stop4 = self.search_dict("Drunk in the wind")
            self.remove_part_point(start4, stop4)
            
        self.lecture_indices = self.get_lecture_indices() # identifying positions of lectures
        
        #self.response_graph()
        self.mark_absent() # marking where students non-response should be an incorrect response
        #self.mark_wrong()
        self.mark_dichot()
        
        
        self.missed = self.missed_qs() # forming an array of total missed qs by student
         # plotting the distribution of missed qs
        
        self.pre_cut_scores = self.produce_scores() # total scores of all students, including those that will be omitted
        self.removed_no = self.omit_no_response() # removing students with low response rate, recording number removed
        
        self.scores = self.produce_scores() # finding total score of each student
        
        self.cd = None
        self.sh = None
        self.ca = None
        self.kr = None
        self.se = None
        self.fe = None        
    
    def readin_titles(self):
        """
        reads in the titles using the exported format 
        """
        
        book = xl.open_workbook(self.infile)
        sheet = book.sheet_by_index(0)
        
        titles = []
        final_titles =[]
        row_of_titles = 2
        
        q_start = 6
        q_end = sheet.ncols
        
        #get_titles and positions of weights
        title_row = np.array(sheet.row_slice(rowx=row_of_titles, start_colx=q_start, end_colx=q_end))
        length_t = len(title_row)
        sys.stdout.write("                             \r")
        for i in range(length_t):
            titles.append(str(title_row[i].value))
            prog_t = (i/length_t) * 100
            sys.stdout.write("Reading Titles: %.1f%%\r"
                              %(prog_t))
            sys.stdout.flush()
            
        q_indices = []
        for i in range(length_t):
            if(titles[i] != "Weight"):
                final_titles.append(titles[i])
                q_indices.append(i)
                prog_w = (i/length_t) * 100
                sys.stdout.write("Removing Weights: %.1f%%\r"
                              %(prog_w))
                sys.stdout.flush()
        
        return final_titles, q_indices
    
    
    def readin_data(self):
        """
        reads in the data from the exported format, returns as an array. first axis is students, second is their answers
        """
        
        book = xl.open_workbook(self.infile)
        sheet = book.sheet_by_index(0)
        
        q_start = 6
        q_end = sheet.ncols
        
        data = []
        final_data = []
        
        for i in range(3, sheet.nrows):
            row = sheet.row_slice(rowx=i, start_colx=q_start, end_colx=q_end)
            for j in range(len(row)):
                row[j] = row[j].value
            data.append(row)
        
        #removing weight values
        for i in range(len(data)):
            final_data.append([])
            for j in range(len(self.q_indices)):
                final_data[i].append(data[i][self.q_indices[j]])
        
        #converting -- values to -1
        for i in range(len(final_data)):
            for j in range(len(final_data[0])):
                if(final_data[i][j] == "--"):
                    final_data[i][j] = -1
        
        
        final_data = np.array(final_data)
        
        return final_data
    
    def count_ans(self, val):
        """
        returns the number of occurences of val in self.answers
        """
        count = 0
        for i in range(len(self.answers)):
            for j in range(len(self.answers[0])):
                if(self.answers[i, j] == val):
                    count += 1
                    
        return count
    
    def search_dict(self, q_title):
        """
        returns index of a q_title
        """
        flipped_dict = dict(map(reversed, self.q_dict.items()))
        
        index = flipped_dict.get(q_title)
        
        return index

    
    def get_lecture_indices(self):
        """
        returns an array giving the start and length of each lecture by question number
        """
        book = xl.open_workbook(self.infile)
        sheet = book.sheet_by_index(0)
        
        q_start=6
        q_end = sheet.ncols
        row_of_lectures = 1
        
        lecture_row = np.array(sheet.row_slice(rowx=row_of_lectures, start_colx=q_start, end_colx=q_end))
        lectures = []
        indices = []
        
        for i in range(len(lecture_row)):
            lectures.append(lecture_row[i].value)
        
        for i in range(len(lectures)):
            if("Lecture" in lectures[i] or "lecture" in lectures[i]):
                first_q = str(sheet.cell_value(row_of_lectures + 1, i + q_start))
                first_q_index = self.search_dict(first_q)
                indices.append(first_q_index)
                
        return indices
    
    def mark_wrong(self):
        """
        altars data so all absences are incorrect
        """
        self.answers[self.answers==-1] = 0
    
    def mark_absent(self):
        """
        altars data for case where a student was not absent
        """
        for i in range(len(self.answers)):
            for j in range(self.q_no):
                if(self.answers[i, j] == -1):
                    if(self.present(i, j)):
                        self.answers[i, j] = 0
                        
    def mark_dichot(self):
        """
        altars data so any question score between 0 and 1 is 0
        """
        for i in range(len(self.answers)):
            for j in range(self.q_no):
                if(self.answers[i, j] < 1 and self.answers[i, j] >= 0):
                    self.answers[i, j] = 0
                
             
    def present(self, s, q):
        """
        determines whether a student, s was present for the lecture of question, q
        if present, returns True
        """
        #find out which lecture it was in
        for i in range(len(self.lecture_indices)):
            if(q >= self.lecture_indices[i]):
                lecture_no = i
                
        #determine number of qs in lecture
        if(lecture_no < len(self.lecture_indices) - 1):
            q_no = self.lecture_indices[lecture_no + 1] - self.lecture_indices[lecture_no]
        else:
            q_no = self.q_no - self.lecture_indices[lecture_no]
        
        #check if other qs were answered
        lecture_tot = 0
        for i in range(self.lecture_indices[lecture_no], self.lecture_indices[lecture_no] + q_no):
            lecture_tot += self.answers[s, i]
            
        if(lecture_tot == -q_no):
            return False
        else:
            return True
        
        
    def missed_qs(self):
        """
        produces an array of missed questions, by student (like scores but for missed qs)
        """
        missed = np.zeros(self.student_no)
        
        for i in range(self.student_no):
            student = self.answers[i]
            negs = student[student<0]
            missed[i] = len(negs)
        
        return missed
    
        
    def missed_dist(self):
        """
        produces a histogram of students by questions missed
        """
        array = self.missed
        
        avg = np.average(array)
        std = np.std(array)
        
        lines = np.array([avg, avg - std, avg + std])
        colours = np.array(["k", "r", "r"])
        hist, hbins = np.histogram(array, bins=40)
        
        fig = plt.figure(figsize=(12,8))
        plt.hist(array, hbins)
        plt.title("Distribution of Questions Missed")
        plt.ylabel("No. Students")
        plt.xlabel("No. Questions Missed")
        plt.vlines(lines, ymin=0, ymax=np.max(hist), colors=colours)
        plt.figtext(120, 25, "$Avg=51.2, Std=44.3$")
        figsave = self.report_file[:-4] + "_MissedDistribution.png"
        fig.savefig(figsave)
           
        
    def missed_score_correlation(self):
        """
        produces a plot of number of questions missed against %score in answered qs
        basically, if you miss more qs are you more likely to get the answered ones wrong
        """
        avg = np.average(self.missed)
        std = np.std(self.missed)
        cutoff = avg + std

        #shortened_missed = self.missed[self.missed<cutoff] # removing vals in missed which were omitted from scores
        
        #should include all scores if possible, use to discuss possible shift in value from removal
        
        #producing a 2d array where element 0 is no qs missed, and element 1 is %score
        missed_w_vals = np.zeros([len(self.missed), 2])
        missed_w_vals[:, 0] = self.missed
        
        non_zero_indices = []
        for i in range(len(missed_w_vals)):
            if(missed_w_vals[i, 0] != self.q_no):
                non_zero_indices.append(i)
        non_zero_indices = np.array(non_zero_indices)
        
        missed_w_vals = missed_w_vals[non_zero_indices]
        scores = self.pre_cut_scores[non_zero_indices]
        denoms = self.q_no - self.missed[self.missed<self.q_no] #number of questions answered by students
        missed_w_vals[:, 1] = (scores / denoms) * 100
        missed_w_vals = missed_w_vals[missed_w_vals[:, 0].argsort()] # sorting array by number of qs missed
        
        #finding trendline
        polynomial_degrees = 1
        fit, cov = np.polyfit(missed_w_vals[:, 0], missed_w_vals[:, 1], polynomial_degrees, cov=True)
        print("[Gradient, Y-Intercept]: " + str(fit))
        print("[Grad Err, Intercept Err]: " + str(np.sqrt(np.diag(cov))))

        fit_plot = np.poly1d(fit)
        
        # plotting figure
        lines = [avg - std, avg, cutoff]
        colours = ["r", "k", "r"]
        fig = plt.figure(figsize=(12,8))
        plt.clf()
        plt.scatter(missed_w_vals[:, 0], missed_w_vals[:, 1])
        plt.plot(missed_w_vals[:, 0], fit_plot(missed_w_vals[:, 0]), color="b")
        plt.vlines(lines, ymin=0, ymax=100, color=colours)
        plt.title("Proportion of correct answers with number of questions missed")
        plt.ylabel("Proportion of answered questions correctly answered")
        plt.xlabel("Number of questions missed")
        figsave = self.report_file[:-4] + "_MissedScoreCorr.png"
        fig.savefig(figsave)
        
    def omit_low_response(self):
        """
        removes students with a response rate beyond a standard deviation below the mean
        """
        init = len(self.answers)
        avg = np.average(self.missed)
        std = np.std(self.missed)
        #self.answers[self.missed > (avg - std)] # potential fast way
        
        #finding indices of high responses
        indices = []
        for i in range(self.answers):
            if(self.missed[i] < (avg + std)):
                indices.append(i)
        indices = np.array(indices)

        #removing low responding students, by copying all others
        self.answers = self.answers[(indices)]
        
        return init - len(self.answers)
        
    def omit_no_response(self):
        """
        removes students who respond to no questions
        """
        init = len(self.answers)
        indices = []
        for i in range(len(self.missed)):
            if(self.missed[i] != self.q_no):
                indices.append(i)
        indices = np.array(indices)
        self.answers = self.answers[(indices)]
        
        return init - len(self.answers)
    
    def remove_part_point(self, start, stop):
        """
        changes the data to remove participation points between qs start and stop, very hacky
        """
        if(start == None or stop == None): #if the questions couldn't be found, change nothing
            return 0
        
        for i in range(len(self.answers)):
            for j in range(start, stop + 1):
                if(self.answers[i, j] == 1):
                    self.answers[i, j] = 0
                if(self.answers[i, j] == 2):
                    self.answers[i, j] = 1           
        
    
    def produce_scores(self):
        """
        produces an array of student scores
        """
        scores = np.zeros(len(self.answers))
        
        for i in range(len(self.answers)):
            row = self.answers[i]
            scores[i] = row[row>=0].sum()
            
        return scores
    
    def prop_answered(self, index):
        """
        returns the proportion of students that answered question at index
        """
        ans = self.answers[:, index]
        ans = ans[ans>=0]
        prop = len(ans) / len(self.answers)
        
        return prop
    
    def resp_rate(self):
        """
        returns the average proportion of responses to questions
        """
        props = np.zeros(self.q_no)
        for i in range(self.q_no):
            props[i] = self.prop_answered(i)
    
        return np.average(props)
    
    def response_graph(self):
        """
        plots a bar chart of responses to each question, called before no responses are marked wrong
        """
        resps = np.zeros(self.q_no)
        for i in range(self.q_no):
           q = self.answers[:, i]
           q = q[q >= 0]
           resps[i] = len(q) / len(self.answers)
        
        avg = np.average(resps)
        std = np.std(resps)
        
        horlines = np.array([avg, avg - std, avg + std])
        verlines = np.copy(self.lecture_indices)
        verlines = verlines.astype(float)
        verlines -= .5
        
        colours = ["k", "r", "r"]
        fig = plt.figure(figsize=(12, 8))
        figuresave = self.report_file[:-4] + "_Response Rate.png"
        ticks = np.arange(len(resps))
        plt.clf()
        plt.title("Response Rate by Question")
        plt.xlabel("Question No.")
        plt.ylabel("Response Rate")
        plt.bar(ticks, resps)
        plt.hlines(horlines, -5, self.q_no + 5, colours)
        plt.vlines(verlines, 0, np.max(resps), "k")
        fig.savefig(figuresave)
        
    def attendance_graph(self):
        """
        plots a chart of attendance, using number of people answering any question in a lecture
        """
        lecture_no = len(self.lecture_indices)
        attendance = np.zeros(lecture_no)
        for i in range(lecture_no):
            index = self.lecture_indices[i]
            q = self.answers[:, index]
            q = q[q >= 0]
            attendance[i] = len(q) / len(self.answers)
            
        avg = np.average(attendance)
        std = np.std(attendance)
        
        lines = np.array([avg, avg - std, avg + std])
        colours = ["k", "r", "r"]
        fig = plt.figure(figsize=(12, 8))
        figuresave = self.report_file[:-4] + "_Engagement.png"
        ticks = np.arange(lecture_no)
        plt.clf()
        plt.title("Engagement by Lecture")
        plt.xlabel("Lecture No.")
        plt.ylabel("Attendance")
        plt.bar(ticks, attendance)
        plt.hlines(lines, -1, lecture_no + 1, colours)
        fig.savefig(figuresave)
    
    #-------------------------------------------------------------------------------------
    
    def report(self):
        """
        performs each analysis and writes data into a txt file
        """
        self.attendance_graph()
        self.missed_dist()
        with open(self.report_file, "a") as f:
            f.write("Course Statistics\n\n")
            f.write("Standard Statistics:\n\n")
            
        #standard stats, fine with new format as based around score array which accounted for -1 vals
        avg = self.avg_score()
        avg_percent = (avg / self.q_no) * 100
        std = self.std_deviation()
        std_err = self.std_err()
        ran = self.score_range()
        self.distribution()
        self.missed_score_correlation()
        resp_rate = self.resp_rate()
        
        self.write("Questions", self.q_no)
        self.write("Students", self.student_no)
        self.write("Students Removed From Analysis", self.removed_no)
        self.write("Students Remaining", self.student_no - self.removed_no)
        self.write("Response Rate", resp_rate)
        with open(self.report_file, "a") as f:
            f.write("Average Score: %.2f (%.2f%%)\n" %(avg, avg_percent))
        self.write("Range", ran)
        self.write("Standard Deviation", std)
        self.write("Standard Error", std_err)
        
        
        with open(self.report_file, "a") as f:
            f.write("\nCTT Statistics:\n\n")
        #ctt stats, full course
        c_diff = self.course_difficulty()
        r_sh = self.split_halves()
        r_ca = self.coeff_alpha()
        r_kr = self.kr_20()
        sem = self.sem()
        fergs = self.fergusons()
        
        self.write("Difficulty", c_diff)
        self.write("Reliability (split halves)", r_sh)
        self.write("Reliability (coeff alpha)", r_ca)
        self.write("Reliability (Kunder-Richardson)", r_kr)
        self.write("Standard Error of Measurement", sem)
        self.write("Ferguson's Delta", fergs)
        
        with open(self.report_file, "a") as f:
            f.write("\nQuestions:\n\n")
            
        q_dif_arr = np.zeros(self.q_no)
        q_var_arr = np.zeros(self.q_no)
        q_disc_arr = np.zeros(self.q_no)
        q_pb_arr = np.zeros(self.q_no)
        
        for i in range(self.q_no):
            q_dif_arr[i] = self.q_difficulty(i)
            q_var_arr[i] = self.q_variance(i)
            q_disc_arr[i] = self.disc_index(i)
            q_pb_arr[i] = self.point_biserial(i)
            
            with open(self.report_file, "a") as f:
                f.write("    " + self.q_dict[i] + "\n")
            
            self.q_write("Difficulty", q_dif_arr[i])
            self.q_write("Variance", q_var_arr[i])
            self.q_write("Discrimination Index", q_disc_arr[i])
            self.q_write("Point Biserial Index", q_pb_arr[i])
            
            with open(self.report_file, "a") as f:
                f.write("\n\n")
            
            progress = (i / self.q_no) * 100
           
            sys.stdout.write("Progress: %.1f%%\r"
                              %(progress))
            sys.stdout.flush()
        
        self.stat_plot(q_dif_arr, "Difficulty")
        self.stat_plot(q_var_arr, "Variance")
        self.stat_plot(q_disc_arr, "Discrimination")
        self.stat_plot(q_pb_arr, "Point Biserial Index")
        
        self.hist_plot(q_dif_arr, "Difficulty", [0, 1])
        self.hist_plot(q_var_arr, "Variance", [0, 1])
        self.hist_plot(q_disc_arr, "Discrimination", [-1, 1])
        self.hist_plot(q_pb_arr, "Point Biserial Index", [-1, 1])
        
        self.outlier_report(q_dif_arr, "Difficulty")
        self.outlier_report(q_var_arr, "Variance")
        self.outlier_report(q_disc_arr, "Discrimination")
        self.outlier_report(q_pb_arr, "Point Biserial Index")
        
        sys.stdout.write("Complete.        ")
        
    def write(self, title, value):
        """
        writes a title and value for a stat to a txt file
        """
        with open(self.report_file, "a") as f:
            f.write(title + ": %.2f\n" %(value))
    
    def q_write(self, title, value):
        """
        writes a title and value for a stat to a txt file with an indent for ease of reading
        """
        with open(self.report_file, "a") as f:
            f.write("    " + title + ": %.2f\n" %(value))
            
    def stat_plot(self, stats, title):
        """
        plots a bar chart of the stat by question number
        """
        avg = np.average(stats)
        std = np.std(stats)
        
        lines = np.array([avg, avg - std, avg + std])
        colours = ["k", "r", "r"]
        fig = plt.figure(figsize=(12, 8))
        figuresave = self.report_file[:-4] + "_" + title + ".png"
        ticks = np.arange(len(stats))
        plt.clf()
        plt.title(title + " by Question")
        plt.xlabel("Question No.")
        plt.ylabel(title)
        plt.bar(ticks, stats)
        plt.hlines(lines, -5, self.q_no + 5, colours)
        fig.savefig(figuresave)
    
    def outlier_report(self, stats, title):
        """
        finds information about the questions relative to eachother and gives outliers
        """
        file_title = self.report_file[:-4] + "_" + title + "_stats.txt"
        
        f = open(file_title, "w") # clearing txt file
        f.close()
        
        avg = np.average(stats)
        std = np.std(stats)
        outlist = [] #list of outlier qs
        args = [] # list of original arguments of outlier qs
        lectures = []
        for i in range(len(stats)):
            if(stats[i] > avg + std or stats[i] < avg - std):
                outlist.append(stats[i])
                args.append(i)
                for j in range(len(self.lecture_indices)):
                    if(i >= self.lecture_indices[j]):
                        lecture_no = j + 1
                lectures.append(lecture_no)
        
        outliers = np.zeros((len(outlist), 4))
        outliers[:, 0] = np.array(outlist)
        outliers[:, 1] = np.array(args)
        outliers[:, 2] = outliers[:, 0] - avg
        outliers[:, 3] = np.array(lectures)
        
        outliers = outliers[outliers[:, 2].argsort()]
        
            
        with open(file_title, "a") as f:
            f.write("Statistics for " + title + "\n\n")
            f.write("Total Questions: " + str(self.q_no) + "\n")
            f.write("Number beyond standard deviation for " + title + ": " + str(len(outliers)) + "\n\n")
            f.write("Average: %.2f\n" %(avg))
            f.write("Standard Deviation: %.2f\n\n" %(std))
            f.write("These questions fell outside the standard deviation (most negative first):\n")
            
            for i in range(len(outliers)):
                q_title = self.q_dict[outliers[i, 1]]
                f.write("    " + q_title + "\n")
                f.write("    Lecture %.0f\n" %(outliers[i, 3]))
                f.write("    " + title + ": %.2f\n" %(outliers[i, 0]))
                f.write("    " + "Difference from Average: %.2f\n\n" %(outliers[i, 2]))
            
    #------------------------------------------------------------------------------------------
    #standard analysis
    
    def avg_score(self):
        """
        returns the avg score of students
        """
        return np.average(self.scores)
    
    def variance(self):
        """
        returns variance of course
        """
        return np.var(self.scores)
        
    def std_deviation(self):
        """
        returns the standard deviation of scores on the course
        """
        return np.std(self.scores)
    
    def std_err(self):
        """
        returns the standard error on the mean of the scores
        """
        err = self.std_deviation() / np.sqrt(len(self.scores))
        
        return err
    
    def score_range(self):
        """
        returns the range of scores on the course
        """
        return np.max(self.scores) - np.min(self.scores)
        
    def distribution(self):
        """
        displays a plot of score distribution for the course
        """
        hist, hbins = np.histogram(self.scores, bins=30)
        avg = self.avg_score()
        std = self.std_deviation()
        
        lines = np.array([avg, avg - std, avg + std])
        colours = np.array(["k", "r", "r"])
        plt.clf()
        fig = plt.figure(figsize=(12,8))
        plt.hist(self.scores, hbins)
        plt.vlines(lines, ymin=0, ymax=np.max(hist), colors=colours)
        plt.title("Score Distribution")
        plt.ylabel("Frequency")
        plt.xlabel("Score")
        figsave = self.report_file[:-4] + "_ScoreDistribution.png"
        fig.savefig(figsave)
        

    def q_variance(self, index):
        """
        returns the variance of a specific question
        """
        q_resp = self.answers[:, index]
        q_ans = q_resp[q_resp>=0]
        
        return np.var(q_ans)
    
    def frequency(self, score):
        """
        returns how many of a score appear in the data
        """
        
        count = len(self.scores[self.scores==score])
        
        return count
    
    #-------------------------------------------------------------------------------------------
        
    #ctt stats
        
    def q_difficulty(self, index):
        """
        returns the difficulty of an individual question (using ctt)
        """
        q_resp = self.answers[:, index]
        q_ans = q_resp[q_resp>=0]
        
        difficulty = len(q_ans[q_ans>0]) / len(q_ans)
        
        return difficulty
    
    def course_difficulty(self):
        """
        returns the average difficulty for the whole course
        """
        if(self.cd != None):
            return self.cd
        
        difficulties = np.zeros(self.q_no)
        
        for i in range(self.q_no):
            difficulties[i] = self.q_difficulty(i)
        
        self.cd = np.average(difficulties)
        return self.cd
    
    def split_halves(self):
        """
        returns the reliability coeff of the course, using the spearman brown prophecy
        """
        
        if(self.sh != None):
            return self.sh
        
        subtest_1 = self.answers[:, 0::2]
        subtest_2 = self.answers[:, 1::2]
        
        scores_1 = np.zeros(len(self.answers))
        scores_2 = np.zeros(len(self.answers))
        
        for i in range(len(subtest_1)):
            student_first = subtest_1[i]
            student_first = student_first[student_first >= 0] #removing no responses
            scores_1[i] = np.sum(student_first)
            
        for i in range(len(subtest_2)):
            student_second = subtest_2[i]
            student_second = student_second[student_second >= 0]
            scores_2[i] = np.sum(student_second)
        
        avg_1 = np.average(scores_1)
        avg_2 = np.average(scores_2)
        
        differences_1 = scores_1 - avg_1
        differences_2 = scores_2 - avg_2
        
        std_1 = np.sqrt(np.sum((differences_1 ** 2)))
        std_2 = np.sqrt(np.sum((differences_2 ** 2)))
        
        differences_product = differences_1 * differences_2
        
        summation = np.sum(differences_product)
        std_product = std_1 * std_2
        
        r_hh = summation / std_product
        
        r_tt = (2 * r_hh) / (1 + r_hh)
        
        self.sh = r_tt
        
        return self.sh
    
    def coeff_alpha(self):
        """
        returns the coefficient alpha for the course
        """
        if(self.ca != None):
            return self.ca
        
        summation = 0
        for i in range(self.q_no):
            summation += self.q_variance(i)
            
            
        pre_factor = self.q_no / (self.q_no - 1)
            
        alpha = pre_factor * (1 - (summation / self.variance()))
        
        self.ca = alpha
        return self.ca
            
    def kr_20(self):
        """
        returns the reliability using kunder richardson method DICHOT
        """
        if(self.kr != None):
            return self.kr
        
        summation = 0
        for i in range(self.q_no):
            q_diff = self.q_difficulty(i)
            increment = q_diff * (1 - q_diff)
            summation += increment

        r_tt = (self.q_no / (self.q_no - 1)) * (1 - (summation / self.variance()))
        
        self.kr = r_tt
        return self.kr
    
    def sem(self):
        """
        returns standard error of measurement
        """
        reliability = self.coeff_alpha()
        if(reliability >= 1):
            return 0
        else:
            return self.std_deviation() * np.sqrt(1 - reliability)
    
    def fergusons(self):
        """
        returns the ferguson delta for the course
        """
        if(self.fe != None):
            return self.fe
        
        score_no = len(self.scores)
        summation = 0
        for i in range(self.q_no):
            summation += self.frequency(i)**2
        
        delta = ((score_no ** 2) - summation) / (score_no ** 2 - (score_no ** 2 / (self.q_no + 1)))
        
        self.fe = delta
        return self.fe
    
    def disc_index(self, index):
        """
        returns the discrimination index of a question DICHOT
        """
        
        h_list = []
        l_list = []
        
        q_resps = self.answers[:, index]
        q_answered = q_resps[q_resps>=0]
        
        #having removed no response answers, we now have to remove the elements of self.scores which those relate to
        #otherwise when we go to find the scores of the people we're assigning high/low, we'll get the wrong score element
        scores_answered = self.scores[q_resps>=0]
        
        #identifying cutoff point at upper / lower 27%
        sorted_scores = self.scores.copy()
        sorted_scores = np.sort(sorted_scores)
        ts_index = int(len(sorted_scores) * .27)
        l_cutoff = sorted_scores[ts_index]
        u_cutoff = sorted_scores[-ts_index]
        for i in range(len(q_answered)):
            if(scores_answered[i] >= u_cutoff): # if score is in upper 27% add index to high score list
                h_list.append(i)
            if(scores_answered[i] <= l_cutoff): # if score is in lower 27% add index to low score list
                l_list.append(i)
        
        
        h_count = 0
        l_count = 0
            
        for i in range(len(h_list)):
            if(q_answered[h_list[i]] != 0):
                h_count += 1
                
        for i in range(len(l_list)):
            if(q_answered[l_list[i]] != 0):
                l_count += 1
    
        if(h_count == 0): # catching occassions when nobody who answered falls into the upper/lower 27%
            u = 0
        else:
            u = h_count / len(h_list)
        if(l_count == 0):
            l = 0
        else:
            l = l_count / len(l_list)
        
        d = (u - l) 
        
        return d
    
    def point_biserial(self, index):
        """
        returns the point biserial coefficient for a question
        """
        
        avg_total = self.avg_score()
        std_total = self.std_deviation()
        correct_scores = []
        for i in range(len(self.answers)):
            if(self.answers[i, index] == 1):
                correct_scores.append(self.scores[i])
        if(len(correct_scores) == 0):
            avg_correct_score = 0
            
        else:
            correct_scores = np.array(correct_scores)
            avg_correct_score = np.average(correct_scores)
        
        diff = avg_correct_score - avg_total
        
        n = len(self.answers)
        n1 = len(correct_scores)
        n0 = n - n1
        
        r = (diff / std_total) * np.sqrt(n1 * n0 / (n ** 2))
        
        return r 
        
    def hist_plot(self, array, title, minmax):
        """
        produces a histogram of the array passed
        """
        #vals, bins = np.histogram(array)
        fig = plt.figure(figsize=(12, 8))
        figuresave = self.report_file[:-4] + "_" + title + "_hist.png"
        
        if(minmax[0] == 0):
            bins = 10
        if(minmax[0] == -1):
            bins = 20
            
        plt.clf()
        plt.title(title + " Distribution")
        plt.xlabel(title)
        plt.ylabel("Number")
        plt.hist(array, bins, range=minmax)
        fig.savefig(figuresave)
    
    #---------------------------------------------------------------------------------------------------
    
    
    