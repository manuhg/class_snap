from __future__ import print_function
from time import time

class time_tracker:
    def __init__(self):
        self.data = {}
        self.ctr = 1
        self.intervals = {}

    def time_diff(self,lst,col=1):
        diff = None
        #for i in range(len(lst)-1,-1,-1):
        #    tmp = lst[i][col]
        #    diff = diff-tmp if diff is not None else tmp
        diff = lst[-1][col]-lst[0][col]
        return diff

    def interval_start(self,what): #to measure time of repeating events
        if not self.data.get(what):
            self.intervals[what]={'index':self.ctr,'info':[]}
            self.ctr += 1
        self.intervals[what]['tmp']=time()

    def interval_stop(self,what):
        if not self.intervals.get(what):
            return
        self.intervals[what]['info'].append(time()-self.intervals[what]['tmp'])
        self.intervals[what]['tmp'] = None

    def process_intervals(self):
        for what in self.intervals:
            e = self.intervals[what]['info']
            self.intervals[what].update({'avg':sum(e)/len(e),'max':max(e),'min':min(e),'total':sum(e)})
    
    def note_time(self,what,info):
        if not self.data.get(what):
            self.data[what]={'index':self.ctr,'info':[]}
            self.ctr += 1
        self.data[what]['info'].append([info,time()])

    def calculate(self):
        for what in self.data.keys():
           self.data[what]['time'] = self.time_diff(self.data[what]['info'],1)

    def sort(self):
        sd = {}
        for item in self.data.items():
            sd[item[1]['index']] = item
        sk = list(sd.keys())
        sk.sort()
        sorted_data = []
        for k in sk:
            sorted_data.append(sd[k])
        self.sorted_data = sorted_data

    def print_intervals_summary(self):
        if not self.intervals:
            return
        print('Repeatedly occuring operations:')
        for what in self.intervals:
            e = self.intervals[what]['info']
            print('\n',what,'=>')
            list(map(lambda x:print(x[0],':',x[1],end=' | '),list(e.items())))

    def print_summary(self):
        if not self.sorted_data:
            return
        for sd in self.sorted_data: #this is a list
            print(sd[0],':',sd[1]['time'],'seconds')

    def summary(self):
        self.calculate()
        self.sort()
        self.process_intervals()
        self.print_summary()
        self.print_intervals_summary()
