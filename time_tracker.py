from __future__ import print_function
from time import time
import pandas as pd
import json
class time_tracker:
    def __init__(self,name='yolo',target_type='tracked_obj'):
        self.target_type = target_type
        self.filename = target_type+'-time.json'
        self.data = {}
        self.ctr = 1
        self.intervals = {}
        self.data_dict = {'name':name } #comfortable for pandas

    def time_diff(self,lst,col=1):
        diff = None
        #for i in range(len(lst)-1,-1,-1):
        #    tmp = lst[i][col]
        #    diff = diff-tmp if diff is not None else tmp
        diff = lst[-1][col]-lst[0][col]
        return diff

    def interval_start(self,what,kv): #to measure time of repeating events
        if not self.intervals.get(what):
            self.intervals[what]={'index':self.ctr,'info':[]}
            self.ctr += 1
        self.intervals[what]['tmp']=time()
        self.intervals[what]['key']=kv

    def interval_stop(self,what): #kv is index name
        if not self.intervals.get(what):
            return
        self.intervals[what]['info'].append(time()-self.intervals[what]['tmp'])
        self.intervals[what]['tmp'] = None

    def process_intervals(self):
        for what in self.intervals:
            e = self.intervals[what]['info']
            if not e:
                continue
            self.intervals[what]['stats']={'avg':sum(e)/len(e),'max':max(e),'min':min(e),'total':sum(e)}
    
    def note_time(self,what,info,kv=None):
        if not self.data.get(what):
            self.data[what]={'index':self.ctr,'info':[]}
            self.ctr += 1
        self.data[what]['info'].append([info,time()])
        if kv is not None:
            self.data[what]['key']=kv

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

    def set_input_filename(self,input_file):
        self.input_file = input_file
        self.data_dict['input_file'] = input_file

    def print_intervals_summary(self):
        if not self.intervals:
            return
        data,cols =[],[]
        ddict = {}
        print('\nRepeatedly occuring operations:',end='')
        
        for what in self.intervals:
            e = self.intervals[what]['stats']
            if not e:
                continue
            print('\n',what,'=>',end='')
            #list(map(lambda x:print(x[0],':',str(round(x[1],4))+'s',end=' | '),list(e.items())))
            for x in e.items():
                col = self.intervals[what]['key']+'-'+x[0]
                val = round(x[1],4)
                print(x[0],':',str(val)+'s',end=' | ')
                if not ddict.get(col):
                    ddict[col]=[]
                ddict[col].append(val)
        return ddict
        
    def print_summary(self):
        ddict = {}
        if not self.sorted_data:
            return
        for sd in self.sorted_data: #this is a list
            col = sd[1]['key']
            val = round(sd[1]['time'],4)
            if not ddict.get(col):
                    ddict[col]=[]
            print(sd[0],':',val,'s')
            ddict[col].append(val)
        return ddict

    def summary(self):
        self.calculate()
        self.sort()
        self.process_intervals()
        self.data_dict.update(self.print_summary())
        self.data_dict.update(self.print_intervals_summary())
        print('\n')
        self.save_to_file()
        return self.data_dict
    
    def save_to_file(self,filename=None):
        filename = self.filename if not filename else filename
        pd_file_dict = None
        try:
            f = open(filename,'r+')
            pd_file_dict = pd.DataFrame(json.load(f))
        except Exception as e:
                print('Cannot open file ',filename,'\nCreating new file')
                f = open(filename,'w+')
        finally:
            f.close()
        
        pd_data_dict = pd.DataFrame(self.data_dict)
        df = None
        if pd_file_dict is not None:
            df = pd_file_dict
            try:
                df.append(pd_data_dict)
            except Exception as e:
                print('Error while merging data frames',e)
        else:
            df = pd_data_dict
        with open(filename,'w+'):
            json.dump(df.to_dict(),f)
