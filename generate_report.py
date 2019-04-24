import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import json
import re

def load_dataframe(filename):
    with open(filename,'r') as f:
        df = pd.DataFrame(json.load(f))
        df['index'] = range(len(df))
        return df

def drop_cols(df,dropcols):
    return df[list(filter(lambda x:not re.search('|'.join(dropcols),x) ,df.columns))]

def plot_pie_chart(df,dropcols,dropre=['min','max','avg'],index=0,ax_elem=None):
    if ax_elem is None:
        print('ERROR: pass an axis element')
        return
    df = df.drop(dropcols,axis=1)
    df = drop_cols(df,dropre)
    dfi = df.loc[str(index),:]
    ax_elem.axis('equal')
    labels = [ '-'.join(list(map(str,item))) for item in list(dfi.items())]
    ax_elem.pie(dfi, labels=labels,startangle=0)

def save_figs_as_pdf(fig_arr,filename):
    with PdfPages(filename) as pdf:
        for fig in fig_arr:
            pdf.savefig(fig)
            
def list_detections(output):
    dets_d = {}
    for ko in output.keys():
        d = output[ko]
        dets = []
        for k in d.keys():
            dc = d[k]
            dets.append(dc['output']['labels_detected'])
        dets_d[ko]=dets
    return dets_d

def count_elems(dct):
    count = {}
    sums = {}
    for k in dct.keys():
        count[k] = list(map(lambda l:len(l) ,dct[k]))
        sums[k] = sum(count[k])
    return count,sums

def read_dict(jsonfile):
    op = {}
    with open(jsonfile,'r') as f:
        op = json.load(f)
    return op

def round_cols(df,cols,digits=4):
    df.loc[:,cols] = round(df.loc[:,cols],digits)
    return df

def get_intervals_data(intervals_dicts):
    its = list(map(read_dict,intervals_dicts))
    return its

def get_info_what_key(dct,key):
    for i in dct.keys():
        d = dct[i]
        for k in d.keys():
            if d[k]['key']==key:
                return k

def detections_count(ints_dm):
    detcount={}
    for f in ints_dm.keys():
        detcount[f]=sum(ints_dm[f][1])
    return detcount

def get_int_info(dct):
    res = {}
    for i in dct.keys():
        d = dct[i]
        res[i]={}
        for k in d.keys():
            key = d[k]['key']
            res[i][key]=[d[k]['info'],d[k]['success']]
            #res[i][key]['what']=k
    return res

def info_det(dct):
    res = {}
    for i in dct.keys():
        res[i] = dct[i]['detect']
    return res

def merge_intervals_info(detector_intervals):
    rj = {}
    for k in dts[1].keys():
        d = dts[1][k]
        rj[k]={'post_detection_ops':d['post_detection_ops'][0],'success':d['post_detection_ops'][1],'detect':d['detect'][0]}
    return rj

def cpy_det_sucess_data(detector_intervals_merged,ints_dm):
    for k in ints_dm.keys():
        ints_dm[k][1] = detector_intervals_merged[k]['success']
    return ints_dm

def plot_data_for_file(data_ex,ints_ex,ints_dm,file_index = 0,cols_to_drop=['name','index','input_file','Total_time','video_duration','video_file_size']):
    file_index = str(file_index)
    print('Plotting data for performance of',data_ex.loc[file_index]['name'],'on',data_ex.loc[file_index]['input_file'])
    fig, axes = plt.subplots(2,2,figsize=(20,20))
    pie_ax,lax,oax1,oax2=axes.reshape(-1,1)
    plt.subplots_adjust(wspace=0.4)
    plt.close(fig)
    pie_ax = pie_ax[0];oax1 = oax1[0];oax2 = oax2[0];lax=lax[0];
    
    df = pd.DataFrame()
    det_count = ints_dm[file_index][1]
    time_taken_per_frame_ex = ints_ex[file_index][0]
    time_taken_per_frame_dm = ints_dm[file_index][0]
    
    #pie chart
    plot_pie_chart(data_ex,cols_to_drop,index=file_index,ax_elem=pie_ax)
    
    #contour X(det count) vs time taken(multiple Y) for dm_detect and ex_detect
    df = pd.DataFrame()
    df['number of detections'] = det_count
    df['time taken per frame - detector model'] = time_taken_per_frame_dm
    df['time taken per frame - overall'] = time_taken_per_frame_ex
    #sns.jointplot("number of detections", "time taken per frame - detector model", data=df, height=5, ratio=6, color="black", kind="kde", space=0,shade=True)
    sns.lineplot(x="number of detections",y= "time taken per frame - detector model", data=df,ax=oax1)
    sns.lineplot(x="number of detections",y= "time taken per frame - overall", data=df,ax=oax2)
    #running X (ith frame) , det count, time_dm,time_ex,
    dfdm = pd.DataFrame()
    dfdm['number of detections'] = det_count
    dfdm['time taken per frame'] = time_taken_per_frame_dm
    dfdm['type'] = 'detector model'
    dfdm['frame'] = list(range(len(dfdm)))
    dfex = pd.DataFrame()
    dfex['number of detections'] = det_count
    dfex['time taken per frame'] = time_taken_per_frame_ex
    dfex['type'] = 'overall'
    dfex['frame'] = list(range(len(dfex)))
    df = dfdm.append(dfex)
    sns.lineplot(x="frame",y="time taken per frame",style="type", data=df,ax=lax)
    return fig

def generate_report():
    sns.set()
    df_files = ['detection_model-time.json','extractor-time.json']
    output_data_dict_file = 'output_data.json' 
    intervals_dicts = ['ext-intervals_data.json','det-intervals_data.json']

    drop_columns = False
    drop_re=['min','max'] #drop all columns that have these words
    cols_to_round = ['video_duration','detect-avg']
    cols_to_drop = ['name','index','input_file','Total_time','video_duration','video_file_size']

    dfs = list(map(load_dataframe,df_files))
    if drop_columns:
        dfs = list(map(lambda d:drop_cols(d,drop_re),dfs))

    dfs = list(map(lambda d:round_cols(d,cols_to_round),dfs))
    data_dm,data_ex = dfs 
    #dfex - DataFrame of extractor (main program)
    #dfdm - DataFrame of Detector model
    output_dict = read_dict(output_data_dict_file)
    #detections_lst = list_detections(output_dict)

    #detections_count, detections_sum = count_elems(detections_lst)

    dts = list(map(get_int_info,get_intervals_data(intervals_dicts)))
    ints_ex,ints_dm = list(map(info_det,dts)) #intervals data
    detector_intervals_merged = merge_intervals_info(dts[1])
    ints_dm = cpy_det_sucess_data(detector_intervals_merged,ints_dm)

    num_detections = list(map(int,detections_count(ints_dm)))
    data_dm['num_detections']=num_detections
    data_ex['num_detections']=num_detections
    print(data_ex.columns,'\n',data_dm.columns)

    plot_data_for_file(data_ex,ints_ex,ints_dm,file_index = 0)
    print('Index ','\t Filename')
    for i in range(len(data_ex)):
        print(data_ex.loc[str(i),'index'],'\t',data_ex.loc[str(i),'input_file'])

    fig, axes = plt.subplots(2,2,figsize=(15,15))
    xmetric = ['video_duration','index'][0]
    dft = data_dm
    sns.barplot(data=dft,x=xmetric,y='detect-total',ax=axes[0,0],hue='name')
    sns.barplot(data=dft,x=xmetric,y='detect-min',ax=axes[0,1],hue='name')
    sns.barplot(data=dft,x=xmetric,y='detect-max',ax=axes[1,0],hue='name')
    sns.barplot(data=dft,x=xmetric,y='detect-avg',ax=axes[1,1],hue='name')

    plt.title('Detection time inside detector model vs video duration')
    #plt.legend(title='Video file name', loc='lower right', labels=data_ex.loc[:,'input_file'])

    sns.barplot(data=data_ex,x='index',y='num_detections',hue='name')
    fig1,ax1 = plt.subplots()
    dd = data_ex.drop( ['name','input_file','Total_time','video_duration','video_file_size'],axis=1)
    dd=drop_cols(dd,['avg','min','max'])
    print(dd.columns)
    dd = dd[['download_video','load_video_file','fetch_frame-total', 'save_annotations']]#'init&prepare_model','load_model'
    dd[['import_deps','init_model','load_model','detect-total','post_detection_ops-total']] = data_dm[['import_deps','init_model','load_model','detect-total','post_detection_ops-total']]


    dd = dd[['download_video','load_video_file','import_deps','init_model','load_model','fetch_frame-total','detect-total','post_detection_ops-total','save_annotations']]
    dd.plot(kind='barh', stacked=True,figsize=(20,2.5*len(dd)),title='Breakup of total time taken',ax=ax1)

    fig2,axs = plt.subplots(2,2,figsize=(20,12))
    x,y,hue='num_detections','detect-avg','name'
    sns.lineplot(data=data_dm,x=x,y=y,hue=hue,ax=axs[0,0]).set_title('Time taken by detector model vs number of detections')
    sns.lineplot(data=data_ex,x=x,y=y,hue=hue,ax=axs[0,1]).set_title('Time taken Overall vs number of detections')
    sns.barplot(data=data_dm,x='index',y='num_detections',hue='name',ax=axs[1,0]).set_title('Number of detections per file')
    sns.barplot(data=data_ex,y='detect-total',hue='name',x='video_duration',ax=axs[1,1]).set_title('Total time for detection vs video duration')

    save_figs_as_pdf([fig1,fig2],'Report.pdf')

if __name__=="__main__":
    generate_report()