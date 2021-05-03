from kensu.client.models import *


def model_performance(predicted_dataset, month):
    from sklearn.metrics import precision_score
    import pandas as pd
    predictions = predicted_dataset['prediction']
    real_data = pd.read_csv('../data/real/'+month+'-data.csv')[['id','y']]
    precision = (precision_score(real_data['y'],predictions,labels=1))
    print('The precision of the model in {} was of {}'.format(month,round(precision,2)))
    data_revenues = pd.merge(predicted_dataset,real_data,on='id')[['id','prediction','y']]
    TP = 0 
    P = sum(data_revenues['prediction'])
    if P>300:
        factor = (300/P)
    else:
        factor=1
    for i in range(len(real_data['y'])): 
        if data_revenues['prediction'][i]==data_revenues['y'][i]==1:
            TP += 1
    revenues = 70 * TP * factor
    costs = 5 * P * factor
    profit = revenues - costs
    print('This results in a profit of ${}'.format(round(profit)))


def data_prep(data):
    import numpy as np
    import pandas as pd
    data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
    data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
    data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

    cat = [i for i in ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'] if i in data.columns]

    data_dummy = pd.get_dummies(data,columns=cat)

    features=[i for i in ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 
      'month_apr', 'month_aug', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_success"] if i in data_dummy.columns]

    data_final = data_dummy[features]
    return data_final


def create_observability_report(logfilename,pdfname,SL = None):
    import numpy as np
    import pandas as pd
    import os
    from pylab import rcParams
    import json
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image,Table, TableStyle,PageBreak,LongTable
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    import networkx as nx
    from reportlab.lib.styles import getSampleStyleSheet
    from kensu.utils.injection import Injection
    Injection().set_kensu_api('api')

    data = []
    with open(logfilename) as f:
        for line in f:
            data.append(json.loads(line))


    df= pd.DataFrame(columns=['Schema','Column','FromSchema','FromColumn'])

    LINEAGES = []
    DS = []
    SCHEMAS = []
    PROCESS = []
    RUN = []
    CODE_BASE =[]
    CODE_VERSION = []
    PROJECT = []
    STATS =[]
    for element in data:
        if element['entity'] == 'PROCESS_LINEAGE':
            LINEAGES.append(element['jsonPayload'])
        if element['entity'] == 'DATA_SOURCE':
            DS.append(element['jsonPayload'])
        if element['entity'] == 'SCHEMA':
            SCHEMAS.append(element['jsonPayload'])
        if element['entity']=='PROCESS' :
            PROCESS.append(element['jsonPayload'])
        if element['entity']=='PROCESS_RUN':
            RUN.append(element['jsonPayload'])
        if element['entity']=='CODE_BASE':
            CODE_BASE.append(element['jsonPayload'])
        if element['entity']=='CODE_VERSION':
            CODE_VERSION.append(element['jsonPayload'])
        if element['entity']=='PROJECT':
            PROJECT.append(element['jsonPayload'])
        if element['entity']=='DATA_STATS':
            STATS.append(element['jsonPayload'])


    ds_dict={}

    for e in DS:

        v=DataSource(name=e['name'],pk=DataSourcePK(location=e['pk']['location'], 
                                                    physical_location_ref=PhysicalLocationRef(by_guid=e['pk']['physicalLocationRef']['byGUID'])))

        ds_dict[v.to_guid()]=e   

    schema_dict = {}

    for e in SCHEMAS:
        s=Schema(name=e['name'],
               pk=SchemaPK(fields=[FieldDef(v['name'],v['fieldType'],v['nullable']) for v in e['pk']['fields']],
                               data_source_ref=DataSourceRef(by_guid=e['pk']['dataSourceRef']['byGUID'])))
        schema_dict[s.to_guid()]=e

    for el in LINEAGES:
        for m in el['pk']['dataFlow']:
            for to in m['columnDataDependencies']:
                for fr in m['columnDataDependencies'][to]:
                    df.loc[len(df)+1]={'Schema':m['toSchemaRef']['byGUID'],
                                       'Column':to,
                                       'FromSchema':m['fromSchemaRef']['byGUID'],
                                       'FromColumn':fr}  

    schema_ds_name = {}
    schema_ds = {}
    for el in schema_dict:
        key = (schema_dict[el]['pk']['dataSourceRef']['byGUID'])
        if key in ds_dict:
            schema_ds_name[el]=(ds_dict[key]['name'])
            schema_ds[el] = ds_dict[key]

    df['ToName'] = df['Schema'].map(schema_ds_name)
    df['FromName'] = df['FromSchema'].map(schema_ds_name)
    stats_dict={}
    for i in STATS:
        schem = i['pk']['schemaRef']['byGUID']
        stats = i['stats']
        columns = {}
        value = []
        for el in stats:
            s=stats[el]
            meta_name = el.split('.')[0]

            if meta_name not in columns:
                columns[meta_name]=[]
            name = el.split('.')[1]

            columns[meta_name].append({name:s})

        #stats_dict[schem] = columns
        aio=[]
        for i in columns:
            p=[]
            e=[]
            l=[]
            for vf in columns[i]:
                p.append(i) if i not in p else p.append('')
                e.append(list(vf.keys())[0])
                l.append(round(vf[list(vf.keys())[0]],2))
            aio.append([p,e,l])
        stats_dict[schem]=aio

    elems =[]
    styles = getSampleStyleSheet()
    styleH = styles['Heading1']
    styleH2 = styles['Heading2']
    styleH3 = styles['Heading3']
    styleN = styles['BodyText']
    elems.append(Paragraph("AI Observability report",styleH))
    elems.append(Paragraph("Context",styleH2))
    env = RUN[0]['environment']
    project = PROJECT[0]['pk']['name']
    date = RUN[0]['pk']['qualifiedName'].split('@')[1]

    elems.append(Paragraph("<br /> Project : "+ project+
                           "<br /> Application: "+PROCESS[0]['pk']['qualifiedName'] +
                           "<br /> Environment: "+ env +
                           "<br /> Run on: " + date+
                           "<br /> With version " + CODE_BASE[0]['pk']['location']+','+CODE_VERSION[0]['pk']['version']
                           ,styleN))


    filename = pdfname
    pdf = SimpleDocTemplate(
        filename,
        pagesize = letter
    )
    
    DG = nx.DiGraph()

    for row,el in df.iterrows():
        DG.add_node(el['FromName'])
        DG.add_node(el['ToName'])
        DG.add_edge(el['FromName'],el['ToName'])

    import matplotlib.pyplot as plt

    rcParams['figure.figsize'] = 14, 10
    pos = nx.spring_layout(DG, scale=20, k=3/np.sqrt(DG.order()))
    d = dict(DG.degree)
    v=nx.draw_networkx(DG, pos, node_color='lightblue', 
            with_labels=True,node_size=1800)
    plt.savefig('lin.png')
    elems.append(Paragraph("Lineage",styleH2))
    f = open('lin.png', 'rb')

    elems.append(Image(f,width=600,height=350))
    elems.append(PageBreak())
    elems.append(Paragraph("Data Sources",styleH2))

    for el in schema_dict:
        used={}
        fields =[v['name'] for v in schema_dict[el]['pk']['fields']]
        sub_df=df[df['FromSchema'] == el]
        for field in fields:
            if field in sub_df.FromColumn.to_list():
                used[field]='used'
            else:
                used[field]='unused'
        data = pd.DataFrame(columns=['Field','Used or not'])
        for e in used:
            data.loc[len(data)+1] = [e,used[e]] 
        ds = schema_ds[el]

        ds_id = pd.DataFrame.from_dict({'Name':[ds['name']],'Format':[ds['format']],'Location':[ds['pk']['location']]})

        schema_table = Table([data.columns.to_list()]+data.values.tolist())

        style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0),colors.lightblue),
        ('TEXTCOLOR', (0,0),(-1,0),colors.whitesmoke),
        ('BACKGROUND', (0,1), (-1,-1),colors.beige)

        ])

        schema_table.setStyle(style)

        rowNumb = len(data)
        for i in range(1,rowNumb):
            if i % 2 ==0:
                bc = colors.burlywood
            else:
                bc = colors.beige
            ts = TableStyle([
                ('BACKGROUND', (0,i), (-1,i),bc)
            ]
            )
            schema_table.setStyle(ts)
        values = []
        for i in ds_id.values.tolist():
            for ds_attr in i:
                values.append(Paragraph(ds_attr,styleN))
        table = Table([ds_id.columns.to_list()]+[values])
        style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0),colors.lightblue),
            ('TEXTCOLOR', (0,0),(-1,0),colors.whitesmoke),
            ('BACKGROUND', (0,1), (-1,-1),colors.beige)

        ])
        table.setStyle(style)

        elems.append(Paragraph('Data Source Name: '+ds['name'],styleH3))
        elems.append(Paragraph("<br/> Data Source ID:",styleN))
        elems.append(table)

        stats = ([Table(e,colWidths='*',style=style) for e in stats_dict[el]])
        elems.append(Paragraph("<br/> \
                               <br/>Data Source Schema:",styleN))
        elems.append(schema_table)

        elems.append(Paragraph("<br/> \
                                Data Source Stats:",styleN))
        for stat in stats:
            elems.append(stat)
            
    if SL:
        elems.append(PageBreak())
        sla = False
        slo = False
        slos_title=[]
        slos_table=[]
        slos_message =[]
        for element in SL:
            if 'SLA' in str(element.__class__) and sla==False:
                elems.append(Paragraph("Service Level Agreement:",styleH2))
                sla= True
            if 'SLA' in str(element.__class__):
                elems.append(Paragraph(element.get_sla(),styleH3))
            if 'SLO' in str(element.__class__) and slo==False:
                elems.append(Paragraph("Service Level Objectives:",styleH2))
                slo= True
            if 'SLO' in str(element.__class__):
                slos_table.append(element.get_slo())
                slos_message.append(element.datastrophes)
                slos_table.append(element.rules)
                if 'SLO' not in slos_title:
                    slos_title.append('SLO') 
                else:
                    slos_title.append('') 
        
        i=0
        sl_table = [slos_title]
        for el in range(len(slos_table)):
            if (el+1)%2 == 0:
                v=[Paragraph(str(i),styleN) for i in slos_table[i:el+1]]
               
                sl_table.append(v)
                i=el+1

        slos = Table(sl_table,colWidths='*')
        slos.setStyle(style)
        elems.append(slos)
        elems.append(Paragraph("Service Level Indicators, here are the associated alerts:",styleH2))
        #print(slos_message)
        v=[]
        for el in slos_message:
            for key in el:
                new_line = [Paragraph(str(el[key]['message']),styleN),
                            Paragraph(str(el[key]['expected']),styleN),
                            Paragraph(str(el[key]['actual']),styleN)]
                v = v + new_line
        if v == []:
            new_line = [Paragraph('No alert for this run',styleN),'','']
            v = v + new_line
        
        slis = Table([['Message','Expected','Actual'],v],colWidths='*')

        slis.setStyle(style)
        elems.append(slis)      

    pdf.build(elems)
    os.remove('lin.png')
    


def extract_data_sources(logfile):
    import json
    import pandas as pd
    data=[]
    with open(logfile) as f:
        for line in f:
            data.append(json.loads(line))


    df= pd.DataFrame(columns=['Schema','Column','FromSchema','FromColumn'])

    LINEAGES = []
    DS = []
    SCHEMAS = []
    PROCESS = []
    RUN = []
    CODE_BASE =[]
    CODE_VERSION = []
    PROJECT = []
    STATS =[]
    for element in data:
        if element['entity'] == 'PROCESS_LINEAGE':
            LINEAGES.append(element['jsonPayload'])
        if element['entity'] == 'DATA_SOURCE':
            DS.append(element['jsonPayload'])
        if element['entity'] == 'SCHEMA':
            SCHEMAS.append(element['jsonPayload'])
        if element['entity']=='PROCESS' :
            PROCESS.append(element['jsonPayload'])
        if element['entity']=='PROCESS_RUN':
            RUN.append(element['jsonPayload'])
        if element['entity']=='CODE_BASE':
            CODE_BASE.append(element['jsonPayload'])
        if element['entity']=='CODE_VERSION':
            CODE_VERSION.append(element['jsonPayload'])
        if element['entity']=='PROJECT':
            PROJECT.append(element['jsonPayload'])
        if element['entity']=='DATA_STATS':
            STATS.append(element['jsonPayload'])
    
    Ds_names = [i['name'] for i in DS]
    
    return Ds_names



def extract_data_stats(logfile, ds_name):
    from kensu.utils.kensu_provider import KensuProvider
    from kensu.utils.injection import Injection
    Injection().kensu_inject_entities('api')
    kensu = KensuProvider()
    data=[]
    import json
    import pandas as pd
    with open(logfile) as f:
        for line in f:
            data.append(json.loads(line))


    df= pd.DataFrame(columns=['Schema','Column','FromSchema','FromColumn'])

    LINEAGES = []
    DS = []
    SCHEMAS = []
    PROCESS = []
    RUN = []
    CODE_BASE =[]
    CODE_VERSION = []
    PROJECT = []
    STATS =[]
    for element in data:
        if element['entity'] == 'PROCESS_LINEAGE':
            LINEAGES.append(element['jsonPayload'])
        if element['entity'] == 'DATA_SOURCE':
            DS.append(element['jsonPayload'])
        if element['entity'] == 'SCHEMA':
            SCHEMAS.append(element['jsonPayload'])
        if element['entity']=='PROCESS' :
            PROCESS.append(element['jsonPayload'])
        if element['entity']=='PROCESS_RUN':
            RUN.append(element['jsonPayload'])
        if element['entity']=='CODE_BASE':
            CODE_BASE.append(element['jsonPayload'])
        if element['entity']=='CODE_VERSION':
            CODE_VERSION.append(element['jsonPayload'])
        if element['entity']=='PROJECT':
            PROJECT.append(element['jsonPayload'])
        if element['entity']=='DATA_STATS':
            STATS.append(element['jsonPayload'])
            
    

    ds_dict={}
    for e in DS:
        v=DataSource(name=e['name'],pk=DataSourcePK(location=e['pk']['location'], 
                                                    physical_location_ref=PhysicalLocationRef(by_guid=e['pk']['physicalLocationRef']['byGUID'])))

        ds_dict[v.to_guid()]=e   
    
    Ds_names = [i['name'] for i in DS]
    


    schema_dict = {}

    for e in SCHEMAS:
        s=Schema(name=e['name'],
               pk=SchemaPK(fields=[FieldDef(v['name'],v['fieldType'],v['nullable']) for v in e['pk']['fields']],
                               data_source_ref=DataSourceRef(by_guid=e['pk']['dataSourceRef']['byGUID'])))
        schema_dict[s.to_guid()]=e

    ds_schema_name = {}
    schema_ds = {}
    for el in schema_dict:
        key = (schema_dict[el]['pk']['dataSourceRef']['byGUID'])
        if key in ds_dict:
            ds_schema_name[ds_dict[key]['name']]=el
            schema_ds[el] = ds_dict[key]
    
            
    pk = ds_schema_name[ds_name]
    
    schema_stats = {}
    for e in STATS:
        stats = e['stats']
        schema = e['pk']['schemaRef']['byGUID']
        schema_stats[schema]=stats
   
    stat = schema_stats[pk]
    return stat


def extract_all_schemas(logfile):

    data=[]
    import json
    import pandas as pd
    with open(logfile) as f:
        for line in f:
            data.append(json.loads(line))

    SCHEMAS = []

    for element in data:
        if element['entity'] == 'SCHEMA':
            SCHEMAS.append(element['jsonPayload'])

    schemas = {i['name']:[e['name'] for e in i['pk']['fields']] for i in SCHEMAS}
    return schemas


class SLA(object):
    def __init__(self,definition):
        self.definition = definition
    def get_sla(self):
        return self.definition
        
class SLO(object):
    def __init__(self, definition, logfile):
        self.definition = definition
        self.logfile = logfile
        self.rules = {}
        self.datastrophes = {}
    def get_slo(self):
        return self.definition
        
    def monitor_schema_change(self, schema_name, reference):
        
        if 'schema' not in self.rules:
            self.rules['schema'] = [schema_name + ' must be ' + str(reference)]
        else:
            self.rules['schema'].append(schema_name + ' must be ' + str(reference))
        
        schema = extract_all_schemas(self.logfile)[schema_name]
        if schema != reference:
            self.datastrophes['Schema_change']= {'message':str(set(reference) - set(schema)) +' is missing',
                                                 'expected': reference,
                                                 'actual':schema}
    def monitor_data_stats(self, stat_name, value, min, max):
        if value < min or value > max:
            self.datastrophes['out-of-range']= {'message':str(stat_name) + ' is out of bounds', 
                                                'expected': [min,max],
                                                'actual': value}
        if 'stats' not in self.rules:
            self.rules['stats'] = [stat_name + ' must be in range ' + str([min,max])]
        else:
            self.rules['stats'].append(stat_name + ' must be in range ' + str([min,max]))

def kensu_client_init():
    import ipywidgets as widgets
    api_url = widgets.Text(
        value='',
        placeholder='Type the ingestion URL',
        description='API:',
        disabled=False
    )
    display(api_url)



    token = widgets.Password(
        value='',
        placeholder='Enter token',
        description='Token:',
        disabled=False
    )
    display(token)

    project = widgets.Text(
        value='',
        placeholder='Type a project name',
        description='Project:',
        disabled=False
    )
    display(project)


    def initialization_kensu(button):
        from kensu.utils.kensu_provider import KensuProvider
        kensu = KensuProvider().initKensu(api_url=api_url.value,
                                          auth_token=token.value,
                                          process_name='demo-monitoring-rules-online',
                                            user_name='Sammy', 
                                            code_location='https://gitlab.example.com', 
                                            init_context=True, 
                                            project_names=[project.value], 
                                            environment="Production",
                                            report_to_file=False,logical_naming='File')

        print('Kensu Initialization done')

    button = widgets.Button(
        description='Click to initialize',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me' )
    button.on_click(initialization_kensu)
    display(button)
    
    