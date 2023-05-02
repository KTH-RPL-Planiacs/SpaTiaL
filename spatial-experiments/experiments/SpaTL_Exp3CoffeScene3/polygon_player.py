import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from spatial.geometry import Polygon, DynamicObject, PolygonCollection, StaticObject
from spatial.logic import Spatial
import copy

def post_process_data(data: list) -> list:
    keys = [10,11,12,13,14,15,16]

    # initialize with default vertices
    default_verts = [[-100,-100],[-100,-99],[-99,-99],[-99,-100]]
    def_dict = {k:default_verts for k in keys}
    data_p = [copy.deepcopy(def_dict) for i in range(len(data))]

    # add missing data?
    for t,obj in enumerate(data):
        # check per key and replace default value
        ks = list(obj.keys())
        for k in ks:
            data_p[t][k] = obj[k]

        #for i,k in enumerate(keys):
        #    if not k in obj and not added[i]:
        #        added[i] = True
    return data_p




def create_objects_in_time(data: list)->list:

    # post-process data first
    data = post_process_data(data)

    # create empty object in time for all keys
    keys = list(data[0].keys())
    objects = dict()
    for k in keys:
        objects[k] = DynamicObject()

    # get polygons and fill objects in time
    for timestep, d in enumerate(data):
        for k in keys:
            p = copy.copy(d[k])
            p.append(p[0])
            p = np.array(p)*np.array([1,-1])
            poly = PolygonCollection(set([Polygon(p)]))
            objects[k].addObject(poly, timestep)

    return objects

if __name__ == '__main__':

    # load file
    data = pickle.load(open('polygons.pkl','rb'))

    # get objects in time
    objects = create_objects_in_time(data)
    print(objects)

    # example spec
    spatial = Spatial(quantitative=True)
    spatial.assign_variable('mug',objects[13])
    spatial.assign_variable('plate',objects[16])
    spatial.assign_variable('cookies',objects[14])
    spatial.assign_variable('orange',objects[12])
    spatial.assign_variable('coffee',objects[10])
    spatial.assign_variable('milk',objects[11])
    spatial.assign_variable('jug',objects[15])
    # create goal regions
    red_goal_poly = Polygon(np.array([[650, 120], [900, 120], [900, 420], [650, 420]]))
    red_goal = StaticObject(PolygonCollection(set([red_goal_poly])))
    green_goal_poly = Polygon(np.array([[600, 500], [800, 500], [800, 730], [600, 730]]))
    green_goal = StaticObject(PolygonCollection(set([green_goal_poly])))
    spatial.assign_variable('redGoal',red_goal)
    spatial.assign_variable('greenGoal',green_goal)

    specification  = '(F (cookies above plate) and (cookies ovlp plate) )'
    specification += 'and (F (orange above plate) and (orange ovlp plate) )'
    specification += 'and ( (not((mug leftof plate) and (mug closeto plate))) U ((orange ovlp plate) and (orange rightof cookies)))'


    tree = spatial.parse(specification)
    #spatial.svg_from_tree(tree,'specification.svg')

    import time
    t0 = time.time()
    y = list()
    for i in np.arange(1,len(data)):
        y.append(spatial.interpret(tree, 1, i))
    t1 = time.time()-t0
    print('Specification value is = {}'.format(y[-1]))
    print('Evaluation took {}ms'.format(t1*1000))
    print('Evaluation took {}ms per step'.format(t1*1000/len(data)))
    plt.figure(figsize=(6,4))
    plt.plot([0,len(data)/30],[0,0],'-k',linewidth=2)
    plt.plot(np.array(range(len(y)))/30,y,linewidth=2)
    plt.plot(len(y)/30,y[-1],'ob',linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Satisfaction of Specification')
    plt.show()

    # go through time
    for i,timestep in enumerate(data):
        keys = list(timestep.keys())
        for k in keys:
            p = timestep[k]
            p.append(p[0])
            p = np.array(p)
            colorcode = '-'
            if k in (0,1):
                colorcode += 'r'
            if k in (2,3):
                colorcode += 'b'
            if k in (4,5):
                colorcode += 'g'
            plt.plot(p[:,0],p[:,1],'b' if i > 2 else 'k',alpha=0.3 if i > 2 else 1, zorder=1 if i > 2 else 1000)

    plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()
    plt.autoscale()
    plt.axis('equal')
    plt.show()


    # create single plots
    path = os.getcwd()+'/specification/'
    os.mkdir(path)
    for t,val in enumerate(y):
        plt.figure(figsize=(6,4))
        plt.plot([0,len(data)/30],[0,0],'-k',linewidth=2)
        plt.plot(np.array(range(t+1))/30,y[:(t+1)],'-b',linewidth=2)
        plt.plot(t/30,y[t],'ob',linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Satisfaction of Specification')
        plt.savefig(path+str(t).zfill(6)+'.jpg',dpi=300)
        plt.clf()





