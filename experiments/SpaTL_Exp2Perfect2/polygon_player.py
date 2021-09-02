import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from spatial.geometry import Polygon, DynamicObject, PolygonCollection, StaticObject
from spatial.logic import Spatial
import copy


def create_objects_in_time(data: dict)->list:
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
            p = np.array(p)
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
    spatial.assign_variable('red1',objects[0])
    spatial.assign_variable('red2',objects[1])
    spatial.assign_variable('blue1',objects[2])
    spatial.assign_variable('blue2',objects[3])
    spatial.assign_variable('green1',objects[4])
    spatial.assign_variable('green2',objects[5])
    # create goal regions
    red_goal_poly = Polygon(np.array([[45,450],[45,700],[250,700],[250,450]]))
    red_goal = StaticObject(PolygonCollection(set([red_goal_poly])))
    green_goal_poly = Polygon(np.array([[700, 120], [900, 120], [900, 420], [700, 420]]))
    green_goal = StaticObject(PolygonCollection(set([green_goal_poly])))
    spatial.assign_variable('redGoal',red_goal)
    spatial.assign_variable('greenGoal',green_goal)

    specification = '(F(red1 enclosedin redGoal)) and (F(red2 enclosedin redGoal))'
    specification += 'and (F(green1 enclosedin greenGoal)) and (F(green2 enclosedin greenGoal))'
    specification += 'and (G((red1 closeto blue1) -> (red1 below blue1)))'
    specification += 'and (G((red1 closeto blue2) -> (red1 above blue2)))'
    specification += 'and (G((green1 closeto blue1) -> (green1 below blue1)))'
    specification += 'and (G((green1 closeto blue2) -> (green1 above blue2)))'
    R_M = "(red1 moved red1[-1])"
    NOT_G_M = "(not(green1 moved green1[-1]) and (green2 moved green2[-1]))"
    specification += f'and (G ({R_M} ->t  (G[0,10] {NOT_G_M}))) '


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
            plt.plot(p[:,0],p[:,1],colorcode if i > 2 else 'k',alpha=0.3 if i > 2 else 1, zorder=1 if i > 2 else 1000)


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





