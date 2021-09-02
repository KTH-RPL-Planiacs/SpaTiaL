import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
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

def plot_time_step(time_step: int, data: dict, trajectories: int, polygons= None):
    # go through time
    img = plt.imread('img/'+str(time_step).zfill(6)+'.jpg')
    figsize = (3.5, 2.5)
    plt.imshow(img)
    time_int = np.arange(time_step+trajectories,time_step+1) if trajectories<0 else np.arange(time_step,time_step+trajectories+1) if trajectories > 0 else np.arange(time_step,len(data))
    ax = plt.gca()
    if polygons is not None:
        for p in polygons:
            p.plot(ax)
    for t in time_int:
        timestep = data[t]
        keys = list(timestep.keys())
        for k in keys:
            p = timestep[k]
            p.append(p[0])
            p = np.array(p)
            colorcode = '-'
            if k in (0, 1):
                colorcode += 'r'
            if k in (2, 3):
                colorcode += 'b'
            if k in (4, 5):
                colorcode += 'g'
            plt.plot(p[:, 0], p[:, 1], colorcode if i > 2 else 'k', alpha=0.2 if i > 2 else 0.2,
                     zorder=1 if i > 2 else 1000)

    plt.autoscale()
    plt.axis('equal')
    #plt.gca().invert_yaxis()

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
    red_goal_poly = Polygon(np.array([[50, 120], [250, 120], [250, 350], [50, 350]]))
    red_goal = StaticObject(PolygonCollection(set([red_goal_poly])))
    green_goal_poly = Polygon(np.array([[600, 500], [800, 500], [800, 730], [600, 730]]))
    green_goal = StaticObject(PolygonCollection(set([green_goal_poly])))
    spatial.assign_variable('redGoal',red_goal)
    spatial.assign_variable('greenGoal',green_goal)

    specification = '(F(red1 enclosedin redGoal)) and (F(red2 enclosedin redGoal))'
    specification += 'and (F(green1 enclosedin greenGoal)) and (F(green2 enclosedin greenGoal))'
    specification += 'and (G((red1 closeto blue1) -> (red1 below blue1)))'
    specification += 'and (G((red2 closeto blue2) -> (red2 above blue2)))'
    specification += 'and (G((green1 closeto blue1) -> (green1 below blue1)))'
    specification += 'and (G((green2 closeto blue2) -> (green2 above blue2)))'

    # this is just the reach part of the spec
    specification_red = '(F(red1 enclosedin redGoal)) and (F(red2 enclosedin redGoal))'
    specification_green = '(F(green1 enclosedin greenGoal)) and (F(green2 enclosedin greenGoal))'
    specification_constr = '(G((red1 closeto blue1) -> (red1 below blue1)))'
    specification_constr += 'and (G((red2 closeto blue2) -> (red2 above blue2)))'
    specification_constr += 'and (G((green1 closeto blue1) -> (green1 below blue1)))'
    specification_constr += 'and (G((green2 closeto blue2) -> (green2 above blue2)))'


    tree = spatial.parse(specification)
    tree_red = spatial.parse(specification_red)
    tree_green = spatial.parse(specification_green)
    tree_cr = spatial.parse(specification_constr)
    #spatial.svg_from_tree(tree,'specification.svg')

    font = {'family': 'normal',
            # 'weight': 'bold',
            'size': 10}
    # matplotlib.rc('font', **font)
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['font.sans-serif'] = 'Latin Modern Math'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 10

    import time
    t0 = time.time()
    y = list()
    y_red = list()
    y_green = list()
    y_cr = list()
    for i in np.arange(1,len(data)):
        y.append(spatial.interpret(tree, 1, i))
        y_red.append(spatial.interpret(tree_red, 1, i))
        y_green.append(spatial.interpret(tree_green, 1, i))
        y_cr.append(spatial.interpret(tree_cr, 1, i))

    t1 = time.time()-t0
    print('Specification value is = {}'.format(y[-1]))
    print('Evaluation took {}ms'.format(t1*1000))
    print('Evaluation took {}ms per step'.format(t1*1000/len(data)))
    t_sat = np.argmin(np.abs(np.array(y)))
    print('Spec satisfied at time t={}'.format(t_sat if y[t_sat] >= 0 else t_sat + 1))
    plt.figure(figsize=(3.5,2.5))
    plt.plot(np.array(range(len(y))) / 30, y_green, '-g', linewidth=2)
    plt.plot(np.array(range(len(y))) / 30, y_red, '-r', linewidth=2)
    plt.plot(np.array(range(len(y))) / 30, y_cr, '--k', linewidth=2)
    plt.plot([0,len(data)/30],[0,0],'-k',linewidth=2)
    plt.plot(np.array(range(len(y)))/30,y,linewidth=2)
    plt.plot(len(y)/30,y[-1],'ob',linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Satisfaction of Specification')
    plt.savefig('exp2_moving_passage_satisfaction.svg')
    plt.show()

    # plot single steps
    plot_time_step(1,data,0,[red_goal.getObject(0),green_goal.getObject(0)])
    plt.savefig('exp2_moving_passage_t1.svg')
    plt.show()
    plot_time_step(7*30,data,120)
    plt.savefig('exp2_moving_passage_t210.svg')
    plt.show()
    plot_time_step(23*30,data,120)
    plt.savefig('exp2_moving_passage_t690.svg')
    plt.show()
    plot_time_step(46 * 30, data, 120)
    plt.savefig('exp2_moving_passage_t1380.svg')
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





