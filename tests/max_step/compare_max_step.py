import sys
sys.path.insert(0, '/home/matthieu/Documents/masterthesus/Coverage-problem/')

from scripts.proper_classes4 import *
from other.voronoi_cells import *
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--process',metavar='process', help='foo help')
args = parser.parse_args()
process = args.process

print(process)

folder = "home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step"
x = np.random.rand(120*2)*100
x = array([6.11120577e+01,7.50231685e+01,9.69884207e+01,3.15027129e+01
,4.31837935e+01,6.30979289e+01,1.39778738e+01,7.37724303e+01
,9.20243058e+01,2.13191395e+00,1.06319720e+01,7.21469057e+01
,5.28595380e+01,4.66559358e+01,9.50508805e+00,7.05104165e+00
,6.67515051e+01,5.86701522e+01,3.21641650e+01,6.45292787e+00
,2.67798733e+01,2.51947636e+01,6.53436669e+00,8.69498713e+01
,9.79514352e+01,2.75544026e+01,6.33039066e+01,6.33939671e+01
,3.79429484e+01,9.75389954e+01,1.32793614e+00,7.96037806e+01
,8.82563393e+01,2.92204599e+01,4.48334870e+01,7.50963255e+01
,9.81112479e+01,9.16159009e+01,9.23294067e+01,6.27325965e+01
,2.26948435e+01,3.85386792e+01,1.30691079e+01,9.11420689e+01
,1.75714002e+01,3.22647190e+01,2.10734238e+01,9.29651838e+01
,5.37900441e+01,2.16978031e+01,3.26596557e+00,4.89759649e+01
,7.09983079e+01,7.54943706e+01,3.83677680e+01,2.21005843e+01
,9.08550483e+01,9.58902053e+01,8.00257763e+01,7.64880830e+01
,1.34230679e+01,8.09773399e+01,6.16831783e+01,1.44101574e+01
,6.66234747e+01,9.51760333e+01,1.55710467e+01,5.17256767e+01
,2.89793403e+01,2.66616815e+01,5.25230035e+01,5.09815604e+01
,6.65348374e+01,2.49390702e+01,8.15757185e+01,7.74029769e+01
,6.06927316e+01,3.22501956e+00,4.59451098e+00,7.42061439e+01
,9.39802263e+01,2.75378752e+01,4.34622666e+01,9.70829746e+01
,2.88335349e+01,2.51180071e+01,8.17567691e+00,3.97364026e+01
,7.82227864e+01,3.88472212e+01,7.34430666e+00,9.34892839e+01
,3.87166579e+01,2.06738303e+01,7.13991129e+01,5.64025728e+01
,6.77812333e+01,8.82100090e+01,9.41036951e+01,4.63721188e+01
,9.14007656e+01,8.69044979e+01,6.61381857e+01,3.85279657e+01
,2.39429704e+00,6.44175716e+01,1.91926282e+01,1.49066996e+01
,7.96294691e+00,2.71516652e+01,2.03800376e+01,9.84067681e+01
,4.63864782e+01,9.01707218e-02,4.12003810e+01,1.25837191e+01
,3.96081307e+01,8.67265089e+01,8.01092219e+01,3.02888091e+01
,3.29639545e+01,6.54255990e+01,5.73526206e+01,9.74038828e+01
,3.25312339e+01,9.75288724e+01,1.76554203e+01,3.18773123e+01
,3.10027165e+01,9.04702576e+01,6.74539856e+00,4.39091643e+01
,6.92710131e+01,1.40481036e+01,7.45986733e+01,1.67785933e+01
,7.90501036e+00,3.65573367e+01,4.33910018e+01,4.10445967e+01
,4.47615758e+01,2.60841488e+01,2.18281701e+01,5.35616416e+01
,4.98068677e+01,6.75813445e+01,2.41354093e+01,2.88534995e+01
,2.58399317e+01,1.13345631e+00,4.87880481e+01,3.32133945e+01
,5.70468595e+01,7.22309846e+01,8.50683622e+01,7.71893312e+01
,5.84991847e+00,8.79637921e+01,8.92767275e+01,1.73924409e+01
,7.82756648e+01,4.72964466e+01,7.06217476e-01,9.31275698e+01
,4.85729556e+01,5.15248182e+01,7.18933582e+01,6.65865331e+01
,1.74932258e+01,1.55361175e+01,9.97753648e+01,6.18254454e+01
,3.38915420e+01,2.91794735e+01,2.27252776e+01,1.69011206e+01
,2.53566141e+01,5.60379850e+01,4.28500472e+01,8.92283359e+01
,8.94708011e+01,4.03352822e+01,1.05026626e+00,5.41529739e+01
,2.34930572e+01,2.50389465e+01,5.96993221e+01,7.64177801e+01
,8.83260205e+01,3.50776861e+01,6.31299894e+01,3.20918124e+01
,7.43469244e+01,1.93821099e+01,1.75729682e+01,2.07504472e+01
,7.34338193e+01,8.74727676e+01,9.86924012e+01,7.61333036e+01
,9.85574106e+01,2.24061100e+01,5.33921340e+01,1.68073438e+01
,8.50184615e+01,3.51359392e+01,8.41761554e+01,3.17988707e+01
,5.80229982e+01,3.96630275e+00,4.52110451e+01,1.50064616e+01
,3.02236993e+01,3.48636893e+01,8.69215601e+01,9.55482179e+01
,2.04545878e+01,7.85169915e+01,4.32205994e+01,2.46443201e+01
,1.18711532e+01,3.40753713e+01,4.63500061e+01,6.46258865e+01
,5.17873229e+01,8.87171410e+01,8.34669562e+01,9.08956638e+01
,5.68015716e+01,7.46979891e+01,2.84836003e+01,9.25984861e+01
,1.96973257e+01,7.00777512e+01,5.15326102e+01,8.73181401e+01
,8.13774758e+01,2.20493498e+01,5.18060898e+01,9.34576689e+00])

steps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
N = 200
if(int(process) == 1):
    d = 1
    data = []
    endposes =[]
    for step in steps:
        print(step)
        F = [step]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            x1 += max_step(algorithm1(x1), step)

        data.append(F)
        endposes.append(x1)

    np.savetxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_gradient.txt", data)
    np.savetxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_gradient_endpose.txt", endposes)
    
if(int(process) == 2):
    d = 1
    data = []
    endposes =[]
    for step in steps:
        print(step)
        F = [step]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            try:
                x1 += max_step(minimax(x1.reshape((-1,2)),polygon = default_polygon).reshape(-1), step)
            except:
                x1 = x1+np.random.rand(120*2)/1000

        data.append(F)
        endposes.append(x1)

    np.savetxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_minimax.txt", data)
    np.savetxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_minimax_endpose.txt", endposes)

    
if(int(process) == 3):
    d = 1
    data = []
    endposes = []
    for step in steps:
        print(step)
        F = [step]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            x1 += max_step(centroids(x1.reshape((-1,2)),polygon = default_polygon).reshape(-1), step)

        data.append(F)
        endposes.append(x1)

    np.savetxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_centroids.txt", data)
    np.savetxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_centroids_endpose.txt", endposes)

if(int(process) == 4):
    if(True):
        data_index = 0

        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_gradient.txt")
        for i in data:
            if(int(i[0]*10)%2 == 0):
                plt.plot(i[1:], label = 'max step = '+ str(i[0]) + ' m')
            
        # data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_minimax.txt")
        # for i in data[data_index:data_index + 1]:
        #     plt.plot(i[1:],'r--', label = 'minimax algorithm')
            
        # data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_centroids.txt")
        # for i in data[data_index:data_index + 1]:
        #     plt.plot(i[1:],'b--', label = 'centroids algorithm')
            
        plt.title("Effect of the maximum step size on the coverage ratio evolution")
        plt.ylabel("Coverage ratio")
        plt.xlabel("Iteration")
        plt.ylim(0.95,1.01)
        plt.legend()
        plt.savefig("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_gradients.pdf")
        plt.show()

    if(True):
        colors = ['salmon','gold','greenyellow','turquoise','lightsteelblue','purple','orange','brown']
        cell_text = []
        rows = []
        data_index = 0

        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_gradient.txt")
        plt.plot(steps, data.T[-1], 'o-',color = colors[0], label = 'gradient descent algorithm')
        cell_text.append(np.round(data.T[-1],3))
        rows.append("Gradient")
        
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_minimax.txt")
        plt.plot(steps, data.T[-1],'o-', color = colors[1],label = 'minimax algorithm')
        cell_text.append(np.round(data.T[-1],3))
        rows.append("Minimax")
        
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_centroids.txt")
        plt.plot(steps, data.T[-1],'o-',color = colors[2], label = 'centroids algorithm')
        cell_text.append(np.round(data.T[-1],3))
        rows.append("Centroids")
            
        plt.title("Effect of the maximum step size on the final coverage ratio")
        plt.ylabel("Coverage ratio")
        plt.xlabel("maximum step size [m]")
        # the_table = plt.table(cellText=cell_text,
        #             rowLabels=rows,
        #             rowColours= colors[:3],
        #             cellLoc = 'center', rowLoc = 'center',
        #             loc='bottom', bbox=[0, -0.5, 1, 0.3])
        # plt.subplots_adjust(left=0.2, bottom=0.3)
        # plt.xticks(steps)
        plt.legend()
        plt.savefig("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_coverage.pdf")
        plt.show()
    
    if(True):
        colors = ['salmon','gold','greenyellow','turquoise','lightsteelblue','purple','orange','brown']
        cell_text = []
        rows = []
        
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_gradient_endpose.txt")

        data -= x
        data = np.mean(np.linalg.norm(np.array([t.reshape((-1,2)) for t in data]), axis = 2), axis=1)
        plt.plot(steps, data, 'o-', color = colors[0], label = 'gradient descent algorithm')
        cell_text.append(np.round(data,3))
        rows.append("Gradient")
        
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_minimax_endpose.txt")
        data -= x
        data = np.mean(np.linalg.norm(np.array([t.reshape((-1,2)) for t in data]), axis = 2), axis=1)
        plt.plot(steps, data,'o-', color = colors[1], label ='minimax algorithm')
        cell_text.append(np.round(data,3))
        rows.append("Minimax")
        
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_centroids_endpose.txt")
        data -= x
        data = np.mean(np.linalg.norm(np.array([t.reshape((-1,2)) for t in data]), axis = 2), axis=1)
        plt.plot(steps, data,'o-',  color = colors[2], label = 'centroids algorithm')
        cell_text.append(np.round(data,3))
        rows.append("Centroids")
        
        
        plt.title("Effect of the maximum step size on the mean distance \n between the start and end position of the UAVs")
        plt.ylabel("mean distance [m]")
        plt.xlabel("maximum step size [m]")
        plt.ylim(4.8,6.8)
        # the_table = plt.table(cellText=cell_text,
        #             rowLabels=rows,
        #             rowColours= colors[:3],
        #             cellLoc = 'center', rowLoc = 'center',
        #             loc='bottom', bbox=[0, -0.5, 1, 0.3])
        # plt.xticks(steps)
        # plt.subplots_adjust(left=0.2, bottom=0.3)
        plt.legend()
        plt.savefig("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_distances.pdf")
        plt.show()
    
    if(True):
        colors = ['salmon','gold','greenyellow','turquoise','lightsteelblue','purple','orange','brown']
        cell_text = []
        rows = []
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_gradient.txt")
        data = data.T[1:]/data.T[-1]
        data = ([np.where(i > 0.99)[0][0] for i in data.T])
        
        plt.plot(steps, data, 'o-', color = colors[0], label = 'Gradient descent algorithm')
        cell_text.append(np.round(data,3))
        rows.append("Gradient")
        
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_minimax.txt")
        data = data.T[1:]/data.T[-1]
        data = ([np.where(i > 0.99)[0][0] for i in data.T])
        
        plt.plot(steps, data, 'o-', color = colors[1], label = 'Minimax algorithm')
        cell_text.append(np.round(data,3))
        rows.append("Minimax")
        
        data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_centroids.txt")
        data = data.T[1:]/data.T[-1]
        data = ([np.where(i > 0.99)[0][0] for i in data.T])
        
        plt.plot(steps, data, 'o-', color = colors[2], label = 'Centroids algorithm')
        cell_text.append(np.round(data,3))
        rows.append("Centroids")
            
        plt.title("Effect of the maximum step size on the convergence time")
        plt.ylabel("Iterations")
        plt.xlabel("maximum step size [m]")
        # the_table = plt.table(cellText=cell_text,
        #               rowLabels=rows,
        #               rowColours= colors[:3],
        #               cellLoc = 'center', rowLoc = 'center',
        #               loc='bottom', bbox=[0, -0.5, 1, 0.3])
        # plt.xticks(steps)
        # plt.subplots_adjust(left=0.2, bottom=0.3)   
        plt.legend()     
        plt.savefig("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_convergence.pdf")
        plt.show()
