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

folder = "/home/matthieu/Documents/masterthesus/Coverage-problem/tests/scale_gradient"

x = np.random.rand(180*2)*100
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
,8.13774758e+01,2.20493498e+01,5.18060898e+01,9.34576689e+00
,83.70529367741281, 25.652513749186134, 99.10898328242865, 47.45396541308392,
75.27620675438985, 16.23653791435421, 46.781892017525536, 70.73497455590443,
25.90254671626322, 70.70354660936616, 30.390458940225727, 73.09192937870873,
23.618583744156584, 1.5914390346383533, 16.00066863102676, 36.152842481731504,
18.984125765878435, 95.40668539302953, 81.00262998841367, 91.9019690588732,
68.574638740358, 97.13697815705213, 87.82593603566544, 50.432175598389826,
99.86595858758433, 37.72825608282974, 43.796592445452035, 89.56607437075984,
17.632004508806066, 32.55948877114969, 68.17639598153707, 27.422701468273615,
0.25023876027402414, 67.69985144633578, 80.90905449690328, 61.54266261038019,
90.39217033357201, 83.85048502503534, 69.64726501387368, 56.21298196436333,
75.31793527805844, 7.767589594281576, 44.214090941367466, 51.79075051849513,
76.92356988140807, 15.421774301789725, 83.23024291574018, 75.92847372449695,
20.1413090831174, 7.173255484993534, 92.51110191375192, 10.762573927731045,
22.515023650877907, 72.92571664834294, 33.67759339818684, 86.59023428984237,
41.64292128766984, 55.35180437175144, 8.500357694063677, 40.03411930037159,
36.50189990490672, 70.90634810859669, 51.68235224100183, 53.10321575843096,
86.55192538355385, 22.758981786236, 82.71452825314712, 45.00425297513077,
72.79341668009587, 6.2080637678698425, 64.78069739956928, 52.182101316637905,
47.31766715748621, 33.76342771476899, 13.28320836319955, 11.106378065751432,
56.255129065644674, 64.5406399230684, 4.663065121565657, 63.99050117832184,
96.95181526750052, 39.26416230130132, 24.214522837326445, 59.22852934217941,
91.73411809502514, 28.63578592016435, 56.75478461492898, 4.283611014032463,
11.096285334908096, 75.3490062665442, 79.4859164611158, 43.00367225571894,
19.124510675439144, 16.02623485597747, 37.42640110408073, 87.9436882547594,
95.41839916013149, 9.249552812350725, 11.852661131798081, 27.947571490890866,
72.2764467762961, 47.53608629468954, 77.09338991858694, 95.9265716773588,
56.64067040625154, 13.704949698127466, 93.33409982254327, 5.869820038369733,
26.27791676992541, 28.95414557237901, 70.179448213679, 36.05021250570366,
65.30432362386877, 17.39353992640448, 12.114344933265532, 19.34330574105735,
85.38799100638275, 5.774195865879184, 45.85305438921857, 0.4418455497289697])

x = x[:120*2]

scales = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
max_steps = [0.2,0.4,0.6,0.8,1]
N = 200
if(int(process) == 0):
    data = []
    endposes = []
    for scale in scales:
        print(scale)
        F = [scale]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            x1 += max_step(scale*algorithm1(x1), max_steps[int(process)])

        data.append(F)
        endposes.append(x1)


    np.savetxt(folder + "/process_"+str(process)+".txt", data)
    np.savetxt(folder + "/process_"+str(process)+"_endpose.txt", endposes)
    
if(int(process) == 1):
    data = []
    endposes = []
    for scale in scales:
        print(scale)
        F = [scale]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            x1 += max_step(scale*algorithm1(x1), max_steps[int(process)])

        data.append(F)
        endposes.append(x1)


    np.savetxt(folder + "/process_"+str(process)+".txt", data)
    np.savetxt(folder + "/process_"+str(process)+"_endpose.txt", endposes)

if(int(process) == 2):
    data = []
    endposes = []
    for scale in scales:
        print(scale)
        F = [scale]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            x1 += max_step(scale*algorithm1(x1), max_steps[int(process)])

        data.append(F)
        endposes.append(x1)


    np.savetxt(folder + "/process_"+str(process)+".txt", data)
    np.savetxt(folder + "/process_"+str(process)+"_endpose.txt", endposes)
    
if(int(process) == 3):
    data = []
    endposes = []
    for scale in scales:
        print(scale)
        F = [scale]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            x1 += max_step(scale*algorithm1(x1), max_steps[int(process)])

        data.append(F)
        endposes.append(x1)

    np.savetxt(folder + "/process_"+str(process)+".txt", data)
    np.savetxt(folder + "/process_"+str(process)+"_endpose.txt", endposes)
    
if(int(process) == 4):
    data = []
    endposes = []
    for scale in scales:
        print(scale)
        F = [scale]
        x1 = x.copy()
        for i in range(N):
            F.append(1 - fun(x1)/10000)
            x1 += max_step(scale*algorithm1(x1), max_steps[int(process)])

        data.append(F)
        endposes.append(x1)


    np.savetxt(folder + "/process_"+str(process)+".txt", data)
    np.savetxt(folder + "/process_"+str(process)+"_endpose.txt", endposes)

    



if(int(process) == 5):
    if(False):
        plot(x)
        plt.title("Initial deployment")
        plt.savefig(folder + '/initial_deployment.pdf')
        plt.show()
        data = np.loadtxt(folder + "/process_"+str(0)+"_endpose.txt")
        x = data[0]
        plot(x)
        plt.title("Final deployment")
        plt.savefig(folder + '/final_deployment.pdf')
        plt.show()
        
    if(True):
        data_index = 0

        data = np.loadtxt(folder + "/process_0.txt")
        for j, i in enumerate(data):
            if(j%2==1):
                plt.plot(i[1:], label = r'$\alpha$ = '+ str(i[0]))
            
        # data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_minimax.txt")
        # for i in data[data_index:data_index + 1]:
        #     plt.plot(i[1:],'r--', label = 'minimax algorithm')
            
        # data = np.loadtxt("/home/matthieu/Documents/masterthesus/Coverage-problem/tests/max_step/max_step_centroids.txt")
        # for i in data[data_index:data_index + 1]:
        #     plt.plot(i[1:],'b--', label = 'centroids algorithm')
            
        plt.title(r"Effect of the gradient scale factor $\alpha$ on the coverage ratio evolution")
        plt.ylabel("Coverage ratio")
        plt.xlabel("Iteration")
        plt.ylim(0.95,1.01)
        plt.legend()
        plt.savefig(folder + "/scale_gradient_evolution.pdf")
        plt.show()

    if(False):
        colors = ['salmon','gold','greenyellow','turquoise','lightsteelblue','purple','orange','brown']
        cell_text = []
        rows = []
        for p in [0,1,2,3,4]:
            data = np.loadtxt(folder + "/process_"+str(p)+".txt")
            plt.plot(scales, np.mean(data.T[-10:], axis = 0),'o-', color = colors[p], label = "max step = " +str(max_steps[p]))
            cell_text.append(np.around(np.mean(data.T[-10:], axis = 0),3))
            rows.append("M = " + str(max_steps[p]))
            
            
        plt.title(r"Effect of the gradient scale factor $\alpha$ on the coverage ratio")
        plt.ylabel("Coverage ratio")
        plt.xlabel(r"gradient scale factor $\alpha$")
        # the_table = plt.table(cellText=cell_text,
        #             rowLabels=rows,
        #             rowColours= colors[:len(scales)],
        #             cellLoc = 'center', rowLoc = 'center',
        #             loc='bottom', bbox=[0, -0.5, 1, 0.3])
        # plt.xticks(scales)
        # plt.subplots_adjust(left=0.2, bottom=0.3)
        plt.legend()
        plt.savefig(folder + "/scale_gradient_coverage.pdf")
        plt.show()
    
    if(False):
        colors = ['salmon','gold','greenyellow','turquoise','lightsteelblue','purple','orange','brown']
        cell_text = []
        rows = []
        for p in [0,1,2,3,4]:
            data = np.loadtxt(folder + "/process_"+str(p)+"_endpose.txt")
            data -= x
            data = np.mean(np.linalg.norm(np.array([t.reshape((-1,2)) for t in data]), axis = 2), axis=1)
            print(data)
            plt.plot(scales, data, 'o-', color = colors[p], label = 'max step = ' + str(max_steps[p]))
            cell_text.append(np.around(data,2))
            rows.append("M = " + str(max_steps[p]))
        
        
        plt.title(r"Effect of the gradient scale factor $\alpha$ on the mean distance" + "\n between the start and end position of the UAVs")
        plt.ylabel("mean distance [m]")
        plt.xlabel(r"gradient scale factor $\alpha$")
        plt.ylim(4.8,6.8)
        # the_table = plt.table(cellText=cell_text,
        #             rowLabels=rows,
        #             rowColours= colors[:len(scales)],
        #             cellLoc = 'center', rowLoc = 'center',
        #             loc='bottom', bbox=[0, -0.5, 1, 0.3])
        # plt.xticks(scales)
        # plt.subplots_adjust(left=0.2, bottom=0.3)
        plt.legend()
        plt.savefig(folder + "/scale_gradient_distances.pdf")
        plt.show()
    
    if(False):
        colors = ['salmon','gold','greenyellow','turquoise','lightsteelblue','purple','orange','brown']
        cell_text = []
        rows = []
        for p in [0,1,2,3,4]:
            data = np.loadtxt(folder + "/process_"+str(p)+".txt")      
            data = data.T[1:]/data.T[-1]
            data = ([np.where(i > 0.99)[0][0] for i in data.T])
            cell_text.append(data)
            rows.append("M = " + str(max_steps[p]))
        
            plt.plot(scales, data, 'o-', color = colors[p], label = 'max step = ' + str(max_steps[p]))
            
        plt.title(r"Effect of the gradient scale factor $\alpha$ on the convergence time")
        plt.ylabel("Iterations")
        plt.xlabel(r"gradient scale factor $\alpha$")
        # the_table = plt.table(cellText=cell_text,
        #               rowLabels=rows,
        #               rowColours= colors[:len(scales)],
        #               cellLoc = 'center', rowLoc = 'center',
        #               loc='bottom', bbox=[0, -0.5, 1, 0.3])
        # plt.xticks(scales)
        # plt.subplots_adjust(left=0.2, bottom=0.3)
        plt.legend()
        plt.savefig(folder + "/scale_gradient_convergence.pdf")
        plt.show()

