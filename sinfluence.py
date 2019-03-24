__author__ = "Moses A. Boudourides & Sergios T. Lenis"
__copyright__ = "Copyright (C) 2015 Moses A. Boudourides & Sergios T. Lenis"
__license__ = "Public Domain"
__version__ = "1.0"

'''
This script computes and plots simulations of network infuence.
'''

import networkx as nx
import utils_attributes as utilat
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import itertools as it
import numpy as np
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
np.seterr(all='ignore')




def influence_sim(G,pos,sa,iterations,scale=1000): #nodes,p,
    # while  True:
    #     # G=nx.connected_watts_strogatz_graph(25, 2, 0.8, tries=100)
    #     G=nx.erdos_renyi_graph(nodes,p)
    #     if nx.is_connected(G):
    #         break
    # G.remove_nodes_from(nx.isolates(G))
    # # col=y
    # pos0=graphviz_layout(G) #nx.spring_layout(G)
    # pos=pos0 #nx.spring_layout(G)

    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    # scale=1000
    F,asoc=utilat.create_random_scalar_attributes(G,scale)
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]

    fig = plt.figure(figsize=(17,17))


    sstt="Initial distribution of the opinions over graph nodes" #\n (diffusion source at node %s with initial attribute =%f)" %(F.nodes()[0],starting_value_of_zero_node)
    plt.subplot(3,2,1).set_title(sstt)
    # plt.set_cmap('cool')
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')
    
    # plt.figure(2)
    sstt = "Time variation of opinions over graph nodes" #%F.nodes()[0]#,starting_value_of_zero_node)
    plt.subplot(3,2,5).set_title(sstt)

    # for i in F.nodes(data=True):
        # print i
    # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
    iterat=[]
    assort=[]
    y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    ckck=4
    kckc=2
    for ii in range(iterations):
        # sa=0.05
        checkin=True
        nd=list(F.nodes())[0]
        # print nd
        uu=0
        nei=list(F.neighbors(nd))
        # print nei
        for nnei in nei:
            uu+=F.node[nnei]['scalar_attribute']
        Xnei=uu/len(nei)
        X=F.node[nd]['scalar_attribute']
        uX=(sa*Xnei)+(1-sa)*X
        insau=int(uX*scale)
        F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

        for nd in list(F.nodes())[1:]:
            # sa=1-(1./nx.degree(F,nd))
            uu=0
            nei=list(nx.neighbors(F,nd))
            # print nei
            for nnei in nei:
                uu+=F.node[nnei]['scalar_attribute']
            X=F.node[nd]['scalar_attribute']
            Xnei=uu/len(nei)
            uX=(sa*Xnei)+(1-sa)*X
            insau=int(uX*scale)
            F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

            
        # for i in F.nodes(data=True):
        #   print i
        # Checking for attributes equality
        y1=y
        y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        if ii ==ckck and ii<10:
            # col=['%.5f' %yy for yy in y]
            col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
            sstt="Distribution of the opinions over graph nodes at %i iterations" %(ii+1)

            plt.subplot(3,2,kckc).set_title(sstt)
            nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
            plt.axis('equal')
            plt.axis('off')
            ckck+=5
            kckc+=1
            plt.subplot(3,2,5)
        # for yy in it.combinations(y,2):
        #   checkin=checkin and yy[0] -yy[1] <(1./scale)
        # for yy in range(len(y1)):
        #   # print y[yy]-y1[yy]<(1./scale)
        #   checkin=checkin and y[yy]-y1[yy]<(0.1/(scale))
        #   # print checkin
        # if checkin:
        #   break
        asss=nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        if np.isnan(asss):
            asss=1.
        # print 'Iteration %i ==> %f' %(ii,asss),y 
        # print type(asss), asss<-1,asss>1
        iterat.append(ii)

        assort.append(asss)

        # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        # y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        plt.plot(F.nodes(),y)
        # if asss==1. :
        #   break
    plt.plot(F.nodes(),y,linewidth=3.)
    sstt= "Time variation of the opinion assortativity coefficient"

    plt.subplot(3,2,6).set_title(sstt)

    # plt.figure(3)

    plt.plot(iterat,assort)
    plt.ylim(-1.1,1.1)
    # plt.figure(3)
    # plt.plot(F.nodes(),y)
    # col=['%.5f' %yy for yy in y]
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    sstt="Distribution of the opinions over graph nodes at %i iterations\n (consensual attribute = %s)" %(iterations,col[0])
    plt.subplot(3,2,4).set_title(sstt)

    # plt.figure(4)

    
    # print col
    # print [F.node[i]['scalar_attribute_numeric'] for i in F.nodes()]
    # pos=nx.spring_layout(G)
    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')

    plt.show()



#     plt.subplot(2,2,1).set_title("Original distribution of the opinions over graph nodes")
#     # plt.set_cmap('cool')
#     nx.draw_networkx(G,pos=pos, node_color=col,vmin=0.,vmax=1.)
#     # nx.draw_networkx(G,pos=pos, node_color=col,cmap=plt.cm.Cool)
#     plt.axis('equal')
#     plt.axis('off')

#     # plt.figure(2)
#     plt.subplot(2,2,2).set_title("Time variation of opinions over graph nodes")

#     # for i in F.nodes(data=True):
#         # print i
#     # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
#     iterat=[]
#     assort=[]
#     y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
#     for ii in range(iterations):
#         # sa=0.05
#         checkin=True

#         for nd in F.nodes():
#             # sa=1-(1./nx.degree(F,nd))
#             uu=0
#             nei=nx.neighbors(F,nd)
#             # print nei
#             for nnei in nei:
#                 uu+=F.node[nnei]['scalar_attribute']

#             sau=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
#             insau=int(sau*scale)
#             F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)

#         # for i in F.nodes(data=True):
#         #   print i
#         # Checking for attributes equality
#         y1=y
#         y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
#         # for yy in it.combinations(y,2):
#         #   checkin=checkin and yy[0] -yy[1] <(1./scale)
#         # for yy in range(len(y1)):
#         #   # print y[yy]-y1[yy]<(1./scale)
#         #   checkin=checkin and y[yy]-y1[yy]<(0.1/(scale))
#         #   # print checkin
#         # if checkin:
#         #   break
#         asss=nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
#         if np.isnan(asss):
#             asss=1.
#         # print 'Iteration %i ==> %f' %(ii,asss),y 
#         # print type(asss), asss<-1,asss>1
#         iterat.append(ii)

#         assort.append(asss)

#         # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
#         # y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
#         plt.plot(F.nodes(),y)
#         # if asss==1. :
#         #   break
#     plt.plot(F.nodes(),y,linewidth=3.)


#     plt.subplot(2,2,3).set_title("Time variation of the opinion assortativity coefficient")

#     # plt.figure(3)

#     plt.plot(iterat,assort)
#     plt.ylim(-1.1,1.1)
#     # plt.figure(3)
#     # plt.plot(F.nodes(),y)

#     plt.subplot(2,2,4).set_title("Final distribution of the opinions over graph nodes")

#     # plt.figure(4)

#     col=['%.5f' %yy for yy in y]
#     # print col
#     # print [F.node[i]['scalar_attribute_numeric'] for i in F.nodes()]
#     # pos=nx.spring_layout(G)
#     ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
#     nx.draw_networkx(G,pos=pos, node_color=col)
#     plt.axis('equal')
#     plt.axis('off')

#     plt.show()


def infdif_sim(G,pos,sa,sb,iterations,scale=1000):  #nodes,p,
    # while  True:
    #     # G=nx.connected_watts_strogatz_graph(25, 2, 0.8, tries=100)
    #     G=nx.erdos_renyi_graph(nodes,p)
    #     if nx.is_connected(G):
    #         break
    # G.remove_nodes_from(nx.isolates(G))
    # # col=y
    # pos0=graphviz_layout(G) #nx.spring_layout(G)
    # pos=pos0 #nx.spring_layout(G)

    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    # scale=1000
    F,asoc=utilat.create_random_scalar_attributes(G,scale)
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]

    fig = plt.figure(figsize=(17,17))

    starting_value_of_zero_node=F.node[0]['scalar_attribute']
    sstt="Initial distribution of the opinions over graph nodes\n (diffusion source at node %s with initial attribute = %f)" %(list(F.nodes())[0],starting_value_of_zero_node)
    plt.subplot(3,2,1).set_title(sstt)
    # plt.set_cmap('cool')
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')
    
    # plt.figure(2)
    sstt = "Time variation of opinions over graph nodes" #%F.nodes()[0]#,starting_value_of_zero_node)
    plt.subplot(3,2,5).set_title(sstt)

    # for i in F.nodes(data=True):
        # print i
    # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
    iterat=[]
    assort=[]
    y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    ckck=4
    kckc=2
    for ii in range(iterations):
        # sa=0.05
        checkin=True
        nd=list(F.nodes())[0]
        # print nd
        uu=0
        nei=list(nx.neighbors(F,nd))
        # print nei
        for nnei in nei:
            uu+=F.node[nnei]['scalar_attribute']
        Xnei=uu/len(nei)
        X=F.node[nd]['scalar_attribute']
        uX=(sa*Xnei)+(1-sa)*X
        insau=int(uX*scale)
        F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

        for nd in list(F.nodes())[1:]:
            # sa=1-(1./nx.degree(F,nd))
            uu=0
            nei=list(nx.neighbors(F,nd))
            # print nei
            for nnei in nei:
                uu+=F.node[nnei]['scalar_attribute']
            Xnei=uu/len(nei)
            X=F.node[nd]['scalar_attribute']
            uX=(sb*Xnei)+(1-sb)*X
            insau=int(uX*scale)
            F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

        # for i in F.nodes(data=True):
        #   print i
        # Checking for attributes equality
        y1=y
        y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        if ii ==ckck and ii<10:
            # col=['%.5f' %yy for yy in y]
            col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
            sstt="Distribution of the opinions over graph nodes at %i iterations" %(ii+1)

            plt.subplot(3,2,kckc).set_title(sstt)
            nx.draw_networkx(G,pos=pos, node_color=col,vmin=0.,vmax=1.)
            plt.axis('equal')
            plt.axis('off')
            ckck+=5
            kckc+=1
            plt.subplot(3,2,5)
        # for yy in it.combinations(y,2):
        #   checkin=checkin and yy[0] -yy[1] <(1./scale)
        # for yy in range(len(y1)):
        #   # print y[yy]-y1[yy]<(1./scale)
        #   checkin=checkin and y[yy]-y1[yy]<(0.1/(scale))
        #   # print checkin
        # if checkin:
        #   break
        asss=nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        if np.isnan(asss):
            asss=1.
        # print 'Iteration %i ==> %f' %(ii,asss),y 
        # print type(asss), asss<-1,asss>1
        iterat.append(ii)

        assort.append(asss)

        # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        # y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        plt.plot(F.nodes(),y)
        # if asss==1. :
        #   break
    plt.plot(F.nodes(),y,linewidth=3.)
    sstt= "Time variation of the opinion assortativity coefficient"

    plt.subplot(3,2,6).set_title(sstt)

    # plt.figure(3)

    plt.plot(iterat,assort)
    plt.ylim(-1.1,1.1)
    # plt.figure(3)
    # plt.plot(F.nodes(),y)
    # col=['%.5f' %yy for yy in y]
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    sstt="Distribution of the opinions over graph nodes at %i iterations\n (consensual attribute = %s)" %(iterations,col[0])
    plt.subplot(3,2,4).set_title(sstt)

    # plt.figure(4)

    
    # print col
    # print [F.node[i]['scalar_attribute_numeric'] for i in F.nodes()]
    # pos=nx.spring_layout(G)
    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')

    plt.show()



def polinf_sim(G,pos,iterations,scale=1000):  #nodes,p,
    # while  True:
    #     # G=nx.connected_watts_strogatz_graph(25, 2, 0.8, tries=100)
    #     G=nx.erdos_renyi_graph(nodes,p)
    #     if nx.is_connected(G):
    #         break
    # G.remove_nodes_from(nx.isolates(G))
    # # col=y
    # pos0=graphviz_layout(G) #nx.spring_layout(G)
    # pos=pos0 #nx.spring_layout(G)

    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    # scale=1000
    F,asoc=utilat.create_random_scalar_attributes(G,scale)
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]

    fig = plt.figure(figsize=(17,17))

    starting_value_of_zero_node=F.node[0]['scalar_attribute']
    sstt="Initial distribution of the opinions over graph nodes"#\n (diffusion source at node %s with initial attribute =%f)" %(F.nodes()[0],starting_value_of_zero_node)
    plt.subplot(3,2,1).set_title(sstt)
    # plt.set_cmap('cool')
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')
    
    # plt.figure(2)
    sstt = "Time variation of opinions over graph nodes" #%F.nodes()[0]#,starting_value_of_zero_node)
    plt.subplot(3,2,5).set_title(sstt)

    # for i in F.nodes(data=True):
        # print i
    # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
    iterat=[]
    assort=[]
    y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    ckck=4
    kckc=2
    for ii in range(iterations):
        # sa=0.05
        checkin=True
        # nd=F.nodes()[0]
        # # print nd
        # uu=0
        # nei=nx.neighbors(F,nd)
        # # print nei
        # for nnei in nei:
        #     uu+=F.node[nnei]['scalar_attribute']
        # if 
        # sau=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
        # insau=int(sau*scale)
        # F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)

        for nd in list(F.nodes()):
            # sa=1-(1./nx.degree(F,nd))
            uu=0
            nei=list(nx.neighbors(F,nd))
            # print nei
            for nnei in nei:
                uu+=F.node[nnei]['scalar_attribute']
            X=F.node[nd]['scalar_attribute']
            if X<0.5 and uu<0.5:
                F.add_node(nd,scalar_attribute=min(X,uu),scalar_attribute_numeric=int(min(X,uu)*scale))
            elif X>=0.5 and uu>=0.5:
                F.add_node(nd,scalar_attribute=max(X,uu),scalar_attribute_numeric=int(max(X,uu)*scale))
            
            # sau=(sb*uu/len(nei))+(1-sb)*F.node[nd]['scalar_attribute']
            # insau=int(sau*scale)
            # F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)

        # for i in F.nodes(data=True):
        #   print i
        # Checking for attributes equality
        y1=y
        y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        if ii ==ckck and ii<10:
            # col=['%.5f' %yy for yy in y]
            col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
            sstt="Distribution of the opinions over graph nodes at %i iterations" %(ii+1)

            plt.subplot(3,2,kckc).set_title(sstt)
            nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
            plt.axis('equal')
            plt.axis('off')
            ckck+=5
            kckc+=1
            plt.subplot(3,2,5)
        # for yy in it.combinations(y,2):
        #   checkin=checkin and yy[0] -yy[1] <(1./scale)
        # for yy in range(len(y1)):
        #   # print y[yy]-y1[yy]<(1./scale)
        #   checkin=checkin and y[yy]-y1[yy]<(0.1/(scale))
        #   # print checkin
        # if checkin:
        #   break
        asss=nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        if np.isnan(asss):
            asss=1.
        # print 'Iteration %i ==> %f' %(ii,asss),y 
        # print type(asss), asss<-1,asss>1
        iterat.append(ii)

        assort.append(asss)

        # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        # y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        plt.plot(F.nodes(),y)
        # if asss==1. :
        #   break
    plt.plot(F.nodes(),y,linewidth=3.)
    sstt= "Time variation of the opinion assortativity coefficient"

    plt.subplot(3,2,6).set_title(sstt)

    # plt.figure(3)

    plt.plot(iterat,assort)
    plt.ylim(-1.1,1.1)
    # plt.figure(3)
    # plt.plot(F.nodes(),y)
    # col=['%.5f' %yy for yy in y]
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    sstt="Distribution of the opinions over graph nodes at %i iterations"#\n (consensual attribute = %s)" %(iterations,col[0])
    plt.subplot(3,2,4).set_title(sstt)

    # plt.figure(4)

    
    # print col
    # print [F.node[i]['scalar_attribute_numeric'] for i in F.nodes()]
    # pos=nx.spring_layout(G)
    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')

    plt.show()




def sidif_sim(G,pos,b,iterations,scale=1000):  #nodes,p,
    # while  True:
    #     # G=nx.connected_watts_strogatz_graph(25, 2, 0.8, tries=100)
    #     G=nx.erdos_renyi_graph(nodes,p)
    #     if nx.is_connected(G):
    #         break
    # G.remove_nodes_from(nx.isolates(G))
    # # col=y
    # pos0=graphviz_layout(G) #nx.spring_layout(G)
    # pos=pos0 #nx.spring_layout(G)

    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    # scale=1000
    F,asoc=utilat.create_random_scalar_attributes(G,scale)
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]

    fig = plt.figure(figsize=(17,17))

    starting_value_of_zero_node=F.node[0]['scalar_attribute']
    sstt="Initial distribution of the opinions over graph nodes\n (diffusion source at node %s with initial attribute = %f)" %(list(F.nodes())[0],starting_value_of_zero_node)
    plt.subplot(3,2,1).set_title(sstt)
    # plt.set_cmap('cool')
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')
    
    # plt.figure(2)
    sstt = "Time variation of opinions over graph nodes" #%F.nodes()[0]#,starting_value_of_zero_node)
    plt.subplot(3,2,5).set_title(sstt)

    # for i in F.nodes(data=True):
        # print i
    # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
    iterat=[]
    assort=[]
    y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    ckck=4
    kckc=2
    for ii in range(iterations):
        # sa=0.05
        checkin=True
        # nd=F.nodes()[0]
        # # print nd
        # uu=0
        # nei=nx.neighbors(F,nd)
        # # print nei
        # for nnei in nei:
        #     uu+=F.node[nnei]['scalar_attribute']

        # sau=F.node[nd]['scalar_attribute'] + b*(1. - F.node[nd]['scalar_attribute'])*(uu/len(nei))
        # # sau=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
        # insau=int(sau*scale)
        # F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)

        for nd in list(F.nodes()):
            # sa=1-(1./nx.degree(F,nd))
            uu=0
            nei=list(nx.neighbors(F,nd))
            # print nei
            for nnei in nei:
                uu+=F.node[nnei]['scalar_attribute']
            X=F.node[nd]['scalar_attribute']
            Xnei=uu/len(nei)
            # uX= len(nei)*(1-Xnei)
            uX= b*(1. - X)*Xnei+X**2
            # uX= b*(1. - Xnei)*X

            # sau=(sb*uu/len(nei))+(1-sb)*F.node[nd]['scalar_attribute']
            insau=int(uX*scale)
            F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

        # for i in F.nodes(data=True):
        #   print i
        # Checking for attributes equality
        y1=y
        y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        if ii ==ckck and ii<10:
            # col=['%.5f' %yy for yy in y]
            col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
            sstt="Distribution of the opinions over graph nodes at %i iterations" %(ii+1)

            plt.subplot(3,2,kckc).set_title(sstt)
            nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
            plt.axis('equal')
            plt.axis('off')
            ckck+=5
            kckc+=1
            plt.subplot(3,2,5)
        # for yy in it.combinations(y,2):
        #   checkin=checkin and yy[0] -yy[1] <(1./scale)
        # for yy in range(len(y1)):
        #   # print y[yy]-y1[yy]<(1./scale)
        #   checkin=checkin and y[yy]-y1[yy]<(0.1/(scale))
        #   # print checkin
        # if checkin:
        #   break
        asss=nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        if np.isnan(asss):
            asss=1.
        # print 'Iteration %i ==> %f' %(ii,asss),y 
        # print type(asss), asss<-1,asss>1
        iterat.append(ii)

        assort.append(asss)

        # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        # y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        plt.plot(F.nodes(),y)
        # if asss==1. :
        #   break
    plt.plot(F.nodes(),y,linewidth=3.)
    sstt= "Time variation of the opinion assortativity coefficient"

    plt.subplot(3,2,6).set_title(sstt)

    # plt.figure(3)

    plt.plot(iterat,assort)
    plt.ylim(-1.1,1.1)
    # plt.figure(3)
    # plt.plot(F.nodes(),y)
    # col=['%.5f' %yy for yy in y]
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    sstt="Distribution of the opinions over graph nodes at %i iterations\n (consensual attribute = %s)" %(iterations,col[0])
    plt.subplot(3,2,4).set_title(sstt)

    # plt.figure(4)

    
    # print col
    print [F.node[i]['scalar_attribute_numeric'] for i in F.nodes()]
    # pos=nx.spring_layout(G)
    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')

    plt.show()



def polinfluence_sim(G,pos,sa,iterations,scale=1000):  #nodes,p,
    # while  True:
    #     # G=nx.connected_watts_strogatz_graph(25, 2, 0.8, tries=100)
    #     G=nx.erdos_renyi_graph(nodes,p)
    #     if nx.is_connected(G):
    #         break
    # G.remove_nodes_from(nx.isolates(G))
    # # col=y
    # pos0=graphviz_layout(G) #nx.spring_layout(G)
    # pos=pos0 #nx.spring_layout(G)

    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    # scale=1000
    F,asoc=utilat.create_random_scalar_attributes(G,scale)
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]

    fig = plt.figure(figsize=(17,17))


    sstt="Initial distribution of the opinions over graph nodes" #\n (diffusion source at node %s with initial attribute =%f)" %(F.nodes()[0],starting_value_of_zero_node)
    plt.subplot(3,2,1).set_title(sstt)
    # plt.set_cmap('cool')
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')
    
    # plt.figure(2)
    sstt = "Time variation of opinions over graph nodes" #%F.nodes()[0]#,starting_value_of_zero_node)
    plt.subplot(3,2,5).set_title(sstt)
    plt.ylim(-0.01,1.01)
    # for i in F.nodes(data=True):
        # print i
    # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
    iterat=[]

    assort=[]
    y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    ckck=4
    kckc=2
    for ii in range(iterations):
        # sa=0.05
        checkin=True
        # nd=F.nodes()[0]
        # # # print nd
        # uu=0

        # nei=nx.neighbors(F,nd)
        # # # print nei
        # for nnei in nei:
        #     uu+=F.node[nnei]['scalar_attribute']
        
        # if F.node[nd]['scalar_attribute'] < uu/len(nei):
        #     sau=sa*max(2.*F.node[nd]['scalar_attribute'] - uu/len(nei),0.)+(1-sa)*F.node[nd]['scalar_attribute']
        # if F.node[nd]['scalar_attribute'] > uu/len(nei):
        #     sau=sa*min(2.*F.node[nd]['scalar_attribute'] - uu/len(nei),1.)+(1-sa)*F.node[nd]['scalar_attribute']

        # # sau=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
        # insau=int(sau*scale)
        # F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)

        # nd=F.nodes()[-1]
        # # # print nd
        # uu=0

        # nei=nx.neighbors(F,nd)
        # # # print nei
        # for nnei in nei:
        #     uu+=F.node[nnei]['scalar_attribute']
        
        # if F.node[nd]['scalar_attribute'] < uu/len(nei):
        #     sau=sa*max(2.*F.node[nd]['scalar_attribute'] - uu/len(nei),0.)+(1-sa)*F.node[nd]['scalar_attribute']
        # if F.node[nd]['scalar_attribute'] > uu/len(nei):
        #     sau=sa*min(2.*F.node[nd]['scalar_attribute'] - uu/len(nei),1.)+(1-sa)*F.node[nd]['scalar_attribute']

        # # sau=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
        # insau=int(sau*scale)
        # F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)


        # if F.node[nd]['scalar_attribute'] < (uu/len(nei)):

        #     sau=(sa*max(2.*F.node[nd]['scalar_attribute'] - (uu/len(nei)),0.)+(1-sa)*F.node[nd]['scalar_attribute']
        # if F.node[nd]['scalar_attribute'] > (uu/len(nei)):

        # # else:
        #     sau=(sa*min(2.*F.node[nd]['scalar_attribute'] - (uu/len(nei)),1.)+(1-sa)*F.node[nd]['scalar_attribute']

        # # sau=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
        
        # insau=int(sau*scale)
        # F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)
        # for nd in F.nodes()[1:-1]:
        #     # sa=1-(1./nx.degree(F,nd))
        #     uu=0
        #     nei=nx.neighbors(F,nd)
        #     # print nei
        #     for nnei in nei:
        #         uu+=F.node[nnei]['scalar_attribute']

        #     sau=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
        #     insau=int(sau*scale)
        #     F.add_node(nd,scalar_attribute=sau,scalar_attribute_numeric=insau)


        for nd in list(F.nodes()):
            # sa=1-(1./nx.degree(F,nd))
            uu=0
            nei=list(nx.neighbors(F,nd))
            # print nei
            for nnei in nei:
                uu+=F.node[nnei]['scalar_attribute']
            X=F.node[nd]['scalar_attribute']
            Xnei=uu/len(nei)

            if X < Xnei:
                uX=sa*max(2.*X - Xnei,0.)+(1-sa)*X
            if X > Xnei:
                uX=sa*min(2.*X - Xnei,1.)+(1-sa)*X

            # uX=(sa*uu/len(nei))+(1-sa)*F.node[nd]['scalar_attribute']
            insau=int(uX*scale)
            F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

            
        # for i in F.nodes(data=True):
        #   print i
        # Checking for attributes equality
        y1=y
        y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        if ii ==ckck and ii<10:
            # col=['%.5f' %yy for yy in y]
            col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
            sstt="Distribution of the opinions over graph nodes at %i iterations" %(ii+1)

            plt.subplot(3,2,kckc).set_title(sstt)
            nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
            plt.axis('equal')
            plt.axis('off')
            ckck+=5
            kckc+=1
            plt.subplot(3,2,5)
        # for yy in it.combinations(y,2):
        #   checkin=checkin and yy[0] -yy[1] <(1./scale)
        # for yy in range(len(y1)):
        #   # print y[yy]-y1[yy]<(1./scale)
        #   checkin=checkin and y[yy]-y1[yy]<(0.1/(scale))
        #   # print checkin
        # if checkin:
        #   break
        asss=nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        # if np.isnan(asss):
        #     asss=1.
        # print 'Iteration %i ==> %f' %(ii,asss),y 
        # print type(asss), asss<-1,asss>1
        iterat.append(ii)

        assort.append(asss)

        # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        # y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        plt.plot(F.nodes(),y)
        # if asss==1. :
        #   break
    plt.plot(F.nodes(),y,linewidth=3.)
    sstt= "Time variation of the opinion assortativity coefficient"

    plt.subplot(3,2,6).set_title(sstt)

    # plt.figure(3)

    plt.plot(iterat,assort)
    plt.ylim(-1.1,1.1)
    # plt.figure(3)
    # plt.plot(F.nodes(),y)
    # col=['%.5f' %yy for yy in y]
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    sstt="Distribution of the opinions over graph nodes at %i iterations\n (polarized attributes = (%.2f, %.2f))" %(iterations,min(col),max(col))#col[0])
    plt.subplot(3,2,4).set_title(sstt)

    # plt.figure(4)

    
    # print col
    # print [F.node[i]['scalar_attribute_numeric'] for i in F.nodes()]
    # pos=nx.spring_layout(G)
    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')

    plt.show()





def cl_influence_sim(G,pos,sa1,sa2,iterations,scale=1000):  # nodes,p, b1,b2,
    # while  True:
    #     # G=nx.connected_watts_strogatz_graph(25, 2, 0.8, tries=100)
    #     G=nx.erdos_renyi_graph(nodes,p)
    #     if nx.is_connected(G):
    #         break
    # G.remove_nodes_from(nx.isolates(G))
    # # col=y
    # pos0=graphviz_layout(G) #nx.spring_layout(G)
    # pos=pos0 #nx.spring_layout(G)

    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    # scale=1000
    F,asoc=utilat.create_random_scalar_attributes(G,scale)
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]

    fig = plt.figure(figsize=(17,17))


    sstt="Initial distribution of the opinions over graph nodes" #\n (diffusion source at node %s with initial attribute =%f)" %(F.nodes()[0],starting_value_of_zero_node)
    plt.subplot(3,2,1).set_title(sstt)
    # plt.set_cmap('cool')
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')
    
    # plt.figure(2)
    sstt = "Time variation of opinions over graph nodes" #%F.nodes()[0]#,starting_value_of_zero_node)
    plt.subplot(3,2,5).set_title(sstt)

    # for i in F.nodes(data=True):
        # print i
    # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
    iterat=[]
    assort=[]
    y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    ckck=4
    kckc=2
    for ii in range(iterations):
        # sa=0.05
        checkin=True
        # nd=F.nodes()[0]
        # # print nd
        # uu=0
        # nei=nx.neighbors(F,nd)
        # # print nei
        # for nnei in nei:
        #     uu+=F.node[nnei]['scalar_attribute']
        # Xnei=uu/len(nei)
        # X=F.node[nd]['scalar_attribute']

        # if X <= 1/3. and Xnei <= 1/2.:  #1/3.
        #     uX=(sa1*Xnei)+(1-sa1)*X
        # elif 1/3. < X <= 2/3.:  #and 1/3. < Xnei <= 2/3.: 
        #     uX=(sa1*Xnei)+(1-sa1)*X
        # elif X > 2/3. and Xnei > 1/2.:  #2/3.
        #     uX=(sa1*Xnei)+(1-sa1)*X

        # elif X <= 1/3. and 1/3. < Xnei <= 2/3.:
        #     uX=(sa2*Xnei)+(1-sa2)*X
        # elif 1/3. < X <= 2/3. and Xnei <= 1/3.:
        #     uX=(sa2*Xnei)+(1-sa2)*X
        # elif 1/3. < X <= 2/3. and 2/3. < Xnei:
        #     uX=(sa2*Xnei)+(1-sa2)*X
        # elif 1/3. < Xnei <= 2/3. and 2/3. < X:
        #     uX=(sa2*Xnei)+(1-sa2)*X

        # elif X <= 1/3. and 1/3. < Xnei <= 2/3.:
        #     uX=(sa2*Xnei)+(1-sa2)*X
        # elif 1/3. < X <= 2/3. and Xnei <= 1/3.:
        #     uX=(sa2*Xnei)+(1-sa2)*X
        # elif 1/3. < X <= 2/3. and 2/3. < Xnei:
        #     uX=(sa2*Xnei)+(1-sa2)*X
        # elif 1/3. < Xnei <= 2/3. and 2/3. < X:
        #     uX=(sa2*Xnei)+(1-sa2)*X

        # else:
        #     uX = X

        # if b1 <= abs(X - Xnei) <= b2:
        #     uX=(sa*Xnei)+(1-sa)*X
        # # else:
        # #     uX = X
        # # elif X < Xnei:
        # #     uX=0
        # # elif X > Xnei:
        # #     uX=1
        # elif X < Xnei:
        #     uX=sa*max(2.*X - Xnei,0.)+(1-sa)*X
        # elif X > Xnei:
        #     uX=sa*min(2.*X - Xnei,1.)+(1-sa)*X
        # insau=int(uX*scale)
        # F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

        for nd in list(F.nodes()):
            # sa=1-(1./nx.degree(F,nd))
            uu=0
            nei=list(nx.neighbors(F,nd))
            # print nei
            for nnei in nei:
                uu+=F.node[nnei]['scalar_attribute']
            X=F.node[nd]['scalar_attribute']
            Xnei=uu/len(nei)

            if X <= 1/3. and Xnei <= 1/2.:  #1/3.
                uX=(sa1*Xnei)+(1-sa1)*X
            elif 1/3. < X <= 2/3.:  #and 1/3. < Xnei <= 2/3.: 
                uX=(sa1*Xnei)+(1-sa1)*X
            elif X > 2/3. and Xnei > 1/2.:  #2/3.
                uX=(sa1*Xnei)+(1-sa1)*X

            # elif X <= 1/3. and 1/3. < Xnei <= 2/3.:
            #     uX=(sa2*Xnei)+(1-sa2)*X
            # elif 1/3. < X <= 2/3. and Xnei <= 1/3.:
            #     uX=(sa2*Xnei)+(1-sa2)*X
            # elif 1/3. < X <= 2/3. and 2/3. < Xnei:
            #     uX=(sa2*Xnei)+(1-sa2)*X
            # elif 1/3. < Xnei <= 2/3. and 2/3. < X:
            #     uX=(sa2*Xnei)+(1-sa2)*X

            else:
                uX = X
            # print uX
            # if b1 <= abs(X - Xnei) <= b2:
            #     uX=(sa*Xnei)+(1-sa)*X
            # # else:
            # #     uX = X
            # # elif X < Xnei:
            # #     uX=0
            # # elif X > Xnei:
            # #     uX=1
            # elif X < Xnei:
            #     uX=sa*max(2.*X - Xnei,0.)+(1-sa)*X
            # elif X > Xnei:
            #     uX=sa*min(2.*X - Xnei,1.)+(1-sa)*X
            # # uX=(sa*Xnei)+(1-sa)*X
            insau=int(uX*scale)
            F.add_node(nd,scalar_attribute=uX,scalar_attribute_numeric=insau)

            
        # for i in F.nodes(data=True):
        #   print i
        # Checking for attributes equality
        y1=y
        y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        if ii ==ckck and ii<10:
            # col=['%.5f' %yy for yy in y]
            col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
            sstt="Distribution of the opinions over graph nodes at %i iterations" %(ii+1)

            plt.subplot(3,2,kckc).set_title(sstt)
            nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
            plt.axis('equal')
            plt.axis('off')
            ckck+=5
            kckc+=1
            plt.subplot(3,2,5)
        # for yy in it.combinations(y,2):
        #   checkin=checkin and yy[0] -yy[1] <(1./scale)
        # for yy in range(len(y1)):
        #   # print y[yy]-y1[yy]<(1./scale)
        #   checkin=checkin and y[yy]-y1[yy]<(0.1/(scale))
        #   # print checkin
        # if checkin:
        #   break
        asss=nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        if np.isnan(asss):
            asss=1.
        # print 'Iteration %i ==> %f' %(ii,asss),y 
        # print type(asss), asss<-1,asss>1
        iterat.append(ii)

        assort.append(asss)

        # print nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')
        # y=[F.node[i]['scalar_attribute'] for i in F.nodes()]
        plt.plot(F.nodes(),y,'-o')
        # if asss==1. :
        #   break
    momo1=[1./3 for nd in F.nodes()]
    momo2=[2./3 for nd in F.nodes()]
    plt.plot(F.nodes(),y,linewidth=3.)
    plt.plot(F.nodes(),momo1,linewidth=2.,color='r')
    plt.plot(F.nodes(),momo2,linewidth=2.,color='r')
    sstt= "Time variation of the opinion assortativity coefficient"

    plt.subplot(3,2,6).set_title(sstt)

    # plt.figure(3)

    plt.plot(iterat,assort)
    plt.ylim(-1.1,1.1)
    # plt.figure(3)
    # plt.plot(F.nodes(),y)
    # col=['%.5f' %yy for yy in y]
    col=[F.node[i]['scalar_attribute'] for i in F.nodes()]
    sstt="Distribution of the opinions over graph nodes at iterations" #%i  ss\n (consensual attribute = %s)" %(iterations,col[0])
    plt.subplot(3,2,4).set_title(sstt)

    # plt.figure(4)

    
    # print col
    # print [F.node[i]['scalar_attribute_numeric'] for i in F.nodes()]
    # pos=nx.spring_layout(G)
    ##nx.draw_networkx(g,pos=pos, node_color=col,cmap=plot.cm.Reds)
    nx.draw_networkx(G=G,pos=pos, node_color=col,vmin=0.,vmax=1.)
    plt.axis('equal')
    plt.axis('off')

    plt.show()





# influence_sim(25,0.2,.1,500)
# infdif_sim(25,.2,0.,.1,500)
# polinf_sim(25,.2,500)
# sidif_sim(25,.2,1.,500)
# polinfluence_sim(25,0.2,.041,500)
# cl_influence_sim(25,0.2,0.9,0.1,500)