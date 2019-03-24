#!/usr/bin/env python
# encoding: utf-8
import random
import networkx as nx

def create_scalar_attributes_0_1(G,source=[]):
	F=nx.Graph()
	for ed in G.edges():
		attr_dic=G.edge[ed[0]][ed[1]]
		F.add_edge(ed[0],ed[1],attr_dict=attr_dic)
	# print source
	# print len(source)
	# print aaaaa
	for nd in G.nodes():
		attr_dic=G.node[nd]
		if len(source)<2:
			if nd==source[0]:
				# print aaa
				rand=1.
				irand=1
				F.add_node(nd,attr_dict=attr_dic,scalar_attribute=rand,scalar_attribute_numeric=irand)
			else:
				rand=0.
				irand=0
				F.add_node(nd,attr_dict=attr_dic,scalar_attribute=rand,scalar_attribute_numeric=irand)
		elif len(source)==2:
			if nd==source[0]:
				rand=1.
				irand=1
				F.add_node(nd,attr_dict=attr_dic,scalar_attribute=rand,scalar_attribute_numeric=irand)
			elif nd==source[1]:
				rand=0.
				irand=0
				F.add_node(nd,attr_dict=attr_dic,scalar_attribute=rand,scalar_attribute_numeric=irand)
			else:
				rand=random.random()
				irand=int(rand)
				F.add_node(nd,attr_dict=attr_dic,scalar_attribute=rand,scalar_attribute_numeric=irand)
	# print F.nodes(data=True)
	# print aaa
	return F, nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')

def create_random_scalar_attributes(G,scale):
	F=nx.Graph()
	# print F.nodes(data=True)
	for ed in G.edges():
		attr_dic=G.edges[ed[0],ed[1]]
		F.add_edge(ed[0],ed[1],attr_dict=attr_dic)
	for nd in G.nodes():
		attr_dic=G.node[nd]
		rand=random.random()
		irand=int(rand*scale)
		F.add_node(nd,attr_dict=attr_dic,scalar_attribute=rand,scalar_attribute_numeric=irand)
	return F, nx.numeric_assortativity_coefficient(F,'scalar_attribute_numeric')

def create_random_discrete_attributes(G,k):
	F=nx.Graph()
	for ed in G.edges():
		# print ed
		attr_dic=G.edge[ed[0]][ed[1]]
		F.add_edge(ed[0],ed[1],attr_dict=attr_dic)
	range_list=range(k)
	# print F.nodes(data=True)
	for nd in G.nodes():
		attr_dic=G.node[nd]
		if len(range_list)!=0:
			raa=random.choice(range_list)
			range_list.remove(raa)
		else:
			range_list=range(k)
			raa=random.choice(range_list)
		F.add_node(nd,attr_dict=attr_dic,discrete_attribute=raa)
	return F,nx.attribute_assortativity_coefficient(F,'discrete_attribute')

# G=nx.Graph()
