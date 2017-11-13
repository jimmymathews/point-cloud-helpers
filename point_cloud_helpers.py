# -*- coding: UTF-8 -*-
import numpy as np
import colorama
colorama.init(autoreset=True)

'''
This file contains simple code for loading a multi-variate dataset from CSV, labelling
its dimensions with abstract identifiers, and manipulating the datasets by restricting
subsets of the dimensions and obtaining groupings from CSV annotations.

The main data structures are the classes Identifier, IdentifierGroup, and PointCloud.
The main functions are
  - PointCloud's restrict_to(IdentifierGroup)
  - PointCloud's least_variable_spatial_coordinates(N), most_variable_spatial_coordinates(N), etc.
  - point_cloud_from_csvs(data_file, point_ids_file, coordinate_ids_file)
  - point_cloud_from_csv(filename)
  - load_grouping(a csv file of identifier / grouping name pairs)
'Test code' follows each main chunk.
'''

class Identifier:
    '''
    An Identifier encapsulates a "dimension" of a data set. For example, a "case" or "sample" ID number,
    or the name of a property being measured. It mainly stores a string (self.label) describing this dimension.
    '''
    def __init__(self, label, type_name):
        '''
        Only 'self.type_name' requires more explanation. This is intended to be used as a general designation,
        all Identifiers appearing in the same list of rows or columns of a data set having the same type_name.
        The annotations, in contrast, are not expected to exhibit this regularity (annotation strings will
        be used to indicate groupings).
        '''
        self.label = label
        self.type_name = type_name
        self.annotations = []
    
    def equals(self,other):
        return (self.label == other.label)

    def add_annotation(self, ann):
        self.annotations.append(ann)

    def to_string(self):
        a = ""
        for annotation in self.annotations:
            a = a + annotation + " "
        return "("+self.type_name+") " + self.label + " " + colorama.Fore.GREEN + a + colorama.Style.RESET_ALL

    def show(self):
        print(self.to_string())


class IdentifierGroup:
    '''
    An IdentifierGroup encapsulates a collection of dimensions of a point-cloud-type data set.
    It mainly stores the Identifiers of the rows (or columns) of some unspecified data set. 
    '''
    def __init__(self, ids):
        '''
        ids should be a list of Identifiers.
        '''
        self.ids = ids

    def __getitem__(self, key):
        return self.ids[key]

    def type_name(self):
        return self.ids[0].type_name

    def show(self):
        for i in self.ids:
            i.show()

    def add(self, identifier):
        if(not self.contains(identifier)):
            self.ids.append(identifier)

    def contains(self, other_id):
        for identifier in self.ids:
            if(identifier.equals(other_id)):
                return True
        return False

    def contains_subset(self, other_group):
        for other_id in other_group.ids:
            if(not self.contains(other_id)):
                return False
        return True

    def intersect(self, other_group):
        intersection = []
        for identifier in self.ids:
            if(other_group.contains(identifier)):
                intersection.append(identifier)
        return IdentifierGroup(intersection)

    def union(self, other_group):
        union = IdentifierGroup(self.ids)
        for identifier in other_group.ids:
            union.add(identifier)
        return union

    def complement(self, other_group):
        complement = []
        for identifier in self.ids:
            if(not other_group.contains(identifier)):
                complement.append(identifier)
        return IdentifierGroup(complement)


class PointCloud:
    '''
    A PointCloud has a raw matrix of numerical values, and two groups of identifiers ('IdentifierGroup's)
    describing the contextual meaning of the rows and columns. The class's main function is to reduce
    dimensionality by restricting to subsets of identifiers and returning a new PointCloud.
    '''
    def __init__(self, identifiers1, identifiers2, data, name = "pointcloud", quiet = False):
        self.point_ids = identifiers1
        self.coordinate_ids = identifiers2
        self.data = data
        self.quiet = quiet
        if(self.number_of_points() != len(self.point_ids.ids)):
            print("Warning! Number of points is "+str(self.number_of_points())+" but you provided "+str(len(self.point_ids.ids))+" IDs for them.")
        if(self.number_of_dimensions() != len(self.coordinate_ids.ids)):
            print("Warning! Number of coordinates is "+str(self.number_of_dimensions())+" but you provided "+str(len(self.coordinate_ids.ids))+" IDs for them.")

    def be_quiet(self):
        self.quiet = True

    def has_no_data(self):
        if(len(self.data) == 0):
            return True
        else:
            return False

    def number_of_points(self):
        return len(self.data)

    def number_of_dimensions(self):
        return len(self.data[0])

    def scale_normalize(self):
        scales = self.scales()
        self.scale_by(scales)

    def scales(self):
        scales = [(1.0/np.std(abs(self.data[:,j]))) for j in range(0, len(self.data[0]))]
        return scales

    def scale_by(self,ss):
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[0])):
                self.data[i,j] = self.data[i,j] * ss[j]        

    def get_max(self):
        return self.data.max()

    def get_data_of(self, identifier):
        '''
        Gets the row or column matching a given identifier, returning a new PointCloud (consisting either
        of one point, or of several points of dimension 1).
        It assumes that either the data types of the 2 axes of dimensions are distinct, or, if the same,
        that the data itself is symmetric. If you want truly asymmetric behavior, write yer own "get_row_of" and "get_column_of" functions!
        '''
        focus_identifiers = []
        unmodified_identifiers = []
        modifying = ""
        if(self.point_ids.type_name() == identifier.type_name):
            focus_identifiers      = self.point_ids
            unmodified_identifiers = self.coordinate_ids
            modifying = "points"
        if(self.coordinate_ids.type_name() == identifier.type_name):
            focus_identifiers      = self.coordinate_ids
            unmodified_identifiers = self.point_ids
            modifying = "coordinates"
        
        vector = []
        if(modifying == "points"):
            for i in range(0, len(focus_identifiers.ids)):
                if(focus_identifiers.ids[i].equals(identifier)):
                    vector = [self.data[i]]
        if(modifying == "coordinates"):
            for j in range(0, len(focus_identifiers.ids)):
                if(focus_identifiers.ids[j].equals(identifier)):
                    for i in range(0, len(self.data)):
                        vector.append([self.data[i,j]])
                    continue
        nparray = np.array(vector)

        pc = []
        if(modifying == "points"):
            pc = PointCloud(IdentifierGroup([identifier]),unmodified_identifiers,nparray, quiet = self.quiet)
        if(modifying == "coordinates"):
            pc = PointCloud(unmodified_identifiers,IdentifierGroup([identifier]),nparray, quiet = self.quiet)
        return pc

    def concatenate_rows(self, pc):
        #Assumes that column identifiers are equal.
        self.data = np.concatenate((self.data,pc.data),0)
        self.point_ids.ids = self.point_ids.ids + pc.point_ids.ids
 
    def concatenate_cols(self, pc):
        #Assumes that row identifiers are equal.
        self.data = np.concatenate((self.data,pc.data),1)
        self.coordinate_ids.ids = self.coordinate_ids.ids + pc.coordinate_ids.ids
 
    def restrict_to(self, subset):
        #Return PointCloud's data is ordered in same way as the ids of subset.
        if((not self.point_ids[0].type_name == subset[0].type_name) and
           (not self.coordinate_ids[0].type_name == subset[0].type_name) ):
            print("Data set doesn't have dimensions of type " + subset[0].type_name + "... *"+self.point_ids[0].type_name+","+self.coordinate_ids[0].type_name +"*")
            return
        identifiers_to_modify = []
        modifying = ""
        if(self.point_ids.type_name() == subset.type_name()):
            identifiers_to_modify = self.point_ids
            modifying = "points"
        if(self.coordinate_ids.type_name() == subset.type_name()):
            identifiers_to_modify = self.coordinate_ids
            modifying = "coordinates"

        intersection = subset
        if(not identifiers_to_modify.contains_subset(subset)):
            intersection = subset.intersect(identifiers_to_modify)
            if(not self.quiet):
                print("Mild warning: The subset you are restricting to is not contained in the set you are restricting. Variable order may vary accordingly.")
                print("Using intersection instead, of size "+str(len(intersection.ids))+ " instead of what would be a subset of size "+str(len(subset.ids)))

        if(len(intersection.ids)==0):
            print("Error, no intersection.")
            return

        if(modifying == "points"):
            iterator = iter(intersection.ids)
            new_pc = self.get_data_of(next(iterator))
            for additional_label in iterator:
                new_pc.concatenate_rows(self.get_data_of(additional_label))
        if(modifying == "coordinates"):
            iterator = iter(intersection.ids)
            new_pc = self.get_data_of(next(iterator))
            for additional_label in iterator:
                new_pc.concatenate_cols(self.get_data_of(additional_label))
        return new_pc

    def first_name(self):
        a = self.point_ids[0].annotations
        if(len(a)>0):
            return str(a[0])

    def grab_group_labels(self, grouping):
        '''
        Imports the group names appearing in the IdentifierGroups of grouping into the Identifiers
        of this PointCloud, self.
        '''
        point_ids = self.point_ids
        for group in grouping:
            group_name = group.ids[0].annotations[0]
            for point_id in point_ids:
                if( (len(point_id.annotations)==0 or point_id.annotations[0] != group_name)
                        and group.contains(point_id)  ):
                    point_id.annotations = []
                    point_id.add_annotation(group_name)
                if( len(point_id.annotations)==0):
                    point_id.annotations = ["None specified"]

    def calculate_spatial_variances(self):
        self.variances = np.var(self.data, axis = 0)

    def calculate_spatial_means(self):
        self.means = np.mean(self.data, axis = 0)

    def key_func(self, item):
        return item[1]

    def most_variable_spatial_coordinates(self, N):
        '''
        Returns an IdentifierGroup containing the top N coordinates in terms of the ordinary variance
        across the point cloud.
        '''
        self.calculate_spatial_variances()
        indexed_ids = []
        l = self.coordinate_ids.ids
        for i in range(0,len(l)):
            indexed_ids.append([l[i], self.variances[i]])
        sorted_list = sorted(indexed_ids, key = self.key_func, reverse=True)
        if(N>len(sorted_list)):
            print("Asked for too many coordinates in restricting to most highly variable coordinates, "+str(N)+"; Just going to use the whole set of " + str(len(sorted_list)))
            N = len(sorted_list)
        new_ids = [label[0] for label in sorted_list[:N]]
        for label in sorted_list[:N]:
            s = label[0].to_string() + "   "+str(label[1])
        return IdentifierGroup(new_ids)

    def least_variable_spatial_coordinates(self, N):
        '''
        Returns an IdentifierGroup containing the top N coordinates in terms of the ordinary variance
        across the point cloud.
        '''
        self.calculate_spatial_variances()
        indexed_ids = []
        l = self.coordinate_ids.ids
        for i in range(0,len(l)):
            indexed_ids.append([l[i], self.variances[i]])
        sorted_list = sorted(indexed_ids, key = self.key_func)
        if(N>len(sorted_list)):
            print("Asked for too many coordinates in restricting to least variable coordinates, "+str(N)+"; Just going to use the whole set of " + str(len(sorted_list)))
            N = len(sorted_list)
        new_ids = [label[0] for label in sorted_list[:N]]
        for label in sorted_list[:N]:
            s = label[0].to_string() + "   "+str(label[1])
        return IdentifierGroup(new_ids)

    def highest_valued_coordinates(self, N):
        '''
        Returns an IdentifierGroup containing the top N coordinates in terms of the (mean of the) values
        of the coordinates themselves across the point cloud.
        '''
        self.calculate_spatial_means()
        indexed_ids = []
        l = self.coordinate_ids.ids
        for i in range(0,len(l)):
            indexed_ids.append([l[i], self.means[i]])
        sorted_list = sorted(indexed_ids, key = self.key_func, reverse=True)
        if(N>len(sorted_list)):
            print("Asked for too many coordinates in restricting to most highly valued coordinates, "+str(N)+"; Just going to use the whole set of " + str(len(sorted_list)))
            N = len(sorted_list)
        new_ids = [label[0] for label in sorted_list[:N]]
        for label in sorted_list[:N]:
            s = label[0].to_string() + "   "+str(label[1])
        return IdentifierGroup(new_ids)

    def lowest_valued_coordinates(self, N):
        '''
        Returns an IdentifierGroup containing the top N coordinates in terms of the (mean of the) values
        of the coordinates themselves across the point cloud.
        '''
        self.calculate_spatial_means()
        indexed_ids = []
        l = self.coordinate_ids.ids
        for i in range(0,len(l)):
            indexed_ids.append([l[i], self.means[i]])
        sorted_list = sorted(indexed_ids, key = self.key_func)
        if(N>len(sorted_list)):
            print("Asked for too many coordinates in restricting to most highly valued coordinates, "+str(N)+"; Just going to use the whole set of " + str(len(sorted_list)))
            N = len(sorted_list)
        new_ids = [label[0] for label in sorted_list[:N]]
        for label in sorted_list[:N]:
            s = label[0].to_string() + "   "+str(label[1])
        return IdentifierGroup(new_ids)

# Test code

# a = Identifier("CRP","gene")
# b = Identifier("IGF1R","gene")
# c = Identifier("ACT1","gene")
# d = Identifier("CASQ1","gene")
# e = Identifier("CASQ2","gene")
# l = Identifier("s1","sample")
# m = Identifier("s2","sample")
# n = Identifier("s3","sample")
# o = Identifier("s4","sample")
# p = Identifier("s5","sample")
# g1=IdentifierGroup([l,m,n,o,p])
# g2=IdentifierGroup([a,b,c,d,e])
# data = np.array([[2,3,4,5,0],[0,1,0,1,0],[3,2,3,4,5],[11,12,13,14,15],[0,-1,-1,-1,-1]])
# pc = PointCloud(g1,g2,data)
# pc2 = pc.restrict_to(IdentifierGroup([a,b,e]))
# pc3 = pc.restrict_to(IdentifierGroup([m,n]))
# pc.coordinate_ids.show()
# pc.point_ids.show()
# pc3.point_ids.show()
# print(pc.data)
# print(pc3.data)
# pc.most_variable_spatial_coordinates(3).show()


import csv
import re

def main_name(filename):
    '''
    Gets the extension-less version of the filename.
    '''
    return re.compile("[\\w\\d]+/?([\\w\\d]+)\.[\\w\\d]+").match(filename).group(1)

def labels_from_csv(filename):
    labels = []

    num_rows = 0
    num_cols = 0

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            num_rows= num_rows+1
            num_cols = len(row)

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        if(num_rows == 1):
            for row in reader:
                labels = row
        else:
            if(num_cols == 1):
                for row in reader:
                    labels.append(row[0])
    if(len(labels)==0):
        print(filename+" not a single row or column")
        return
    identifiers = []
    for label in labels:
        identifiers.append(Identifier(label, main_name(filename)));
    return IdentifierGroup(identifiers)

def point_cloud_from_csvs(data_file, names1_file, names2_file):
    g1 = labels_from_csv(names1_file)
    g2 = labels_from_csv(names2_file)

    values = raw_values_from_csv(data_file)
    numerical_values = [[] for i in range(len(values))]
    for i in range(0,len(values)):
        for j in range(0,len(values[0])):
            numerical_values[i].append(float(values[i][j]))
    npa = np.array(numerical_values)
    return PointCloud(g1,g2,npa)

def raw_values_from_csv(filename):
    values = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            values.append(row)
    return values


def point_cloud_from_csv(filename, type1, type2):
    values = raw_values_from_csv(filename)
    labels1 = [values[i][0] for i in range(1,len(values))]
    labels2 = [values[0][i] for i in range(1,len(values[0]))]    
    numerical_values = [[] for i in range(len(values)-1)]
    for i in range(1,len(values)):
        for j in range(1,len(values[0])):
            numerical_values[i-1].append(float(values[i][j]))
    npa = np.array(numerical_values)
    
    identifiers1 = []
    for label in labels1:
        identifiers1.append(Identifier(label, type1))
    g1 = IdentifierGroup(identifiers1)

    identifiers2 = []
    for label in labels2:
        identifiers2.append(Identifier(label, type2))
    g2 = IdentifierGroup(identifiers2)

    return PointCloud(g1,g2,npa)


def point_cloud_to_csv(pc, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        header = [""]
        for identifier in pc.coordinate_ids.ids:
            header.append(identifier.label)
        writer.writerow(header)
        point_ids = pc.point_ids.ids
        for i in range(0,len(point_ids)):
            row = [point_ids[i].label]
            for j in range(0,len(pc.data[0])):
                row.append(pc.data[i,j])
            writer.writerow(row)

def point_cloud_to_csv_values_only(pc, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        # header = [""]
        # for identifier in pc.coordinate_ids.ids:
        #     header.append(identifier.label)
        # writer.writerow(header)
        point_ids = pc.point_ids.ids
        for i in range(0,len(point_ids)):
            # row = [point_ids[i].label]
            row = []
            for j in range(0,len(pc.data[0])):
                row.append(pc.data[i,j])
            writer.writerow(row)

# Test code

# pc = point_cloud_from_csv("test_data/full_data_set.csv", "samples", "genes")
# pc.data
# pc.point_ids.show()
# pc.coordinate_ids.show()

# pc = point_cloud_from_csvs("test_data/raw_values.csv","test_data/labels1.csv","test_data/labels2.csv")
# pc2=pc.restrict_to(IdentifierGroup([Identifier("S4","labels2"),Identifier("S5","labels2")]))
# pc2.point_ids.show()
# pc2.coordinate_ids.show()


def contains(label, l):
    for other in l:
        if(label == other):
            return True
    return False

def load_grouping(filename, type_name):
    '''
    Returns a list of IdentifierGroups, one for each among the group names across all label / group name
    pairs in the CSV file filename.
    '''
    values = raw_values_from_csv(filename)
    number_of_points = len(values)
    number_of_columns = len(values[0])
    if(number_of_columns != 2):
        print("Need one column of identifiers and one column of grouping names.")
        return

    ids = []
    annotations = []
    labels = []

    for row in values:
        ids.append(row[0])
        annotations.append(row[1])
        if(not contains(row[1],labels)):
            labels.append(row[1])

    ga = [Identifier(identifier, type_name) for identifier in ids]
    for i, annotation in enumerate(annotations):
        ga[i].add_annotation(annotation)
    group_all = IdentifierGroup(ga)

    identifiers_groups = []
    for label in labels:
        g = []
        for i in ga:
            if(i.annotations[0] == label):
                g.append(i)
        identifiers_groups.append(g)
    return [group_all, [IdentifierGroup(g) for g in identifiers_groups]]

# Test code

# a, grouping = load_grouping("test_data/clusters.csv", "samples")
# print("Whole list:")
# a.show()
# print("")
# print("Each group separately:")
# for g in grouping:
#     g.show()
#     print("")
