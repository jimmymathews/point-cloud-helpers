
The file `point_cloud_helpers.py` contains simple code for loading a multi-variate dataset from CSV, labelling its dimensions with abstract identifiers, and manipulating the datasets by restricting subsets of the dimensions and obtaining groupings from CSV annotations.

The main data structures are the classes `Identifier`, `IdentifierGroup`, and `PointCloud`.
The main functions are

  - `PointCloud`'s `restrict_to(IdentifierGroup)`
  - `PointCloud`'s `least_variable_spatial_coordinates(N)`, `most_variable_spatial_coordinates(N)`, etc.
  - `point_cloud_from_csvs(data_file, point_ids_file, coordinate_ids_file)`
  - `point_cloud_from_csv(filename)`
  - `load_grouping(a csv file of identifier / grouping name pairs)`

'Test code' follows each main chunk.


Usage
-----

## Test restricting/subsetting and display

```
l = Identifier("1s",  "time point")
m = Identifier("2s",  "time point")
n = Identifier("5s",  "time point")
o = Identifier("10s", "time point")
p = Identifier("100s","time point")
a = Identifier("temperature",   "measurement")
b = Identifier("pressure",      "measurement")
c = Identifier("concentration", "measurement")
d = Identifier("tastiness",     "measurement")
e = Identifier("greenness",     "measurement")

g1=IdentifierGroup([l,m,n,o,p])
g2=IdentifierGroup([a,b,c,d,e])

data = np.array([[2,3,4,5,0],[0,1,0,1,0],[3,2,3,4,5],[11,12,13,14,15],[0,-1,-1,-1,-1]])
pc = PointCloud(g1,g2,data)
pc.coordinate_ids.show()
pc.point_ids.show()
print(pc.data)

pc2 = pc.restrict_to(IdentifierGroup([m,n]))
pc2.point_ids.show()
print(pc3.data)

pc.most_variable_spatial_coordinates(3).show()
```

## Test loading

```
pc = point_cloud_from_csv("test_data/full_data_set.csv", "samples", "properties")
pc.data
pc.point_ids.show()
pc.coordinate_ids.show()

pc = point_cloud_from_csvs("test_data/raw_values.csv","test_data/labels1.csv","test_data/labels2.csv")
pc2=pc.restrict_to(IdentifierGroup([Identifier("S4","labels2"),Identifier("S5","labels2")]))
pc2.point_ids.show()
pc2.coordinate_ids.show()
```

## Test loading grouping/clustering

```
a, grouping = load_grouping("test_data/clusters.csv", "samples")
print("Whole list:")
a.show()
print("")
print("Each group separately:")
for g in grouping:
    g.show()
    print("")
```



