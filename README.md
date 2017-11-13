The file `point_cloud_helpers.py` contains simple code for loading a multi-variate dataset from CSV, labelling its dimensions with abstract identifiers, and manipulating the datasets by restricting subsets of the dimensions and obtaining groupings from CSV annotations.

The main data structures are the classes `Identifier`, `IdentifierGroup`, and `PointCloud`.
The main functions are

  - `PointCloud`'s `restrict_to(IdentifierGroup)`
  - `PointCloud`'s `least_variable_spatial_coordinates(N)`, `most_variable_spatial_coordinates(N)`, etc.
  - `point_cloud_from_csvs(data_file, point_ids_file, coordinate_ids_file)`
  - `point_cloud_from_csv(filename)`
  - `load_grouping(a csv file of identifier / grouping name pairs)`

'Test code' follows each main chunk.
