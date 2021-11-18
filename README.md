# dat640-smart-task

## Instruction for running run this project

1. Download [short_abstracts_en.ttl](http://downloads.dbpedia.org/2016-10/core-i18n/en/short_abstracts_en.ttl.bz2) and
   [instance_types_en.ttl](http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_en.ttl.bz2).
2. Extract these two files in the directory: `advanced-type-prediction/data`.
3. Run `advanced-type-prediction/util/run_this_file_first.ipynb`. (This will generate the needed files and index 3 datasets into elasticsearch. )
4. to be continue....

## Possible solutions for errors during runing the project

1. Make sure to use 64 bits python.
2. Update the python elasticsearch liabrary to a newer version.
