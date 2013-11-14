#Get_thompson_motif
==================

# Summary

In this repository,my script fetches the thompson's motif index html page. After that,it parses fethed html files and arranges following thompson's index number, and finally constructs tree structure. This script write out constructed tree structure into json format file. Thus, Any script language utilize it.  
The reason why I wrote this script is to arrange thompson's motif index number. As I know, no one copes with any similar work. 

# Outline of scrits
* ./curl_thomson_index_page.sh:fethes html files from [Thompson's motif index page](http://www.ualberta.ca/~urban/Projects/English/Motif_Index.htm)
* ./parse.py:parses and constructs tree structure, writes out json format file into parsed_json
* ./parsed_json/:output json format file is saved under this directory
* ./htmls/:fethed files are saved under this diretory

# Usage 

First, make directory named 'htmls/' in this directory(use ````mkdir htmls````).  
Second, execute curl_thomson_index_page.sh  
Third, you can parse and write out json files from fethed html files. If you want to parse all files in htmls/, execute following commands
````
for f in ./htmls/*
do
python parse.py $f
done
````

# File format

I'll write this section later.
