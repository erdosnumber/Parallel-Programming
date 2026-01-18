#!/bin/bash
folder="pre-proc-img/"
for file in "$folder"*-num*.txt; do
        file_number="${file%%-*}"
        num_after_dash="${file#*-num}"
        num_after_dash="${num_after_dash%.txt}"
        mv "$file" "${file_number}.txt"
	# echo "$file_number $num_after_dash" >> "actual_output.txt"
done
