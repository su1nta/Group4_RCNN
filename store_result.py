import csv

def store_result(output):   
    # output is a list which contains a table consists of 
    #[frame_no, object_no, xmin, ymin, xmax, ymax, id]

    with open('output.csv', 'a') as output_file:
        output_file.write("\n")
        output_writer = csv.writer(output_file)
        for row in output:
            output_writer.writerow(row)

    with open('output.csv', 'r') as output_file_read:
        output_reader = csv.reader(output_file_read)
        for line in output_reader:
            print(line)
