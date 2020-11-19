# hopfield


    DESCRIPTION: 
        Basic Hopfield network model 
    USAGE 1: 
        python -u hop.py --input_seqs=intputfile.json --test_seq=testseq.json --random --N=100 --M=10 
        if --random, input is ignored and M seq of length N are randomly generated and stored.
        inputfile must be a multijson file with 1 seq for each line. 
        testseq is a json with a sequence to be tested. 
    USAGE 2: 
    Finite temperature scan: 
    edit ./resources/.env by setting
    pattern_length = length of randomly stored pattern
    number_patterns = number of random stored patterns
    number_iterations  = update steps toward attractors
    number_replicas = run replicas 
    plot = plot or not the order parameter (memory retrieval coefficient)
    processes = number of parallel threads
    Tmin = minimum temperature to scan
    Tmax = maximum temperature to scan
    Tby  = temperature increment from Tmin to Tmax 
    ./resources/docker-compose up --build 
    
    AUTHOR: 
        David Preti
    DATE: 
        Rome, Sep 2020 

