#!/bin/bash

if [ "${plot}" = "True" ]; then
    python hop_rep.py --N=${pattern_length} --M=${number_patterns} --iter=${number_iterations} --replicas=${number_replicas} --threads=${processes} --plot --Tmin=${Tmin} --Tmax=${Tmax} --Tby=${Tby}
else 
    python hop_rep.py --N=${pattern_length} --M=${number_patterns} --iter=${number_iterations} --replicas=${number_replicas} --threads=${processes} --Tmin=${Tmin} --Tmax=${Tmax} --Tby=${Tby}
fi 
