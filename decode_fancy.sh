#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counting algal hits
echo -e "${GREEN}Counting algal hits...${NC}"
fgrep -c '@' eval-results_10eval-*

echo -e "\n${GREEN}Counting bacterial hits...${NC}"
fgrep -c '!' eval-results_10eval-*

echo -e "\n${YELLOW}In other words, from the algal holdout set there are:${NC}"
count_algae=$(fgrep -c '@' eval-results_10eval-algae.txt)
echo -e "${GREEN}${count_algae} algal signatures.${NC}"

count_bacteria=$(fgrep -c '!' eval-results_10eval-algae.txt)
echo -e "${RED}${count_bacteria} bacterial signatures.${NC}"

echo -e "\n${YELLOW}And from the bacterial holdout set, there are:${NC}"
#count_bacteria_bact=$(fgrep -c '!' eval-results_10eval-bact.txt)
#echo -e "${RED}${count_bacteria_bact} bacterial signatures.${NC}"

count_algae_bact=$(fgrep -c '@' eval-results_10eval-bact.txt)
echo -e "${GREEN}${count_algae_bact} algal signatures.${NC}"

count_bacteria_bact=$(fgrep -c '!' eval-results_10eval-bact.txt)
echo -e "${RED}${count_bacteria_bact} bacterial signatures.${NC}"
