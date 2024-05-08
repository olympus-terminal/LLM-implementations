echo ''
echo 'counting algal hits'
  fgrep -c \@ eval-results_10eval-*

echo ''
 
 echo 'counting bacterial hits'
  fgrep -c \! eval-results_10eval-*
 
echo ''
 
 echo '    in other words'
  echo 'from the algal holdout set there are '
 
 fgrep -c \@ eval-results_10eval-algae.txt
 
 echo 'algal signatures, and'
 
 fgrep -c \! eval-results_10eval-algae.txt
 
 echo 'bacterial signatures'
 echo ''

 echo ' and from the bacterial holdout set, there are'
 
 fgrep -c \! eval-results_10eval-bact.txt
 
 echo 'bacterial signatures, and '
 
 fgrep -c \@ eval-results_10eval-bact.txt
 
 echo 'algal signatures'
