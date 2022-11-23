commands docer build and run are in files build.sh and run.sh

command to build:
 '''
  docker build -t katia2145/mlops_fastapi .
 '''

command to run:
 '''
  docker run -d -p 8081:80 -e MODULE_NAME="server" katia2145/mlops_fastapi
 '''

command to pull:
'''
  docker pull katia2145/mlops_fastapi
'''
