uvicorn main:app --host 0.0.0.0 --port 8080 --reload
sudo lsof -t -i:8080
sudo kill -9 $(sudo lsof -t -i:8080)
sudo netstat -lpn |grep :8080
fuser -k 8080/tcp
rasa run --enable-api -m models/nlu-20240606-165651-natural-cabinet.tar.gz