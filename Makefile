.PHONY: docker dockerclean

docker:
	docker build --force-rm=true --shm-size=512MB --tag=pyautoplot:latest docker

dockerclean:
	-docker rm -v $(docker ps -a -q -f status=exited);docker rmi $(docker images -f "dangling=true" -q)
