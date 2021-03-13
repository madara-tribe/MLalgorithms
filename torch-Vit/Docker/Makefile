run:
	docker build -t vit .
in:
	NV_GPU=1 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it vit:latest bash
