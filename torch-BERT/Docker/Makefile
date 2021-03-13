run:
	docker build -t pytorch_bert .
in:
	NV_GPU=2 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it pytorch_bert:latest bash
