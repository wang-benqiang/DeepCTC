FROM harbor.miduchina.com/yjy/algorithm/midu_dl:base

RUN apt-get update --fix-missing && apt-get install -y git
RUN echo Asia/Shanghai > /etc/localtime && echo Asia/Shanghai > /etc/timezone && apt-get install tzdata
RUN apt-get update --fix-missing && apt-get install -y vim

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY ./command /data/app/command
COPY ./model /data/app/model
COPY ./configs /data/app/configs
COPY ./pretrained_model /data/app/pretrained_model



COPY ./src /data/app/src
COPY ./logs /data/app/logs


COPY ./tests /data/app/tests
COPY ./runner /data/app/runner
# 下载模型
RUN cd /data/app/command && sh download_ctc_model.sh
WORKDIR /data/app 

EXPOSE 80

CMD cd /data/app/ && python -m runner.start_server_fastapi