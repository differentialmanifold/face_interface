FROM tensorflow-opencv-dlib

WORKDIR /app

COPY . /app

RUN pip3 install --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple -r requirements.txt

EXPOSE 5001

CMD ["python3", "src/app.py"]