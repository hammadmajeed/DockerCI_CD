FROM alpine
RUN mkdir webapp && mkdir my-model

ENV MODEL_DIR=/webapp/my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib
ENV MODEL_FILE_RFC=clf_rfc.joblib
WORKDIR /webapp
COPY requirement.txt ./requirement.txt
COPY Makefile ./Makefile
RUN make install

COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN echo "$HOME"
RUN python3 train.py && python3 inference.py
ENTRYPOINT ["/bin/bash"]
